"""Evaluation protocols.

Primary: nested leave-one-subject-out (LOSO), as described in the manuscript.
  For each held-out participant:
    1. for each remaining participant used as an inner validation user, train on
       the other participants and record the best epoch by validation weighted
       F1 (<= 60 epochs, early stopping patience 15);
    2. set the outer-fold epoch budget to the median of those best epochs;
    3. retrain on all non-test participants for that fixed budget;
    4. predict the held-out participant.

"""

import copy
import hashlib
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from augmentations import augment
from data import chronological_split, loso_folds, standardize, xy
from metrics import all_metrics, basic
from model import TinyStressNet
from robustness import probe as robustness_probe, summarize_folds

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# Reproducible seeding
# --------------------------------------------------------------------------


def derive(master: int, *parts) -> int:
    """Deterministically derive a 31-bit sub-seed from a master seed + context."""
    key = "|".join([str(master)] + [str(p) for p in parts]).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big") & 0x7FFFFFFF


def set_seed(seed: int) -> None:
    """Seed every RNG the training path touches."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_seed(master: int, scheme: str, *parts) -> int:
    """Seed to hand to a single fit call."""
    if scheme == "legacy":
        return master
    if scheme == "nested":
        return derive(master, *parts)
    raise ValueError(f"unknown seed scheme: {scheme!r} (use 'legacy' or 'nested')")


DEFAULT_SEEDS = tuple(range(10))  # fix the list in advance; report all of them


class Frame(Dataset):
    def __init__(self, X_std, y, condition, do_aug):
        self.X = torch.tensor(X_std, dtype=torch.float32).unsqueeze(-1)  # (N,3,1)
        self.y = torch.tensor(y, dtype=torch.long)
        self.condition, self.do_aug = condition, do_aug

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        if self.do_aug and self.condition != "none":
            x = augment(x, self.condition)
        return x, self.y[i]


def _loader(X_std, y, condition, do_aug, batch=32):
    return DataLoader(Frame(X_std, y, condition, do_aug), batch_size=batch, shuffle=do_aug)


@torch.no_grad()
def _predict(model, X_std):
    model.eval()
    X = torch.tensor(X_std, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    return model(X).argmax(1).cpu().numpy()


def _optim(model):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=5, factor=0.5)
    return opt, sched


def _run_epoch(model, loader, opt, loss_fn):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        loss_fn(model(xb), yb).backward()
        opt.step()


def fit_early_stop(loader, val_X, val_y, max_epochs, patience=15, seed=42):
    set_seed(seed)
    model = TinyStressNet().to(DEVICE)
    opt, sched = _optim(model)
    loss_fn = nn.CrossEntropyLoss()
    best_f1, best_epoch, best_state, waited = -1.0, 0, None, 0
    for epoch in range(1, max_epochs + 1):
        _run_epoch(model, loader, opt, loss_fn)
        f1 = basic(val_y, _predict(model, val_X))["weighted_f1"]
        sched.step(f1)
        if f1 > best_f1 + 1e-4:
            best_f1, best_epoch = f1, epoch
            best_state = copy.deepcopy(model.state_dict())
            waited = 0
        else:
            waited += 1
            if waited >= patience:
                break
    return best_epoch, best_state


def fit_fixed(loader, epochs, seed=42):
    set_seed(seed)
    model = TinyStressNet().to(DEVICE)
    opt, _ = _optim(model)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(max(1, epochs)):
        _run_epoch(model, loader, opt, loss_fn)
    return model


def loso_tinystressnet(df, condition, max_epochs=60, seed=42,
                       seed_scheme="legacy", probe_draws=0):
    """Nested LOSO.

    seed_scheme : "legacy" reproduces the submitted manuscript (every fit reset
                  to the same seed); "nested" gives each fit its own derived
                  seed. See seeding.py.
    probe_draws : if > 0, run the Section 3.10 scale-offset robustness probe on
                  each outer fold with this many draws (the manuscript uses 30).
    """
    pt, pp, per_user, foldwise, probes = [], [], {}, [], []
    for fold_idx, (test_user, train_df, test_df) in enumerate(loso_folds(df)):
        outer_users = sorted(train_df["user_id"].unique())
        best_epochs = []
        for inner_idx, val_user in enumerate(outer_users):
            inner = train_df[train_df["user_id"] != val_user]
            val = train_df[train_df["user_id"] == val_user]
            Xtr, ytr = xy(inner)
            Xva, yva = xy(val)
            _, Xtr_s, Xva_s = standardize(Xtr, Xva)
            s_in = fit_seed(seed, seed_scheme, condition, "inner", fold_idx, inner_idx)
            be, _ = fit_early_stop(_loader(Xtr_s, ytr, condition, True), Xva_s, yva,
                                   max_epochs, seed=s_in)
            best_epochs.append(be)
        chosen = int(np.median(best_epochs))
        Xtr, ytr = xy(train_df)
        Xte, yte = xy(test_df)
        _, Xtr_s, Xte_s = standardize(Xtr, Xte)
        s_out = fit_seed(seed, seed_scheme, condition, "outer", fold_idx)
        model = fit_fixed(_loader(Xtr_s, ytr, condition, True), chosen, seed=s_out)
        pred = _predict(model, Xte_s)
        pt.extend(yte.tolist()); pp.extend(pred.tolist())
        per_user[test_user] = all_metrics(yte, pred)
        foldwise.append(basic(yte, pred)["weighted_f1"])
        if probe_draws:
            probes.append(robustness_probe(lambda Z: _predict(model, Z), Xte_s, yte,
                                           n_draws=probe_draws,
                                           seed=fit_seed(seed, seed_scheme, condition,
                                                         "probe", fold_idx)))
    out = _pack(pt, pp, per_user, foldwise)
    if probe_draws:
        out["robustness"] = summarize_folds(probes)
        out["robustness_per_fold"] = probes
    return out


def chronological_tinystressnet(df, condition, max_epochs=80, seed=42,
                                seed_scheme="legacy"):
    train, val, test = chronological_split(df)
    Xtr, ytr = xy(train); Xva, yva = xy(val); Xte, yte = xy(test)
    _, Xtr_s, Xva_s, Xte_s = standardize(Xtr, Xva, Xte)
    s_ch = fit_seed(seed, seed_scheme, condition, "chrono")
    _, best_state = fit_early_stop(_loader(Xtr_s, ytr, condition, True), Xva_s, yva,
                                   max_epochs, seed=s_ch)
    model = TinyStressNet().to(DEVICE)
    model.load_state_dict(best_state)
    pred = _predict(model, Xte_s)
    return all_metrics(yte, pred), (yte, pred)


def _pack(pt, pp, per_user, foldwise):
    agg = all_metrics(np.array(pt), np.array(pp))
    agg["foldwise_weighted_f1_mean"] = float(np.mean(foldwise))
    agg["foldwise_weighted_f1_std"] = float(np.std(foldwise))
    return {"aggregate": agg, "per_user": per_user,
            "foldwise_weighted_f1": foldwise, "pooled": (np.array(pt), np.array(pp))}


# --------------------------------------------------------------------------
# Classical baselines under the same two protocols
# --------------------------------------------------------------------------
def loso_baseline(df, factory):
    pt, pp, per_user, foldwise = [], [], {}, []
    for test_user, train_df, test_df in loso_folds(df):
        Xtr, ytr = xy(train_df)
        Xte, yte = xy(test_df)
        _, Xtr_s, Xte_s = standardize(Xtr, Xte)
        pred = np.asarray(factory().fit(Xtr_s, ytr).predict(Xte_s))
        pt.extend(yte.tolist()); pp.extend(pred.tolist())
        per_user[test_user] = all_metrics(yte, pred)
        foldwise.append(basic(yte, pred)["weighted_f1"])
    return _pack(pt, pp, per_user, foldwise)


def chronological_baseline(df, factory):
    train, _, test = chronological_split(df)
    Xtr, ytr = xy(train)
    Xte, yte = xy(test)
    _, Xtr_s, Xte_s = standardize(Xtr, Xte)
    pred = np.asarray(factory().fit(Xtr_s, ytr).predict(Xte_s))
    return all_metrics(yte, pred), (yte, pred)


def subject_macro_ci(per_user, key="weighted_f1", alpha=0.05):
    vals = np.array([m[key] for m in per_user.values()])
    n = len(vals)
    if n < 2:
        return float(vals.mean()), (float(vals.mean()), float(vals.mean()))
    from scipy import stats
    mean = vals.mean()
    h = stats.t.ppf(1 - alpha / 2, n - 1) * vals.std(ddof=1) / np.sqrt(n)
    return float(mean), (float(mean - h), float(mean + h))
