"""Multi-seed study for the revised manuscript.

Addresses Reviewer 3's point that the comparative conclusions rest on
single-seed runs, and Reviewer 3's point that Sections 3.10 and 3.11 describe
analyses that report no numbers.

Usage
-----
    python multiseed.py                          # 10 seeds, probe on, full run
    python multiseed.py --seeds 0 1 2            # fewer seeds (smoke test)
    python multiseed.py --probe-draws 0          # skip the robustness probe
    python multiseed.py --seed-scheme legacy     # reproduce the submitted numbers

Outputs (in --outdir)
---------------------
    multiseed_results.json   full nested results, rewritten after every seed
    multiseed_runs.csv       one tidy row per (seed, protocol, condition)
    multiseed_report.txt     the printed tables, saved

Design notes
------------
Partitions are identical under every seed; only initialization, dropout,
shuffling and augmentation draws vary. See seeding.py.

The classical baselines are deterministic under their fixed settings, so they
are evaluated once and carry no seed spread. This asymmetry must be stated when
the table is reported: a "+/-" on a TinyStressNet row and the absence of one on
a QDA row do not mean the same thing.

For the paired tests, foldwise weighted F1 is averaged over seeds first, and
the test is then run on the five averaged per-participant values. This removes
initialization noise from each fold but does not increase the number of paired
observations, which stays at five. The p-value floor of 0.0625 is unchanged.
"""

import argparse
import json
import os
import time

import numpy as np

import stats_tests
from baselines import OrderedLogit, ordered_logit_available, sklearn_baselines
from data import load_dataframe
from evaluate import (
    DEFAULT_SEEDS,
    chronological_baseline,
    chronological_tinystressnet,
    loso_baseline,
    loso_tinystressnet,
    subject_macro_ci,
)

CONDITIONS = ("none", "gaussian", "radial", "affine")
PRETTY = {"none": "no augmentation", "gaussian": "Gaussian jitter",
          "radial": "radial-rescaling", "affine": "BA"}


def mean_sd(vals):
    a = np.asarray(vals, float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="stress_data.csv")
    ap.add_argument("--outdir", default="results_multiseed")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--seed-scheme", default="nested", choices=["nested", "legacy"])
    ap.add_argument("--probe-draws", type=int, default=30,
                    help="draws for the Section 3.10 robustness probe; 0 disables")
    ap.add_argument("--skip-chronological", action="store_true")
    ap.add_argument("--no-resume", action="store_true",
                    help="recompute seeds even if a cached seed_<n>.json exists")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_dataframe(args.csv)
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    say(f"seeds        : {args.seeds}")
    say(f"seed scheme  : {args.seed_scheme}")
    say(f"probe draws  : {args.probe_draws}")
    say(f"observations : {len(df)}   participants: {df['user_id'].nunique()}")
    say()

    # ---------------- deterministic classical baselines (run once) -------------
    say("Classical baselines under LOSO (deterministic; no seed spread)")
    bl = sklearn_baselines()
    if ordered_logit_available():
        bl["Ordered logit"] = lambda: OrderedLogit()
    baselines = {}
    for name, factory in bl.items():
        r = loso_baseline(df, factory)
        baselines[name] = r
        a = r["aggregate"]
        say(f"  {name:22s} acc={a['accuracy']:.4f}  wF1={a['weighted_f1']:.4f}  "
            f"mF1={a['macro_f1']:.4f}")
    say()

    # ---------------- TinyStressNet across seeds -------------------------------
    runs = []          # tidy rows
    loso_store = {c: [] for c in CONDITIONS}   # per-seed aggregates
    foldwise_store = {c: [] for c in CONDITIONS}
    probe_store = {c: [] for c in CONDITIONS}
    chrono_store = {c: [] for c in CONDITIONS}
    users = None

    t0 = time.time()
    for si, seed in enumerate(args.seeds):
        cache = os.path.join(args.outdir, "seed_%d.json" % seed)
        if os.path.exists(cache) and not args.no_resume:
            with open(cache, encoding="utf-8") as fh:
                c = json.load(fh)
            users = c["users"]
            for cond in CONDITIONS:
                loso_store[cond].append(c["loso"][cond])
                foldwise_store[cond].append(c["foldwise"][cond])
                if args.probe_draws and c["probe"].get(cond):
                    probe_store[cond].append(c["probe"][cond])
                if c["chrono"].get(cond):
                    chrono_store[cond].append(c["chrono"][cond])
            runs.extend(c["runs"])
            say("--- seed %d  (cached) ---" % seed)
            continue
        say(f"--- seed {seed}  ({si + 1}/{len(args.seeds)}, "
            f"elapsed {time.time() - t0:.0f}s) ---")
        seed_cache = {"users": None, "loso": {}, "foldwise": {}, "probe": {},
                      "chrono": {}, "runs": []}
        for cond in CONDITIONS:
            r = loso_tinystressnet(df, cond, seed=seed,
                                   seed_scheme=args.seed_scheme,
                                   probe_draws=args.probe_draws)
            a = r["aggregate"]
            users = sorted(r["per_user"])
            loso_store[cond].append(a)
            foldwise_store[cond].append([r["per_user"][u]["weighted_f1"] for u in users])
            if args.probe_draws:
                probe_store[cond].append(r["robustness"])
            row = dict(seed=seed, protocol="loso", condition=cond,
                       accuracy=a["accuracy"], weighted_f1=a["weighted_f1"],
                       macro_f1=a["macro_f1"], ordinal_mae=a["ordinal_mae"],
                       qwk=a["qwk"])
            runs.append(row)
            seed_cache["users"] = users
            seed_cache["loso"][cond] = a
            seed_cache["foldwise"][cond] = foldwise_store[cond][-1]
            seed_cache["probe"][cond] = r.get("robustness")
            seed_cache["runs"].append(row)
            msg = (f"  LOSO   {PRETTY[cond]:18s} acc={a['accuracy']:.4f} "
                   f"wF1={a['weighted_f1']:.4f}")
            if args.probe_draws:
                p = r["robustness"]
                msg += (f"  | probe consistency={p['consistency_mean']:.4f} "
                        f"pert-wF1={p['perturbed_weighted_f1_mean']:.4f}")
            say(msg)

            if not args.skip_chronological:
                m, _ = chronological_tinystressnet(df, cond, seed=seed,
                                                   seed_scheme=args.seed_scheme)
                chrono_store[cond].append(m)
                row = dict(seed=seed, protocol="chronological", condition=cond,
                           accuracy=m["accuracy"], weighted_f1=m["weighted_f1"],
                           macro_f1=m["macro_f1"], ordinal_mae=m["ordinal_mae"],
                           qwk=m["qwk"])
                runs.append(row)
                seed_cache["chrono"][cond] = m
                seed_cache["runs"].append(row)

        with open(cache, "w", encoding="utf-8") as fh:
            json.dump(_clean(seed_cache), fh)
        _dump(args.outdir, runs, loso_store, chrono_store, probe_store,
              foldwise_store, baselines, users, args)

    say()
    say("=" * 78)
    say(f"TABLE 4 (revised) - LOSO, mean +/- SD over {len(args.seeds)} seeds")
    say("=" * 78)
    say(f"{'Model':28s} {'Accuracy':>16s} {'Weighted F1':>16s} {'Macro F1':>16s}")
    rows = []
    for name, r in baselines.items():
        a = r["aggregate"]
        rows.append((name, a["accuracy"], 0.0, a["weighted_f1"], 0.0,
                     a["macro_f1"], 0.0, True))
    for cond in CONDITIONS:
        am, asd = mean_sd([a["accuracy"] for a in loso_store[cond]])
        wm, wsd = mean_sd([a["weighted_f1"] for a in loso_store[cond]])
        mm, msd = mean_sd([a["macro_f1"] for a in loso_store[cond]])
        rows.append((f"TinyStressNet / {PRETTY[cond]}", am, asd, wm, wsd, mm, msd, False))
    for name, am, asd, wm, wsd, mm, msd, det in sorted(rows, key=lambda x: -x[3]):
        f = (lambda m, s: f"{m:.4f}" if det else f"{m:.4f}+/-{s:.4f}")
        say(f"{name:28s} {f(am, asd):>16s} {f(wm, wsd):>16s} {f(mm, msd):>16s}")
    say("(baselines are deterministic under their fixed settings: no seed spread)")
    say()

    if not args.skip_chronological:
        say("=" * 78)
        say("TABLE 7 (revised) - chronological split, mean +/- SD over seeds")
        say("=" * 78)
        for name, factory in bl.items():
            m, _ = chronological_baseline(df, factory)
            say(f"{name:28s} {m['accuracy']:.4f}           {m['weighted_f1']:.4f}")
        for cond in CONDITIONS:
            am, asd = mean_sd([m["accuracy"] for m in chrono_store[cond]])
            wm, wsd = mean_sd([m["weighted_f1"] for m in chrono_store[cond]])
            say(f"{'TinyStressNet / ' + PRETTY[cond]:28s} "
                f"{am:.4f}+/-{asd:.4f}  {wm:.4f}+/-{wsd:.4f}")
        say()

    # ---------------- Section 4.5 robustness numbers ---------------------------
    if args.probe_draws:
        say("=" * 78)
        say(f"SECTION 4.5 / 3.10 - robustness probe ({args.probe_draws} draws per fold)")
        say("=" * 78)
        say(f"{'Condition':22s} {'consistency':>18s} {'perturbed wF1':>18s}")
        for cond in CONDITIONS:
            cm, csd = mean_sd([p["consistency_mean"] for p in probe_store[cond]])
            pm, psd = mean_sd([p["perturbed_weighted_f1_mean"] for p in probe_store[cond]])
            say(f"{PRETTY[cond]:22s} {cm:.4f}+/-{csd:.4f}  {pm:.4f}+/-{psd:.4f}")
        say()

    # ---------------- Section 4.2 paired tests ---------------------------------
    say("=" * 78)
    say("SECTION 4.2 / 3.11 - paired foldwise tests (foldwise wF1 averaged over seeds)")
    say("=" * 78)
    best_cond = max(CONDITIONS,
                    key=lambda c: np.mean([a["weighted_f1"] for a in loso_store[c]]))
    say(f"best TinyStressNet condition by mean LOSO wF1: {PRETTY[best_cond]}")
    tiny_fold = np.mean(np.array(foldwise_store[best_cond]), axis=0)
    say(f"participants: {users}")
    say("TinyStressNet foldwise wF1 (seed-averaged): "
        + ", ".join(f"{u}={float(v):.4f}" for u, v in zip(users, tiny_fold)))
    comparisons = {}
    for name in ("QDA", "SVM (RBF)", "LDA"):
        if name not in baselines:
            continue
        base_fold = [baselines[name]["per_user"][u]["weighted_f1"] for u in users]
        c = stats_tests.compare(tiny_fold, base_fold,
                                f"TinyStressNet/{PRETTY[best_cond]}", name)
        comparisons[name] = c
        say("  " + stats_tests.format_comparison(c))
    tmean, tci = subject_macro_ci(
        {u: {"weighted_f1": v} for u, v in zip(users, tiny_fold)})
    say(f"subject-macro wF1 (TinyStressNet): {tmean:.4f} [{tci[0]:.4f}, {tci[1]:.4f}]")
    say()
    say("Reminder: with 5 participants the smallest attainable two-sided p is 0.0625")
    say("for all three tests. More seeds does not lower that floor.")

    _dump(args.outdir, runs, loso_store, chrono_store, probe_store,
          foldwise_store, baselines, users, args, comparisons)
    with open(os.path.join(args.outdir, "multiseed_report.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    say()
    say(f"Wrote {args.outdir}/multiseed_results.json, multiseed_runs.csv, "
        f"multiseed_report.txt")
    say(f"Total time: {time.time() - t0:.0f}s")


def _dump(outdir, runs, loso, chrono, probe, foldwise, baselines, users, args,
          comparisons=None):
    payload = {
        "config": {"seeds": args.seeds, "seed_scheme": args.seed_scheme,
                   "probe_draws": args.probe_draws, "csv": args.csv},
        "participants": users,
        "baselines_loso": {k: v["aggregate"] for k, v in baselines.items()},
        "baselines_foldwise": {
            k: ([v["per_user"][u]["weighted_f1"] for u in users] if users else [])
            for k, v in baselines.items()},
        "tinystressnet_loso_by_seed": {c: loso[c] for c in loso},
        "tinystressnet_foldwise_by_seed": {c: foldwise[c] for c in foldwise},
        "tinystressnet_chronological_by_seed": {c: chrono[c] for c in chrono},
        "robustness_by_seed": {c: probe[c] for c in probe},
    }
    if comparisons:
        payload["paired_tests"] = comparisons
    with open(os.path.join(outdir, "multiseed_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(_clean(payload), f, indent=2)
    import csv
    if runs:
        with open(os.path.join(outdir, "multiseed_runs.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(runs[0].keys()))
            w.writeheader()
            w.writerows(runs)


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


if __name__ == "__main__":
    main()
