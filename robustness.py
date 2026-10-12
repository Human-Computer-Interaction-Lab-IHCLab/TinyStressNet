"""Bounded scale-offset robustness probe (manuscript Section 3.10).

Specification (manuscript Section 3.10): standardized test inputs are
perturbed by random combinations of a global scale and channelwise offsets,

    x' = beta * x + delta,
    beta        ~ U(0.8, 1.2)
    delta_GSR   ~ U(-0.1,  0.1)
    delta_HR    ~ U(-0.1,  0.1)
    delta_Temp  ~ U(-0.07, 0.07)

with 30 draws per setting. Two quantities are summarized:

    prediction consistency - the fraction of test observations whose predicted
                             class is unchanged relative to the unperturbed
                             input. This is the quantity of interest: it
                             measures invariance directly and does not depend
                             on the labels.

    perturbed weighted F1  - weighted F1 against the true labels under
                             perturbation, for comparison with the clean score.

Note the deliberate asymmetry with training. The probe perturbs up to +/-20%
scale and +/-0.1 SD offsets, well beyond the 2-6% and 0.02-0.03 SD amplitudes
used by the bounded affine training family (Section 3.5). The probe is a
stress test, not a re-application of the augmentation.

Scope: this covers bounded drift within one hardware configuration. It does not
represent a different sensor model, sampling regime, or ambient environment.
"""

import numpy as np

from metrics import basic

SCALE_RANGE = (0.8, 1.2)
OFFSET_RANGES = ((-0.10, 0.10), (-0.10, 0.10), (-0.07, 0.07))  # GSR, HR, Temp
N_DRAWS = 30


def perturb(X_std, rng):
    """Apply one draw of x' = beta*x + delta to a standardized (N, 3) array."""
    beta = rng.uniform(*SCALE_RANGE)
    delta = np.array([rng.uniform(lo, hi) for lo, hi in OFFSET_RANGES], dtype=np.float32)
    return (beta * X_std + delta).astype(np.float32)


def probe(predict_fn, X_std, y, n_draws=N_DRAWS, seed=0):
    """Run the probe for one fitted model on one test partition.

    predict_fn: callable (N, 3) -> (N,) integer predictions.
    Returns per-draw consistency and weighted F1, plus the clean score.
    """
    rng = np.random.default_rng(seed)
    clean_pred = np.asarray(predict_fn(X_std))
    clean_f1 = basic(y, clean_pred)["weighted_f1"]

    consistency, perturbed_f1 = [], []
    for _ in range(n_draws):
        pred = np.asarray(predict_fn(perturb(X_std, rng)))
        consistency.append(float(np.mean(pred == clean_pred)))
        perturbed_f1.append(basic(y, pred)["weighted_f1"])

    return {
        "clean_weighted_f1": float(clean_f1),
        "consistency_mean": float(np.mean(consistency)),
        "consistency_std": float(np.std(consistency)),
        "consistency_min": float(np.min(consistency)),
        "perturbed_weighted_f1_mean": float(np.mean(perturbed_f1)),
        "perturbed_weighted_f1_std": float(np.std(perturbed_f1)),
        "perturbed_weighted_f1_min": float(np.min(perturbed_f1)),
        "n_draws": int(n_draws),
    }


def summarize_folds(fold_reports):
    """Aggregate per-fold probe dicts into the numbers Section 4.5 needs."""
    if not fold_reports:
        return {}
    keys = ("clean_weighted_f1", "consistency_mean",
            "perturbed_weighted_f1_mean", "consistency_min",
            "perturbed_weighted_f1_min")
    out = {k: float(np.mean([r[k] for r in fold_reports])) for k in keys}
    out["consistency_across_folds_std"] = float(
        np.std([r["consistency_mean"] for r in fold_reports]))
    out["n_folds"] = len(fold_reports)
    out["n_draws"] = fold_reports[0]["n_draws"]
    return out
