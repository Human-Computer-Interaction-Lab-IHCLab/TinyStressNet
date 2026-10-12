"""Paired foldwise statistical tests (manuscript Section 3.11).

Exact sign-flip permutation test on foldwise weighted-F1 differences, with the
sign test and Wilcoxon signed-rank as secondary checks.

READ THIS BEFORE INTERPRETING ANY OUTPUT
----------------------------------------
With n = 5 participants there are 2^5 = 32 sign assignments, so the smallest
attainable two-sided p-value from the exact sign-flip test is 2/32 = 0.0625.
The sign test and Wilcoxon signed-rank have the same floor at n = 5. No
comparison in this study can reach p < 0.05, whatever the effect size. That is
a property of the design, not a finding. These tests are reported because the
manuscript says they are reported, and they are read descriptively.

Running more seeds does not change this floor. Averaging over seeds reduces the
initialization noise in each of the five foldwise values, which is worth doing,
but the test still has five paired observations.
"""

import itertools

import numpy as np
from scipy import stats


def sign_flip_permutation(deltas):
    """Exact two-sided sign-flip permutation test on paired differences.

    Enumerates all 2^n sign assignments (exact, not sampled). The null is that
    the sign of each difference is exchangeable.
    """
    d = np.asarray(deltas, dtype=float)
    n = len(d)
    observed = float(np.mean(d))
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    null = signs @ d / n
    p = float(np.mean(np.abs(null) >= abs(observed) - 1e-12))
    return {
        "mean_delta": observed,
        "p_value": p,
        "n": n,
        "n_permutations": int(2 ** n),
        "min_attainable_p": float(2.0 / 2 ** n),
    }


def sign_test(deltas):
    """Two-sided exact binomial sign test (ties dropped)."""
    d = np.asarray(deltas, dtype=float)
    nz = d[d != 0]
    n_pos = int(np.sum(nz > 0))
    n = len(nz)
    if n == 0:
        return {"p_value": 1.0, "n_positive": 0, "n": 0}
    p = float(stats.binomtest(n_pos, n, 0.5, alternative="two-sided").pvalue)
    return {"p_value": p, "n_positive": n_pos, "n": n,
            "min_attainable_p": float(2.0 / 2 ** n)}


def wilcoxon_signed_rank(deltas):
    """Exact Wilcoxon signed-rank test on paired differences."""
    d = np.asarray(deltas, dtype=float)
    if np.all(d == 0):
        return {"p_value": 1.0, "statistic": 0.0, "n": len(d), "note": "all differences zero"}
    try:                                    # scipy >= 1.9
        res = stats.wilcoxon(d, alternative="two-sided", method="exact")
    except TypeError:                       # older scipy
        res = stats.wilcoxon(d, alternative="two-sided", mode="exact")
    return {"p_value": float(res.pvalue), "statistic": float(res.statistic), "n": len(d)}


def compare(foldwise_a, foldwise_b, name_a="A", name_b="B"):
    """Full paired comparison of two models over the same folds."""
    a, b = np.asarray(foldwise_a, float), np.asarray(foldwise_b, float)
    if a.shape != b.shape:
        raise ValueError("foldwise vectors must have the same length (same folds)")
    d = a - b
    return {
        "model_a": name_a,
        "model_b": name_b,
        "foldwise_a": a.tolist(),
        "foldwise_b": b.tolist(),
        "deltas": d.tolist(),
        "mean_delta": float(d.mean()),
        "n_folds_a_higher": int(np.sum(d > 0)),
        "permutation": sign_flip_permutation(d),
        "sign_test": sign_test(d),
        "wilcoxon": wilcoxon_signed_rank(d),
    }


def format_comparison(c):
    p = c["permutation"]
    return (
        f"{c['model_a']} vs {c['model_b']}: mean foldwise delta = {c['mean_delta']:+.4f}; "
        f"{c['model_a']} higher in {c['n_folds_a_higher']}/{len(c['deltas'])} folds; "
        f"exact sign-flip permutation p = {p['p_value']:.4f}; "
        f"sign test p = {c['sign_test']['p_value']:.4f}; "
        f"Wilcoxon signed-rank p = {c['wilcoxon']['p_value']:.4f} "
        f"(smallest attainable two-sided p at n = {p['n']} is {p['min_attainable_p']:.4f})"
    )
