"""Reproduce the TinyStressNet results and figures reported in the manuscript.

Usage:
    python run.py --csv stress_data.csv --outdir results

What it produces:
    - Dataset summary (Table 1) and feature statistics (Table 2).
    - LOSO comparison of TinyStressNet (4 augmentation conditions) and classical
      baselines (Table 4), with foldwise weighted-F1 mean +/- std.
    - Per-user LOSO table for the best TinyStressNet config vs QDA (Table 5) and
      subject-macro 95% intervals.
    - Ordinal sensitivity (Table 6) and the per-user chronological split (Table 7).
    - Quantization sizes and float/int8 agreement (Section 4.6 / Table 8).
    - Figures: feature distributions, boxplots, architecture schematic,
      augmentation example, and LOSO confusion matrices.

statsmodels is optional (it provides the ordered-logit baseline).
"""

import argparse
import json
import os

import numpy as np

import torch

import figures
from baselines import OrderedLogit, ordered_logit_available, sklearn_baselines
from data import FEATURES, load_dataframe, standardize, xy
from evaluate import (
    _loader,
    chronological_baseline,
    chronological_tinystressnet,
    fit_fixed,
    loso_baseline,
    loso_tinystressnet,
    subject_macro_ci,
)
from model import count_macs, count_params_analytic
from quantize import quantize_and_report


def hr(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def fmt(m):
    return f"{m['accuracy']:.3f}  {m['weighted_f1']:.3f}  {m['macro_f1']:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="stress_data.csv")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--engine", default="auto", help="quantization backend: auto|fbgemm|x86|qnnpack")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = load_dataframe(args.csv)
    results = {}

    hr("Dataset summary (Table 1) and feature statistics (Table 2)")
    counts = [int((df["y"] == i).sum()) for i in range(3)]
    print(f"observations={len(df)}  participants={df['user_id'].nunique()}  "
          f"classes(low/medium/high)={counts}")
    print("per-user counts:", df["user_id"].value_counts().sort_index().to_dict())
    print(df[FEATURES].describe().round(2).to_string())
    params = f"{count_params_analytic():,}"
    print(f"\nTinyStressNet: parameters={params}  MACs(L=1)={count_macs()}")

    hr("LOSO classical baselines (Table 4 / Table 6)")
    print(f"{'Model':24s} {'acc':>5s} {'wF1':>5s} {'mF1':>5s} {'foldwise wF1':>16s} "
          f"{'MAE':>5s} {'QWK':>5s} {'w1':>5s}")
    bl = sklearn_baselines()
    if ordered_logit_available():
        bl["Ordered logit"] = lambda: OrderedLogit()
    else:
        print("(ordered logit skipped: statsmodels not installed)")
    base_rows = {}
    for name, factory in bl.items():
        r = loso_baseline(df, factory)
        base_rows[name] = r
        a = r["aggregate"]
        print(f"{name:24s} {a['accuracy']:.3f} {a['weighted_f1']:.3f} {a['macro_f1']:.3f}   "
              f"{a['foldwise_weighted_f1_mean']:.3f} ± {a['foldwise_weighted_f1_std']:.3f}   "
              f"{a['ordinal_mae']:.3f} {a['qwk']:.3f} {a['within_one_acc']:.3f}")
    results["baselines_loso"] = {k: v["aggregate"] for k, v in base_rows.items()}

    hr("LOSO TinyStressNet by augmentation condition (Table 4)")
    tiny_rows = {}
    for cond in ("none", "gaussian", "radial", "affine"):
        r = loso_tinystressnet(df, cond, seed=args.seed)
        tiny_rows[cond] = r
        a = r["aggregate"]
        print(f"TinyStressNet / {cond:9s} {a['accuracy']:.3f} {a['weighted_f1']:.3f} "
              f"{a['macro_f1']:.3f}   {a['foldwise_weighted_f1_mean']:.3f} ± "
              f"{a['foldwise_weighted_f1_std']:.3f}")
    results["tinystressnet_loso"] = {k: v["aggregate"] for k, v in tiny_rows.items()}

    hr("Per-user LOSO: best TinyStressNet vs QDA (Table 5)")
    best_cond = max(tiny_rows, key=lambda c: tiny_rows[c]["aggregate"]["weighted_f1"])
    tiny_best, qda = tiny_rows[best_cond], base_rows["QDA"]
    print(f"(best TinyStressNet condition: {best_cond})")
    for u in sorted(tiny_best["per_user"]):
        t, q = tiny_best["per_user"][u], qda["per_user"][u]
        print(f"  {u}: tiny wF1={t['weighted_f1']:.3f} | QDA wF1={q['weighted_f1']:.3f}")
    tmean, tci = subject_macro_ci(tiny_best["per_user"])
    qmean, qci = subject_macro_ci(qda["per_user"])
    print(f"subject-macro wF1: tiny {tmean:.3f} [{tci[0]:.3f},{tci[1]:.3f}] | "
          f"QDA {qmean:.3f} [{qci[0]:.3f},{qci[1]:.3f}]")

    hr("Per-user chronological split (Table 7)")
    for cond in ("none", "gaussian", "radial", "affine"):
        m, _ = chronological_tinystressnet(df, cond, seed=args.seed)
        print(f"TinyStressNet / {cond:9s} {fmt(m)}")
    for name, factory in bl.items():
        m, _ = chronological_baseline(df, factory)
        print(f"{name:24s} {fmt(m)}")

    hr("Quantization and storage (Section 4.6 / Table 8)")
    try:
        print("supported quantization engines:",
              list(torch.backends.quantized.supported_engines))
        X, y = xy(df)
        _, X_std = standardize(X)
        final = fit_fixed(_loader(X_std, y, best_cond, True), epochs=40, seed=args.seed)
        qrep = quantize_and_report(df, final, engine=args.engine, outdir=args.outdir)
        print(json.dumps(qrep, indent=2))
        results["quantization"] = qrep
    except Exception as e:
        print(f"[skipped] quantization not available on this PyTorch build: {e}")
        print("Everything else (Tables 1-7, figures) is unaffected. Re-run the")
        print("quantization on a build with a backend (e.g. a Mac, or a Windows/")
        print("Linux PyTorch wheel that ships fbgemm/x86) to fill in Table 8.")

    yt, yp = tiny_best["pooled"]
    figures.plot_confusion(yt, yp, f"TinyStressNet ({best_cond}) - LOSO",
                           os.path.join(args.outdir, "confusion_tinystressnet_loso.png"))
    yt, yp = base_rows["QDA"]["pooled"]
    figures.plot_confusion(yt, yp, "QDA - LOSO",
                           os.path.join(args.outdir, "confusion_qda_loso.png"))
    figures.plot_feature_distributions(df, os.path.join(args.outdir, "feature_distributions.png"))
    figures.plot_boxplots_by_class(df, os.path.join(args.outdir, "boxplots_by_class.png"))
    figures.plot_architecture(os.path.join(args.outdir, "architecture.png"))
    figures.plot_augmentation_example(df, os.path.join(args.outdir, "augmentation_example.png"))

    with open(os.path.join(args.outdir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(_clean(results), f, indent=2)
    hr("Done")
    print(f"Figures and results.json written to: {args.outdir}/")


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


if __name__ == "__main__":
    main()
