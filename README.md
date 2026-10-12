# TinyStressNet: A Quantization-Ready Model for Subject-Independent Academic Stress Sensing

**[📘 Read this in Spanish / Leer en español](README_ES.md)**

---

## 📚 Citation

If you use or reference this repository, please cite the published paper:

> **Alcaraz-Valencia, P. A., Santana-Mancilla, P. C., Gaytán-Lugo, L. S., & Anido-Rifón, L. E.**
> *TinyStressNet: A Quantization-Ready Model for Subject-Independent Academic Stress Sensing.*
> **Applied Sciences**, 2026. <https://doi.org/10.3390/app16189028>

---

This repository reproduces every table, figure, and statistic in the paper from the released dataset. TinyStressNet is a **3,139-parameter, 2,800-MAC** classifier that estimates three levels of academic stress (low, medium, high) from three physiological features — galvanic skin response, heart rate, and skin temperature — with a validated post-training **int8** export path.

> Designed for **educational and research purposes**. This is a non-clinical proof of concept: three interval-level summary features cannot characterize a physiological state in clinical terms.

---

## 🧩 Overview

### Data (`stress_data.csv`)

1,000 interval-level observations from five engineering students collected between February and May 2024. One row per measurement: `user_id`, `timestamp`, `gsr`, `heart_rate`, `skin_temperature`, `stress_label`. Labels come from STAI-6 self-reports and are independent of the sensor readings. `user_id` and `timestamp` are used only for grouping and the chronological split — never as features.

### Evaluation (`evaluate.py`)

- **Nested leave-one-subject-out (LOSO)** — the primary protocol. Inner folds pick the epoch budget, the held-out participant is never used for model selection, and standardization is fitted inside training folds only.
- **Per-user chronological split** — a secondary, distinct generalization target (60/20/20 in time).

### Augmentation as approximate invariance (`augmentations.py`)

Four conditions: no augmentation, Gaussian jitter, a radial-rescaling family, and a bounded affine (BA) family whose parameters map onto gain drift, per-sensor calibration offsets, and a correlated baseline shift along an arousal-oriented axis.

### Analysis

- `robustness.py` — bounded scale-offset probe (paper §3.10)
- `stats_tests.py` — exact sign-flip permutation, sign test, Wilcoxon signed-rank (§3.11)
- `quantize.py` — post-training static int8 quantization and TorchScript export (§4.6)

---

## ⚙️ Requirements

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.9+, PyTorch 2.1+, scikit-learn, pandas, numpy, scipy, matplotlib. `statsmodels` is optional and only provides the ordered-logit baseline. No GPU is needed — the model is small enough that CPU is faster in practice.

---

## ▶️ Usage

**Reproduce the paper's tables and figures** (single fixed seed, ~10 minutes on one CPU core):

```bash
python run.py --csv stress_data.csv --outdir results
```

Writes Tables 1–8, the confusion matrices, and `results.json`.

**Reproduce the ten-seed study** behind the augmentation conclusions (~70 minutes):

```bash
python multiseed.py --seeds 0 1 2 3 4 5 6 7 8 9 --probe-draws 30 --outdir results_multiseed
```

Results are cached after every seed, so an interrupted run resumes where it stopped. Output goes to `multiseed_report.txt` (readable tables), `multiseed_results.json`, and `multiseed_runs.csv`.

---

## 🔁 Reproducibility notes

**Seeding.** `--seed-scheme legacy` resets every fit to the same seed and reproduces the fixed-seed numbers in Tables 4, 5, and 7. `--seed-scheme nested` (the default for `multiseed.py`) derives a distinct seed per fit and is what the seed-sensitivity analysis uses. In both schemes the **partitions never move**: LOSO folds are determined by participant and the chronological split by date, so the reported spread reflects training variability alone.

**What the seed study found.** Across ten seeds, subject-independent weighted F1 is 0.900 ± 0.006 (BA), 0.899 ± 0.004 (radial), 0.898 ± 0.005 (none), and 0.894 ± 0.006 (Gaussian jitter). The differences between the top three are smaller than the seed-to-seed variation. The BA family is the most consistent condition — strongest in five of ten seeds, never the weakest — but this is a stability observation, not a measured performance gain.

**The p-value floor.** With five participants there are 2⁵ = 32 sign assignments, so the smallest attainable two-sided p-value is 0.0625 for the permutation test, the sign test, and Wilcoxon alike. No comparison in this study can reach conventional significance, at any effect size, with any number of seeds. All model rankings here are descriptive.

**Quantization backends.** `quantize.py` resolves an available backend automatically (`x86`, `fbgemm`, `qnnpack`, or `onednn`). Artifact sizes vary slightly across PyTorch builds; the parameter-level figure is stable at 12.3 KB float32 versus 3.1 KB int8.

---

## ⚖️ License

Released under the **MIT License** for free use, modification, and redistribution in education and research. See [LICENSE](LICENSE).
