# Case Studies Assignment — Part 1.3 Data Analysis

University case-study assignment comparing **XGBoost** and an **MLP** neural network on two banking datasets.

## Datasets

| File | Source | Task |
|------|--------|------|
| `casestudiesDS1.csv` | Bank marketing | Predict term-deposit subscription (`y`) |
| `casestudiesDS2.csv` | Credit card clients | Predict default next month |

Models are run **separately** on each dataset (different targets), then compared for complementary banking insights (campaign opportunity vs credit risk).

## What’s in this repo

| Path | Purpose |
|------|---------|
| `part1_3_ml_analysis.py` | Part 1.3: load data, train XGBoost + MLP, evaluate, save metrics |
| `part1_3_results.json` | Saved holdout metrics and XGBoost feature importances |
| `_figures_part1_3/` | Charts used in the Part 1 written report |
| `Part1_3_Data_Analysis.docx` | Part 1.3 written analysis |
| `part2_fairness_learning_curves.py` | Part 2: stratified learning curves + Fairlearn audits |
| `part2_figures/` | Learning-curve and fairness figures/CSVs |
| `part2_deliberation.tex` | Part 2 Overleaf section (CV, learning curves, Fairlearn, risks) |
| `main_part2.tex` | Minimal Overleaf wrapper for Part 2 |
| `references.bib` | BibTeX for Part 2 |

## How the code works (high level)

1. **Load** DS1 (semicolon-separated) and DS2 (UCI-style header on row 2).
2. **Preprocess** — one-hot encode categoricals; scale numerics for the MLP; class weighting via `scale_pos_weight` for XGBoost.
3. **Train / evaluate** each model with a stratified 80/20 split.
4. **Report** PR-AUC, recall/F1 on the positive class, ROC-AUC (imbalance-aware metrics for the banking case).
5. **Bank ablation** — re-run without `duration` (post-call leakage for targeting).
6. Write metrics to `part1_3_results.json`.

## Run

```bash
pip install pandas scikit-learn xgboost numpy
# macOS may also need: brew install libomp
python part1_3_ml_analysis.py
```

## Note

This repository supports an academic assignment submission. Results and interpretation are in `Part1_3_Data_Analysis.docx`.
