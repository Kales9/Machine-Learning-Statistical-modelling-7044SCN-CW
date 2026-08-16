# ECG Arrhythmia Classification — Statistical Machine Learning Comparison

**Module:** 7044SCN Machine Learning and Statistical Modelling — Coursework
**Author:** Animesh Buwa

## Overview

This project applies five classical machine learning algorithms, plus PCA
(dimensionality reduction) and K-Means clustering, to the problem of
classifying heartbeat segments from the MIT-BIH Arrhythmia Database into
five AAMI-standard diagnostic categories (N, S, V, F, Q), using a
statistically-grounded pipeline with cross-validated, leakage-free
evaluation.

## Dataset

- **Source:** MIT-BIH Arrhythmia Database (PhysioNet), compiled/segmented by
  [dhiah-dev/Dataset-Compiled-ECG-MIT-BIH-Arrhythmia](https://github.com/dhiah-dev/Dataset-Compiled-ECG-MIT-BIH-Arrhythmia)
  (GitHub — not Kaggle, not UCI, not used in module labs).
- 500 ECG segments (10s @ 360Hz, 3600 samples each), balanced across 5 AAMI
  classes (100 each): N (non-ectopic), S (supraventricular ectopic), V
  (ventricular ectopic), F (fusion), Q (unknown/paced).
- Original source: Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia
  Database.* IEEE Eng in Med and Biol 20(3):45-50 (2001).

The raw CSV is included at `data/raw/Compiled-ECG-MIT-BIH-Arrhytmia.csv` for
full reproducibility (no external download needed at run time).

## Repository structure

```
├── data/
│   ├── raw/                     # original compiled ECG CSV
│   ├── features.csv             # extracted feature table (output of step 1)
│   ├── X_scaled.npy, X_pca.npy  # cached preprocessing artefacts
├── src/
│   ├── 01_features.py           # time/frequency-domain feature extraction
│   ├── 02_eda_pca.py            # EDA plots + PCA dimensionality reduction
│   └── 03_models.py             # 5 classifiers + K-Means + statistical tests
├── figures/                     # all generated plots (EDA, PCA, confusion
│                                   matrices, ROC curves, comparisons)
├── results/                     # CSV/JSON of all numeric results
├── run_pipeline.py              # single-command full reproduction
└── requirements.txt
```

## How to reproduce

```bash
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
python run_pipeline.py
```

This runs, in order: feature extraction → EDA/PCA → model training and
evaluation, and regenerates every figure and result file used in the paper.
All random operations use `random_state=42` for exact reproducibility.

## Methods

| Step | Technique | Purpose |
|---|---|---|
| Feature extraction | Statistical (mean, std, skewness, kurtosis, RR-interval stats) + spectral (Welch PSD band powers, spectral centroid) | Reduce 3600-sample raw signal to 26 interpretable features |
| Dimensionality reduction | PCA (95% variance retained) | 8 components from 26 features; used for K-Means and visualisation |
| Classification (×5) | Logistic Regression, Linear Discriminant Analysis, SVM (RBF kernel), Gaussian Naive Bayes, Random Forest | Compared on identical stratified 5-fold CV splits |
| Clustering (×2) | K-Means (k=5, elbow-validated) and DBSCAN (density-based, eps via k-distance/silhouette sweep) | Both compared against known AAMI labels via Adjusted Rand Index — a centroid-based vs density-based contrast |
| Ablation | Raw 26 features vs 8-component PCA space | Tests whether dimensionality reduction actually helps Random Forest (it doesn't, significantly) |

**Leakage control:** all scaling (`StandardScaler`) is fit *only* on each
training fold inside a `sklearn.Pipeline`, never on the full dataset before
splitting — see the Academic Writing / data-leakage discussion in the paper.

## Key results (5-fold stratified cross-validation, mean ± std)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | 0.878 ± 0.023 | 0.876 ± 0.024 |
| LDA | 0.892 ± 0.022 | 0.889 ± 0.025 |
| SVM (RBF) | 0.898 ± 0.027 | 0.895 ± 0.028 |
| Gaussian Naive Bayes | 0.796 ± 0.016 | 0.790 ± 0.016 |
| **Random Forest** | **0.912 ± 0.027** | **0.911 ± 0.028** |

Random Forest was statistically significantly better than the next-best
model (SVM) by a paired t-test across folds (p = 0.025), and a Friedman
test across all five models confirmed genuine performance differences
exist overall (p = 0.006). DBSCAN found 6 density clusters (84 noise
points) with an Adjusted Rand Index of 0.554, comparable to K-Means'
0.549 — a useful cross-check that the recoverable class structure is a
genuine data property, not an artefact of one clustering algorithm's
geometry. Full details, confusion matrices, ROC curves, and the
raw-vs-PCA-features ablation are in `figures/` and `results/`, and
discussed in the accompanying paper.

## AI Use Declaration

Claude (Anthropic) was used to assist with: structuring the codebase and
pipeline; drafting boilerplate plotting/evaluation code, which was reviewed,
executed, and verified against the actual data by the author; and
formatting/copy-editing the accompanying paper. All experimental design
decisions, dataset selection, and interpretation of results were made and
verified by the author. See the AI Use Declaration table in the paper for
full detail per the module's AI usage guidelines.

## References

Moody GB, Mark RG (2001). The impact of the MIT-BIH Arrhythmia Database.
*IEEE Engineering in Medicine and Biology* 20(3), 45–50.

Goldberger AL, et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
*Circulation* 101(23), e215–e220.
