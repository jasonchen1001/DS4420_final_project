# Breast Cancer Survival Prediction with Bayesian Cox and MLP

DS 4420 Final Project — Northeastern University

Yanzhen Chen, Yiyang Bai

## What This Does

Predicts breast cancer patient survival using multi-omics data (gene expression + protein expression + clinical features). Two models:

1. **Bayesian Cox Regression** (R) — identifies prognostic biomarkers through MCMC sampling
2. **MLP Binary Classifier** (Python/NumPy) — classifies patients as long vs short survival using the Cox-selected biomarkers

Both models are hand-written without ML frameworks.

## Project Structure

```
DS4420_final_project-2/
├── data/
│   ├── brca_multimodal.csv          # raw TCGA BRCA data
│   ├── cox_train.csv                # preprocessed, for Cox
│   ├── cox_test.csv
│   ├── mlp_train.csv                # preprocessed, for MLP
│   ├── mlp_test.csv
│   └── features.csv
├── model/
│   ├── preprocess.py                # data preprocessing
│   ├── bayesian_cox.R               # Cox pipeline
│   └── mlp.py                       # MLP classifier
├── results/
│   ├── cox/
│   │   ├── bayesian_cox_summary.csv # posterior summary for 24 features
│   │   ├── test_performance.csv     # C-index, log-rank p
│   │   └── cox_results.rds
│   └── mlp/
│       ├── test_performance.csv     # AUC, accuracy, precision, recall, F1
│       ├── feature_importance.csv   # permutation importance
│       ├── cox_vs_mlp.csv           # feature comparison
│       └── training_history.csv
├── figures/
│   └── cox/
│       ├── forest_plot.pdf          # Bayesian HR with 95% CI
│       └── km_risk_groups.pdf       # KM survival curves by risk group
└── README.md
```

## How to Run

### 1. Preprocess

```bash
python model/preprocess.py
```

Reads `data/brca_multimodal.csv`, outputs train/test CSVs for both Cox and MLP.

### 2. Cox Pipeline

Open R and run:

```r
source("model/bayesian_cox.R")
```

Takes a few minutes (MCMC sampling). Outputs results to `results/cox/` and figures to `figures/cox/`.

### 3. MLP Classifier

```bash
python model/mlp.py
```

Reads Cox-selected features from `results/cox/bayesian_cox_summary.csv`, trains MLP, saves results to `results/mlp/`.

**Run in this order** — MLP depends on Cox output.

## Key Results

- **5 significant biomarkers** identified: age, SPDYC, CYP4Z1, FABP7, ALOX15B
- **Cox**: C-index 0.641, risk stratification log-rank p = 0.0003
- **MLP**: AUC 0.726, precision 0.921
- 3 of 4 gene markers relate to lipid metabolism
- Feature importance rankings consistent across both models

## Dependencies

**Python**: numpy, pandas, scikit-learn (preprocessing only)

**R**: survival, survminer, glmnet, dplyr, ggplot2