# BRCA Survival Analysis Project
## Yiyang Bai, Yanzhen Chen

## Project Goal

The goal of this project is to predict breast cancer patient survival outcomes and identify molecular markers related to prognosis.

---

## Project Workflow

### 1. Define survival label

Survival labels are derived from clinical survival data.

```text
0 = short survival
1 = long survival
```

The labels are determined based on patient survival time (`OS.time`).

---

### 2. Biomarker discovery

Use **gene expression data** and apply a **Bayesian Cox proportional hazards model** to identify genes associated with patient survival.

---

### 3. Identify important markers

Analyze the model results to determine which genes are significantly related to survival outcomes.

These genes may serve as potential prognostic biomarkers.

---

### 4. Dimensionality reduction

Apply **Principal Component Analysis (PCA)** to reduce the dimensionality of gene expression data.

---

### 5. Survival prediction model

Train a **Multilayer Perceptron (MLP)** to predict patient survival group.

Input features include:

* gene expression
* protein expression
* clinical variables

The model performs binary classification of survival outcomes.

---

### 6. Future extension

If model performance is limited, mutation data may be added to improve prediction.

---
