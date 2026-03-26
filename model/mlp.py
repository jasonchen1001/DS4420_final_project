"""
Enhanced Multilayer Perceptron for Breast Cancer Survival Prediction
DS4420 Final Project - Phase II

Dependencies: numpy, pandas (optional: json for config).
"""

import math
import json
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# Labels & metrics & CV
# =============================================================================


def survival_long_short_labels(df):
    """
    Long vs short from OS.time (median split), aligned with data_preprocess.create_label.
    1 = long (strictly above median), 0 = short.
    """
    threshold = df["OS.time"].median()
    y = (df["OS.time"] > threshold).astype(int).values
    return y, float(threshold)


def confusion_matrix_binary(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return np.array([[tn, fp], [fn, tp]], dtype=float)


def accuracy_score_np(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def precision_score_np(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    if tp + fp == 0:
        return 0.0
    return float(tp / (tp + fp))


def recall_score_np(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    if tp + fn == 0:
        return 0.0
    return float(tp / (tp + fn))


def f1_score_np(y_true, y_pred):
    p = precision_score_np(y_true, y_pred)
    r = recall_score_np(y_true, y_pred)
    if p + r == 0:
        return 0.0
    return float(2 * p * r / (p + r))


def _integrate_trapezoid(y, x):
    """Trapezoid rule; prefers numpy.trapezoid (NumPy 2+), else trapz."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def roc_auc_score_binary(y_true, y_score):
    """Area under ROC curve (trapezoid), binary labels in {0,1}."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = np.concatenate([[0.0], tps / n_pos])
    fpr = np.concatenate([[0.0], fps / n_neg])
    return _integrate_trapezoid(tpr, fpr)


def scale_train_val(X_train, X_val):
    """Z-score using training fold only (no leakage to validation)."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_val - mean) / std


def stratified_k_fold_indices(y, n_folds, seed):
    """
    Return list of (train_idx, val_idx) with stratification by class.
    """
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    n = len(y)
    fold_id = np.empty(n, dtype=int)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        sizes = np.full(n_folds, len(idx) // n_folds, dtype=int)
        sizes[: len(idx) % n_folds] += 1
        start = 0
        for f in range(n_folds):
            fold_id[idx[start : start + sizes[f]]] = f
            start += sizes[f]
    splits = []
    for f in range(n_folds):
        val_idx = np.where(fold_id == f)[0]
        train_idx = np.where(fold_id != f)[0]
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        splits.append((train_idx, val_idx))
    return splits


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================


def load_data(csv_path, test_ratio=0.2, seed=42, feature_names=None):
    """
    Load multimodal table; label = long/short from OS.time median (not OS event).
    Returns raw features (no global standardization — fit scaler on train splits as needed).
    """
    df = pd.read_csv(csv_path)
    y, _ = survival_long_short_labels(df)

    drop = [
        "sample",
        "OS",
        "OS.time",
        "sampleID_x",
        "sampleID_y",
        "sampleID",
        "ER.Status",
        "PR.Status",
        "HER2.Final.Status",
        "age_at_initial_pathologic_diagnosis",
    ]
    X = df.drop(columns=[c for c in drop if c in df.columns])
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(X.mean())

    if feature_names is not None:
        available_features = [f for f in feature_names if f in X.columns]
        if len(available_features) > 0:
            X = X[available_features]
            print(f"Using {len(available_features)} selected features")

    X = X.to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    n_test = int(len(X) * test_ratio)
    print(f"train: {len(X)-n_test}, test: {n_test}")

    return X[n_test:], y[n_test:], X[:n_test], y[:n_test]


def select_features_by_variance(X, k=100, feature_names=None):
    """Select top k features by variance."""
    variances = np.var(X, axis=0)
    top_indices = np.argsort(variances)[-k:][::-1]
    return top_indices


def select_features_univariate(X, y, k=100):
    """Select top k features by univariate correlation with target."""
    correlations = np.array([np.corrcoef(X[:, j], y)[0, 1] for j in range(X.shape[1])])
    correlations = np.nan_to_num(correlations)
    top_indices = np.argsort(np.abs(correlations))[-k:][::-1]
    return top_indices


# =============================================================================
# MLP Implementation with Regularization
# =============================================================================


class MLP:
    """Multilayer Perceptron with support for multiple architectures."""

    def __init__(
        self,
        d_in,
        d_hids=[64, 64],
        lr=1e-3,
        l2_reg=0.0,
        dropout_rate=0.0,
        class_weights=None,
        decision_threshold=0.5,
    ):
        self.d_in = d_in
        self.d_hids = d_hids
        self.lr = lr
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.class_weights = class_weights or {0: 1.0, 1: 1.0}
        self.decision_threshold = decision_threshold

        self.weights = []
        self.biases = []

        self.weights.append(
            np.random.randn(d_in, d_hids[0]) * math.sqrt(2 / (d_in + d_hids[0]))
        )
        self.biases.append(np.zeros((1, d_hids[0])))

        for i in range(len(d_hids) - 1):
            self.weights.append(
                np.random.randn(d_hids[i], d_hids[i + 1])
                * math.sqrt(2 / (d_hids[i] + d_hids[i + 1]))
            )
            self.biases.append(np.zeros((1, d_hids[i + 1])))

        self.weights.append(
            np.random.randn(d_hids[-1], 1) * math.sqrt(2 / (d_hids[-1] + 1))
        )
        self.biases.append(np.zeros((1, 1)))

        self.n_layers = len(self.weights)

    def relu(self, x):
        return np.maximum(x, 0)

    def sigmoid(self, x):
        x = np.clip(x, -50, 50)
        return 1 / (1 + np.exp(-x))

    def forward(self, X, training=False):
        activations = [X]
        h = X
        pre_relu = []
        dropout_masks = []

        for i in range(self.n_layers - 1):
            z = h @ self.weights[i] + self.biases[i]
            a = self.relu(z)
            if training:
                pre_relu.append(z)
                if self.dropout_rate > 0:
                    mask = (np.random.rand(*a.shape) > self.dropout_rate).astype(
                        np.float64
                    )
                    mask = mask / (1.0 - self.dropout_rate)
                    h = a * mask
                    dropout_masks.append(mask)
                else:
                    h = a
                    dropout_masks.append(None)
            else:
                h = a

            activations.append(h)

        p = self.sigmoid(h @ self.weights[-1] + self.biases[-1])
        activations.append(p)

        if training:
            return activations, pre_relu, dropout_masks
        return activations

    def loss(self, p, y):
        y = y.reshape(-1, 1)
        p = np.clip(p, 1e-8, 1 - 1e-8)

        w_pos = self.class_weights[1]
        w_neg = self.class_weights[0]
        ce_loss = -np.mean(w_pos * y * np.log(p) + w_neg * (1 - y) * np.log(1 - p))

        l2_loss = 0.5 * self.l2_reg * sum(np.sum(w**2) for w in self.weights)

        return ce_loss + l2_loss

    def step(self, X, y):
        n = X.shape[0]
        activations, pre_relu, dropout_masks = self.forward(X, training=True)
        y = y.reshape(-1, 1)

        dW = []
        db = []

        dout = activations[-1] - y
        dout = dout / n
        dW.append(activations[-2].T @ dout + self.l2_reg * self.weights[-1])
        db.append(dout.sum(axis=0, keepdims=True))

        for i in range(self.n_layers - 2, -1, -1):
            dout = dout @ self.weights[i + 1].T
            if dropout_masks[i] is not None:
                dout = dout * dropout_masks[i]
            dout[pre_relu[i] <= 0] = 0
            dW.append(activations[i].T @ dout + self.l2_reg * self.weights[i])
            db.append(dout.sum(axis=0, keepdims=True))

        dW = dW[::-1]
        db = db[::-1]

        for i in range(self.n_layers):
            self.weights[i] -= self.lr * dW[i]
            self.biases[i] -= self.lr * db[i]

        return self.loss(activations[-1], y)

    def predict(self, X):
        activations = self.forward(X, training=False)
        return (activations[-1] >= self.decision_threshold).astype(int).flatten()

    def predict_proba(self, X):
        activations = self.forward(X, training=False)
        return activations[-1].flatten()


# =============================================================================
# Evaluation Metrics
# =============================================================================


def compute_metrics(y_true, y_pred, y_proba=None):
    metrics = {
        "accuracy": accuracy_score_np(y_true, y_pred),
        "precision": precision_score_np(y_true, y_pred),
        "recall": recall_score_np(y_true, y_pred),
        "f1": f1_score_np(y_true, y_pred),
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score_binary(y_true, y_proba)

    cm = confusion_matrix_binary(y_true, y_pred)
    metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"] = cm.ravel()

    return metrics


# =============================================================================
# Cross-Validation
# =============================================================================


def cross_validate(X, y, config, n_folds=5, seed=42, already_normalized=False):
    """
    Stratified k-fold CV. By default, fits mean/std on each training fold only.
    Set ``already_normalized=True`` if X (e.g. preprocessed CSV) is already z-scored.

    Optional ``config['n_features_select']``: if set and smaller than n_cols, selects
    top features **using training fold only** (variance or univariate), then scales.
    """
    y = np.asarray(y)
    splits = stratified_k_fold_indices(y, n_folds, seed)

    all_metrics = []
    all_fold_results = []

    n_feat_sel = config.get("n_features_select")
    sel_method = config.get("feature_selection", "variance")

    print(f"\n{'='*60}")
    print(f"{n_folds}-Fold Cross-Validation")
    print(f"{'='*60}")

    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\nFold {fold + 1}/{n_folds}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if n_feat_sel is not None and n_feat_sel < X_train.shape[1]:
            if sel_method == "univariate":
                col_idx = select_features_univariate(X_train, y_train, n_feat_sel)
            else:
                var = np.var(X_train, axis=0)
                col_idx = np.argsort(var)[-n_feat_sel:][::-1]
            X_train = X_train[:, col_idx]
            X_val = X_val[:, col_idx]

        if not already_normalized:
            X_train, X_val = scale_train_val(X_train, X_val)

        n_pos = int(np.sum(y_train == 1))
        n_neg = int(np.sum(y_train == 0))
        total = n_pos + n_neg
        w_pos = total / (2 * n_pos) if n_pos > 0 else 1.0
        w_neg = total / (2 * n_neg) if n_neg > 0 else 1.0
        class_weights = {0: w_neg, 1: w_pos}

        threshold = config.get("decision_threshold", 0.3)
        model = MLP(
            d_in=X_train.shape[1],
            d_hids=config.get("hidden_sizes", [64, 64]),
            lr=config.get("learning_rate", 1e-3),
            l2_reg=config.get("l2_reg", 0.0),
            dropout_rate=config.get("dropout_rate", 0.0),
            class_weights=class_weights,
            decision_threshold=threshold,
        )

        train_losses = []
        val_losses = []

        batch_size = config.get("batch_size", 64)
        n_epochs = config.get("n_epochs", 50)

        for epoch in range(n_epochs):
            perm = np.random.permutation(len(X_train))
            X_train_shuf, y_train_shuf = X_train[perm], y_train[perm]

            epoch_loss = 0
            n_batches = 0
            for i in range(0, len(X_train), batch_size):
                xb = X_train_shuf[i : i + batch_size]
                yb = y_train_shuf[i : i + batch_size]
                epoch_loss += model.step(xb, yb)
                n_batches += 1

            train_losses.append(epoch_loss / n_batches)

            val_proba = model.predict_proba(X_val)
            val_proba = np.clip(val_proba, 1e-8, 1 - 1e-8)
            val_loss = -np.mean(
                y_val * np.log(val_proba) + (1 - y_val) * np.log(1 - val_proba)
            )
            val_losses.append(val_loss)

            if (epoch + 1) % 10 == 0:
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)
                train_acc = accuracy_score_np(y_train, y_train_pred)
                val_acc = accuracy_score_np(y_val, y_val_pred)
                print(
                    f"  Epoch {epoch+1:3d} | Train Loss: {train_losses[-1]:.4f} | "
                    f"Val Loss: {val_losses[-1]:.4f} | Val Acc: {val_acc:.4f}"
                )

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)

        metrics = compute_metrics(y_val, y_pred, y_proba)
        metrics["fold"] = fold + 1
        all_metrics.append(metrics)

        all_fold_results.append(
            {
                "fold": fold + 1,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "metrics": metrics,
            }
        )

    agg_metrics = {}
    for key in all_metrics[0].keys():
        if key != "fold":
            values = [m[key] for m in all_metrics]
            agg_metrics[f"{key}_mean"] = np.mean(values)
            agg_metrics[f"{key}_std"] = np.std(values)
        else:
            agg_metrics[key] = [m[key] for m in all_metrics]

    print(f"\n{'='*60}")
    print("Cross-Validation Results (Mean ± Std)")
    print(f"{'='*60}")
    print(
        f"Accuracy:  {agg_metrics['accuracy_mean']:.4f} ± {agg_metrics['accuracy_std']:.4f}"
    )
    print(
        f"Precision: {agg_metrics['precision_mean']:.4f} ± {agg_metrics['precision_std']:.4f}"
    )
    print(
        f"Recall:    {agg_metrics['recall_mean']:.4f} ± {agg_metrics['recall_std']:.4f}"
    )
    print(f"F1:        {agg_metrics['f1_mean']:.4f} ± {agg_metrics['f1_std']:.4f}")
    print(
        f"ROC-AUC:   {agg_metrics['roc_auc_mean']:.4f} ± {agg_metrics['roc_auc_std']:.4f}"
    )

    return agg_metrics, all_fold_results


# =============================================================================
# Main Training Function
# =============================================================================


def _read_feature_names(path):
    path = Path(path)
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def train_model(config_path=None):
    """
    Train MLP with stratified CV.

    Default: loads preprocessed ``mlp_train.csv`` (long/short label column ``y``,
    features already z-scored in preprocessing — CV skips per-fold scaling).

    Alternative: set ``use_preprocessed_train``: false to read ``brca_multimodal.csv``,
    build labels from ``OS.time`` median, optional ``mlp_features.txt`` column subset,
    then per-fold scaling (and optional per-fold top-k selection via ``n_features``).
    """
    model_dir = Path(__file__).resolve().parent
    project_root = model_dir.parent
    data_dir = project_root / "data"
    results_dir = project_root / "results"

    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {
            "data": {
                "use_preprocessed_train": False,
                "train_csv": str(data_dir / "mlp_train.csv"),
                "csv_path": str(data_dir / "brca_multimodal.csv"),
                "feature_list_path": str(data_dir / "mlp_features.txt"),
                "seed": 42,
                "n_features": 200,
                "feature_selection": "variance",
            },
            "model": {
                "hidden_sizes": [64, 32],
                "learning_rate": 0.003,
                "l2_reg": 0.0001,
                "dropout_rate": 0.2,
                "decision_threshold": 0.5
            },
            "training": {
                "batch_size": 64,
                "n_epochs": 80,
                "n_folds": 5,
            },
        }

    print("=" * 60)
    print("Enhanced MLP for Breast Cancer Survival Prediction")
    print("=" * 60)
    print("\nConfiguration:")
    print(json.dumps(config, indent=2))

    print("\n" + "=" * 60)
    print("Loading Data")
    print("=" * 60)

    data_cfg = config["data"]
    use_pre = data_cfg.get("use_preprocessed_train", True)
    already_normalized = False

    if use_pre:
        train_csv = Path(data_cfg.get("train_csv", str(data_dir / "mlp_train.csv")))
        df = pd.read_csv(train_csv)
        if "y" not in df.columns:
            raise ValueError("Preprocessed train CSV must include a 'y' column (long/short).")
        y = df["y"].values.astype(float)
        X_df = df.drop(columns=["y"])
        X = X_df.values.astype(float)
        already_normalized = True
        print(f"Loaded preprocessed train: {train_csv}")
        print("  (features already scaled in data_preprocess — CV uses no extra z-score)")
    else:
        csv_path = Path(data_cfg.get("csv_path", str(data_dir / "brca_multimodal.csv")))
        df = pd.read_csv(csv_path)
        y, med = survival_long_short_labels(df)
        print(f"Median OS.time = {med:.4f} → long=1, short=0 (not OS event)")

        drop = [
            "sample",
            "OS",
            "OS.time",
            "sampleID_x",
            "sampleID_y",
            "sampleID",
            "ER.Status",
            "PR.Status",
            "HER2.Final.Status",
            "age_at_initial_pathologic_diagnosis",
        ]
        X_df = df.drop(columns=[c for c in drop if c in df.columns])
        X_df = X_df.select_dtypes(include=[np.number])
        X_df = X_df.fillna(X_df.mean())

        feat_path = data_cfg.get("feature_list_path")
        if feat_path:
            names = _read_feature_names(Path(feat_path))
            if names:
                use_cols = [c for c in names if c in X_df.columns]
                if use_cols:
                    X_df = X_df[use_cols]
                    print(f"Subset to {len(use_cols)} columns from feature list")

        X = X_df.to_numpy(dtype=float)
        print(f"Raw feature matrix (unscaled): {X.shape}")

    n_feat_cfg = data_cfg.get("n_features")
    cv_cfg = {
        **config["model"],
        "n_features_select": n_feat_cfg,
        "feature_selection": data_cfg.get("feature_selection", "variance"),
    }

    print(f"Final data shape: {X.shape}")
    print(
        f"Class distribution: 0: {int(np.sum(y==0))}, 1: {int(np.sum(y==1))}"
    )

    print("\n" + "=" * 60)
    print("Running Cross-Validation")
    print("=" * 60)

    agg_metrics, fold_results = cross_validate(
        X,
        y,
        config=cv_cfg,
        n_folds=config["training"]["n_folds"],
        seed=data_cfg["seed"],
        already_normalized=already_normalized,
    )

    print("\n" + "=" * 60)
    print("Saving Results (no matplotlib: figures not generated)")
    print("=" * 60)

    results_dir.mkdir(parents=True, exist_ok=True)
    cv_path = results_dir / "mlp_cv_results.csv"
    summary_path = results_dir / "mlp_metrics_summary.csv"

    rows = []
    for fr in fold_results:
        row = {
            "fold": fr["fold"],
            "final_train_loss": fr["train_losses"][-1] if fr["train_losses"] else None,
            "final_val_loss": fr["val_losses"][-1] if fr["val_losses"] else None,
        }
        row.update(fr["metrics"])
        rows.append(row)
    results_df = pd.DataFrame(rows)
    results_df.to_csv(cv_path, index=False)

    metrics_df = pd.DataFrame([agg_metrics])
    metrics_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("Results Saved")
    print("=" * 60)
    print(f"- CV results: {cv_path}")
    print(f"- Metrics summary: {summary_path}")

    return agg_metrics, fold_results


if __name__ == "__main__":
    train_model()
