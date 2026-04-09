import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


# 1. Load data
def load_raw_data():
    path = os.path.join(PROJECT_ROOT, "data", "brca_multimodal.csv")
    df = pd.read_csv(path)
    print(f"Loaded data: {df.shape}")
    return df


# ── Binary label for MLP ───────────────────────────────────────────
def make_binary_survival_label(event, time, threshold):
    """
    Convert survival data to binary labels.

    - Deceased (OS=1): compare OS.time with threshold
    - Censored (OS=0) & OS.time >= threshold: label 1 (long survival)
    - Censored (OS=0) & OS.time <  threshold: NaN (ambiguous, removed later)

    Returns: labels array (with NaN for ambiguous samples)
    """
    labels = np.full(len(event), np.nan)

    # Deceased patients
    labels[(event == 1) & (time >= threshold)] = 1
    labels[(event == 1) & (time <  threshold)] = 0

    # Censored patients
    labels[(event == 0) & (time >= threshold)] = 1
    # (event == 0) & (time < threshold) stays NaN

    return labels


# 2. Preprocess (shared pipeline for both Cox and MLP)
def preprocess_all(df, var_percentile=50, test_size=0.2, seed=42,
                   threshold_days=None, threshold_percentile=50):
    print("\n" + "=" * 50)
    print("Preprocessing")
    print("=" * 50)

    # 2a. Drop useless columns
    drop_cols = [
        "sample", "sampleID_x", "sampleID_y", "sampleID",
        "ER.Status", "PR.Status", "HER2.Final.Status"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 2b. Ensure survival columns are valid
    df["OS"] = pd.to_numeric(df["OS"], errors="coerce")
    df["OS.time"] = pd.to_numeric(df["OS.time"], errors="coerce")
    df = df.dropna(subset=["OS", "OS.time"])
    df = df[df["OS.time"] > 0].copy()

    y_event = df["OS"].values.astype(float)
    y_time  = df["OS.time"].values.astype(float)

    print(f"Valid samples: {len(df)}")
    print(f"Events (OS=1): {int(y_event.sum())}  "
          f"Censored (OS=0): {int((y_event == 0).sum())}  "
          f"Event rate: {y_event.mean():.1%}")

    # ── Compute binary threshold (for MLP) ──
    if threshold_days is None:
        event_times = y_time[y_event == 1]
        threshold = np.percentile(event_times, threshold_percentile)
    else:
        threshold = threshold_days
    print(f"\nBinary threshold: {threshold:.0f} days "
          f"({threshold/365.25:.1f} years)")

    # ── Generate binary labels ──
    y_binary = make_binary_survival_label(y_event, y_time, threshold)

    # Mark which samples are valid for MLP (non-ambiguous)
    mlp_valid = ~np.isnan(y_binary)
    n_removed = (~mlp_valid).sum()
    print(f"MLP valid samples: {mlp_valid.sum()}  "
          f"(removed {n_removed} ambiguous censored)")

    # 2c. Build feature matrix
    X_df = df.drop(columns=["OS", "OS.time"])
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.select_dtypes(include=[np.number])

    X_df = X_df.dropna(axis=1, how="all")
    const_cols = X_df.columns[X_df.nunique() <= 1]
    if len(const_cols) > 0:
        print(f"Dropped {len(const_cols)} constant columns")
        X_df = X_df.drop(columns=const_cols)

    feature_names = X_df.columns.tolist()

    clinical_cols = ["age_at_initial_pathologic_diagnosis"]
    clinical_cols = [c for c in clinical_cols if c in feature_names]

    print(f"Total features: {len(feature_names)}")

    X = X_df.values.astype(float)

    # 2d. Train / Test split
    #     Use ALL samples, stratify by event (works for Cox).
    #     MLP will use a subset after split.
    X_tr, X_te, ye_tr, ye_te, yt_tr, yt_te, yb_tr, yb_te = \
        train_test_split(
            X, y_event, y_time, y_binary,
            test_size=test_size,
            random_state=seed,
            stratify=y_event
        )

    print(f"\nTrain: {X_tr.shape[0]}  Test: {X_te.shape[0]}")

    # 2e. Imputation — train set means
    col_means = np.nanmean(X_tr, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    for j in range(X_tr.shape[1]):
        X_tr[np.isnan(X_tr[:, j]), j] = col_means[j]
        X_te[np.isnan(X_te[:, j]), j] = col_means[j]

    # 2f. Variance filter
    variances = np.var(X_tr, axis=0)
    clinical_mask = np.array([f in clinical_cols for f in feature_names])
    gene_mask = ~clinical_mask

    gene_vars = variances[gene_mask]
    var_threshold = np.percentile(gene_vars, 100 - var_percentile)
    gene_pass = gene_vars >= var_threshold

    keep_mask = clinical_mask.copy()
    keep_mask[gene_mask] = gene_pass

    X_tr = X_tr[:, keep_mask]
    X_te = X_te[:, keep_mask]
    feature_names = [f for f, k in zip(feature_names, keep_mask) if k]
    clinical_idx = [i for i, f in enumerate(feature_names) if f in clinical_cols]

    print(f"After variance filter (top {var_percentile}%): "
          f"{len(feature_names)} features kept")

    # 2g. Standardization — train set mean and std
    mean = X_tr.mean(axis=0)
    std  = X_tr.std(axis=0) + 1e-8
    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std

    # 2h. Package everything
    data = {
        "features": feature_names,
        "clinical_idx": clinical_idx,
        "threshold_days": threshold,
        "norm_mean": mean,
        "norm_std": std,

        # Cox needs these (all samples)
        "X_train": X_tr,
        "X_test": X_te,
        "y_event_train": ye_tr,
        "y_event_test": ye_te,
        "y_time_train": yt_tr,
        "y_time_test": yt_te,

        # MLP needs these (binary labels, with NaN for ambiguous)
        "y_binary_train": yb_tr,
        "y_binary_test": yb_te,
    }

    print(f"\n--- Preprocessing done ---")
    print(f"Output: X_train {X_tr.shape}, X_test {X_te.shape}")

    return data


# 3. Save datasets
def save_data(data):
    print("\nSaving processed data...")
    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)

    feats = data["features"]
    X_tr  = data["X_train"]
    X_te  = data["X_test"]

    # ── Cox: all samples, with OS + OS.time ──
    cox_train = pd.DataFrame(X_tr, columns=feats)
    cox_train["OS"]      = data["y_event_train"]
    cox_train["OS.time"] = data["y_time_train"]
    cox_train.to_csv(os.path.join(data_dir, "cox_train.csv"), index=False)

    cox_test = pd.DataFrame(X_te, columns=feats)
    cox_test["OS"]      = data["y_event_test"]
    cox_test["OS.time"] = data["y_time_test"]
    cox_test.to_csv(os.path.join(data_dir, "cox_test.csv"), index=False)

    # ── MLP: remove ambiguous samples, with survival_label ──
    mlp_tr_mask = ~np.isnan(data["y_binary_train"])
    mlp_te_mask = ~np.isnan(data["y_binary_test"])

    mlp_train = pd.DataFrame(X_tr[mlp_tr_mask], columns=feats)
    mlp_train["survival_label"] = data["y_binary_train"][mlp_tr_mask].astype(int)
    mlp_train.to_csv(os.path.join(data_dir, "mlp_train.csv"), index=False)

    mlp_test = pd.DataFrame(X_te[mlp_te_mask], columns=feats)
    mlp_test["survival_label"] = data["y_binary_test"][mlp_te_mask].astype(int)
    mlp_test.to_csv(os.path.join(data_dir, "mlp_test.csv"), index=False)

    print(f"Cox  -> cox_train.csv ({len(cox_train)}), "
          f"cox_test.csv ({len(cox_test)})")
    print(f"MLP  -> mlp_train.csv ({len(mlp_train)}), "
          f"mlp_test.csv ({len(mlp_test)})")

    # Feature list
    feat_df = pd.DataFrame({
        "feature": feats,
        "is_clinical": [i in data["clinical_idx"]
                        for i in range(len(feats))]
    })
    feat_df.to_csv(os.path.join(data_dir, "features.csv"), index=False)

    # Normalization params
    norm_df = pd.DataFrame({
        "feature": feats,
        "mean": data["norm_mean"],
        "std": data["norm_std"]
    })
    norm_df.to_csv(os.path.join(data_dir, "norm_params.csv"), index=False)

    # Threshold metadata
    meta = pd.DataFrame({
        "key": ["threshold_days", "threshold_years"],
        "value": [data["threshold_days"],
                  data["threshold_days"] / 365.25]
    })
    meta.to_csv(os.path.join(data_dir, "meta.csv"), index=False)

    print("Saved: features.csv, norm_params.csv, meta.csv")


# 4. Main
def main():
    df = load_raw_data()

    data = preprocess_all(
        df,
        var_percentile=50,
        test_size=0.2,
        seed=42,
        threshold_days=None,          # None -> auto percentile
        threshold_percentile=50,      # median survival of deceased patients
    )

    save_data(data)
    print("Done!")


if __name__ == "__main__":
    main()