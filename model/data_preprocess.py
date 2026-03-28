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


# 2. Preprocess
def preprocess_all(df, var_percentile=50, test_size=0.2, seed=42):
    print("\n" + "=" * 50)
    print("Preprocessing (y-agnostic operations only)")
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
    y_time = df["OS.time"].values.astype(float)

    print(f"Valid samples: {len(df)}")
    print(f"Events (OS=1): {int(y_event.sum())}  "
          f"Censored (OS=0): {int((y_event == 0).sum())}  "
          f"Event rate: {y_event.mean():.1%}")

    # 2c. Build feature matrix (all numeric columns)
    X_df = df.drop(columns=["OS", "OS.time"])
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.select_dtypes(include=[np.number])

    # drop columns with all NaN columns
    X_df = X_df.dropna(axis=1, how="all")
    const_cols = X_df.columns[X_df.nunique() <= 1]
    if len(const_cols) > 0:
        print(f"Dropped {len(const_cols)} constant columns")
        X_df = X_df.drop(columns=const_cols)

    feature_names = X_df.columns.tolist()

    # record clinical features
    clinical_cols = ["age_at_initial_pathologic_diagnosis"]
    clinical_cols = [c for c in clinical_cols if c in feature_names]

    print(f"Total features: {len(feature_names)}")
    print(f"Clinical features: {clinical_cols}")

    X = X_df.values.astype(float)

    # 2d. Train / Test split 
    X_tr, X_te, y_ev_tr, y_ev_te, y_ti_tr, y_ti_te = train_test_split(
        X, y_event, y_time,
        test_size=test_size,
        random_state=seed,
        stratify=y_event
    )

    print(f"\nTrain: {X_tr.shape[0]}  Test: {X_te.shape[0]}")
    print(f"Train events: {int(y_ev_tr.sum())}  "
          f"Test events: {int(y_ev_te.sum())}")

    # 2e. Imputation — use train set means to fill NaNs
    col_means = np.nanmean(X_tr, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)

    for j in range(X_tr.shape[1]):
        mask_tr = np.isnan(X_tr[:, j])
        mask_te = np.isnan(X_te[:, j])
        X_tr[mask_tr, j] = col_means[j]
        X_te[mask_te, j] = col_means[j]

    # 2f. Variance filter
    variances = np.var(X_tr, axis=0)

    # keep clinical features
    clinical_mask = np.array([f in clinical_cols for f in feature_names])
    gene_mask = ~clinical_mask

    # variance filter for gene/protein features
    gene_vars = variances[gene_mask]
    var_threshold = np.percentile(gene_vars, 100 - var_percentile)
    gene_pass = gene_vars >= var_threshold

    # combine clinical features and gene/protein features
    keep_mask = clinical_mask.copy()
    keep_mask[gene_mask] = gene_pass

    X_tr = X_tr[:, keep_mask]
    X_te = X_te[:, keep_mask]
    feature_names = [f for f, k in zip(feature_names, keep_mask) if k]
    clinical_idx = [i for i, f in enumerate(feature_names) if f in clinical_cols]

    print(f"\nAfter variance filter (top {var_percentile}%): "
          f"{len(feature_names)} features kept")

    # 2g. Standardization — use train set means and stds
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0) + 1e-8

    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std

    # 2h. Package and return processed data
    data = {
        "features": feature_names,
        "clinical_idx": clinical_idx,   

        "X_train": X_tr,
        "X_test": X_te,
        "y_event_train": y_ev_tr,
        "y_event_test": y_ev_te,
        "y_time_train": y_ti_tr,
        "y_time_test": y_ti_te,
        "norm_mean": mean,
        "norm_std": std,
    }

    print(f"\n--- Preprocessing done ---")
    print(f"Output shape: X_train {X_tr.shape}, X_test {X_te.shape}")

    return data


# 3. Save datasets
def save_data(data):
    print("\nSaving processed data...")
    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Train set
    train_df = pd.DataFrame(data["X_train"], columns=data["features"])
    train_df["OS"] = data["y_event_train"]
    train_df["OS.time"] = data["y_time_train"]
    train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False)

    # Test set
    test_df = pd.DataFrame(data["X_test"], columns=data["features"])
    test_df["OS"] = data["y_event_test"]
    test_df["OS.time"] = data["y_time_test"]
    test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False)

    # Feature list + metadata
    feat_df = pd.DataFrame({
        "feature": data["features"],
        "is_clinical": [i in data["clinical_idx"]
                        for i in range(len(data["features"]))]
    })
    feat_df.to_csv(os.path.join(data_dir, "features.csv"), index=False)

    # Normalization params (for inference)
    norm_df = pd.DataFrame({
        "feature": data["features"],
        "mean": data["norm_mean"],
        "std": data["norm_std"]
    })
    norm_df.to_csv(os.path.join(data_dir, "norm_params.csv"), index=False)

    print(f"Saved: train.csv, test.csv, features.csv, norm_params.csv")


# 4. Main
def main():
    df = load_raw_data()

    data = preprocess_all(
        df,
        var_percentile=50,   
        test_size=0.2,
        seed=42
    )

    save_data(data)

    print("Done!")

if __name__ == "__main__":
    main()