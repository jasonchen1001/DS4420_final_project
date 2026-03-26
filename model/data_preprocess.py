import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


# 1. Load data
def load_raw_data():
    path = os.path.join(PROJECT_ROOT, "data", "brca_multimodal.csv")
    df = pd.read_csv(path)
    print(f"Loaded data: {df.shape}")
    return df

# 2. LASSO feature selection
def lasso_cox_feature_selection(X, y_time, y_event, feature_names, top_k=100):
    df = pd.DataFrame(X, columns=feature_names)
    df["OS.time"] = y_time
    df["OS"] = y_event

    cph = CoxPHFitter(penalizer=0.1, l1_ratio=1.0)
    cph.fit(df, duration_col="OS.time", event_col="OS")

    coefs = cph.params_.values
    idx = np.argsort(np.abs(coefs))[-top_k:][::-1]

    selected_features = [feature_names[i] for i in idx]

    print(f"LASSO selected: {len(selected_features)} features")

    return idx, selected_features

# 3. Unified preprocessing
def preprocess_all(df, n_features=500, test_size=0.2, seed=42):
    print("\nUnified preprocessing")

    # Drop useless columns
    drop_cols = ["sample", "sampleID_x", "sampleID_y", "sampleID"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Ensure survival columns are numeric and drop invalid rows
    df["OS"] = pd.to_numeric(df["OS"], errors="coerce")
    df["OS.time"] = pd.to_numeric(df["OS.time"], errors="coerce")
    df = df.dropna(subset=["OS", "OS.time"]).copy()

    # Extract survival
    y_event = df["OS"].values.astype(float)
    y_time = df["OS.time"].values

    # Feature matrix
    X_df = df.drop(columns=["OS", "OS.time"])
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.select_dtypes(include=[np.number])
    # Two-stage imputation: mean fill, then 0 for all-NaN columns
    X_df = X_df.fillna(X_df.mean())
    X_df = X_df.fillna(0.0)

    X = X_df.values.astype(float)
    feature_names = X_df.columns.tolist()

    # 3. Train/Test Split
    X_train, X_test, y_event_train, y_event_test, y_time_train, y_time_test = train_test_split(
        X,
        y_event,
        y_time,
        test_size=test_size,
        random_state=seed,
        stratify=y_event
    )

    print(f"Train size: {X_train.shape}")
    print(f"Test size: {X_test.shape}")

    # 4. Feature selection

    # clinical features
    clinical_cols = [
        "ER.Status",
        "PR.Status",
        "HER2.Final.Status",
        "age_at_initial_pathologic_diagnosis"
    ]
    clinical_cols = [c for c in clinical_cols if c in feature_names]

    # Step 1: variance filtering
    var = np.var(X_train, axis=0)
    idx_var = np.argsort(var)[-1000:][::-1]

    X_train_var = X_train[:, idx_var]
    X_test_var = X_test[:, idx_var]
    feature_names_var = [feature_names[i] for i in idx_var]

    # Step 2: LASSO Cox
    gene_features = [f for f in feature_names_var if f not in clinical_cols]
    gene_idx = [feature_names_var.index(f) for f in gene_features]

    X_train_gene = X_train_var[:, gene_idx]

    idx_lasso, selected_gene_features = lasso_cox_feature_selection(
        X_train_gene,
        y_time_train,
        y_event_train,
        gene_features,
        top_k=n_features - len(clinical_cols)
    )

    # Combine clinical features
    selected_features = selected_gene_features + clinical_cols

    # Find final index
    final_idx = [feature_names.index(f) for f in selected_features]

    X_train = X_train[:, final_idx]
    X_test = X_test[:, final_idx]
    feature_names = selected_features

    print(f"Final selected features: {len(feature_names)}")
    print(f"Clinical features kept: {len(clinical_cols)}")

    # 5. Normalization
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # 6. MLP label
    threshold = np.median(y_time_train)

    y_mlp_train = (y_time_train > threshold).astype(int)
    y_mlp_test = (y_time_test > threshold).astype(int)

    print(f"Median survival (train): {threshold:.1f}")
    print(f"MLP label balance: {np.mean(y_mlp_train):.2f}")

    return {
        "features": feature_names,

        # Cox
        "X_train": X_train,
        "X_test": X_test,
        "y_event_train": y_event_train,
        "y_event_test": y_event_test,
        "y_time_train": y_time_train,
        "y_time_test": y_time_test,

        # MLP
        "y_mlp_train": y_mlp_train,
        "y_mlp_test": y_mlp_test
    }


# 4. Save datasets
def save_data(data):
    print("\nSaving processed data...")

    # Cox
    cox_train = pd.DataFrame(data["X_train"], columns=data["features"])
    cox_train["OS"] = data["y_event_train"]
    cox_train["OS.time"] = data["y_time_train"]

    cox_test = pd.DataFrame(data["X_test"], columns=data["features"])
    cox_test["OS"] = data["y_event_test"]
    cox_test["OS.time"] = data["y_time_test"]

    cox_train.to_csv(os.path.join(PROJECT_ROOT, "data", "cox_train.csv"), index=False)
    cox_test.to_csv(os.path.join(PROJECT_ROOT, "data", "cox_test.csv"), index=False)

    # MLP
    mlp_train = pd.DataFrame(data["X_train"], columns=data["features"])
    mlp_train["y"] = data["y_mlp_train"]

    mlp_test = pd.DataFrame(data["X_test"], columns=data["features"])
    mlp_test["y"] = data["y_mlp_test"]

    mlp_train.to_csv(os.path.join(PROJECT_ROOT, "data", "mlp_train.csv"), index=False)
    mlp_test.to_csv(os.path.join(PROJECT_ROOT, "data", "mlp_test.csv"), index=False)

    # Feature list
    with open(os.path.join(PROJECT_ROOT, "data", "features.txt"), "w") as f:
        for name in data["features"]:
            f.write(name + "\n")

    print("Saved all datasets")


# 5. Main
def main():
    df = load_raw_data()
    data = preprocess_all(df)
    save_data(data)
    print("\nDone.")


if __name__ == "__main__":
    main()