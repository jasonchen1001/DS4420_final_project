import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Load raw BRCA dataset
def load_raw_data():
    path = os.path.join(PROJECT_ROOT, "data", "brca_multimodal.csv")
    df = pd.read_csv(path)
    print(f"Loaded data: {df.shape}")
    return df

# Create long/short labels
def create_label(df):
    threshold = df["OS.time"].median()
    y = (df["OS.time"] > threshold).astype(int).values

    print(f"Median survival: {threshold:.1f}")
    print(f"Long (1): {(y==1).sum()}  Short (0): {(y==0).sum()}")

    return y

# Separate gene features and protein features.
def split_features(df):
    drop_cols = ["sample", "OS", "OS.time", "sampleID_x", "sampleID_y", "sampleID"]
    X_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X_df = X_df.select_dtypes(include=[np.number])

    gene_cols = [c for c in X_df.columns if not c.isupper()]
    protein_cols = [c for c in X_df.columns if c.isupper()]

    return X_df, gene_cols, protein_cols

# Preprocess data for MLP model
def preprocess_for_mlp(df, n_features=500, test_size=0.2, seed=42):
    print(" MLP preprocessing")

    y = create_label(df)

    X_df, _, _ = split_features(df)
    X_df = X_df.fillna(X_df.mean())

    X = X_df.values.astype(float)
    feature_names = X_df.columns.tolist()
    n_pool = X.shape[1]
    if n_features > n_pool:
        raise ValueError(f"n_features={n_features} exceeds available columns ({n_pool})")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    # Top-k by variance on training set only (no test leakage)
    var = np.var(X_train, axis=0)
    idx = np.argsort(var)[-n_features:][::-1]

    X_train = X_train[:, idx]
    X_test = X_test[:, idx]
    feature_names = [feature_names[i] for i in idx]

    print(f"Top features selected (by train variance): {len(feature_names)}")
    print(f"Total features: {len(feature_names)}")

    # normalization (train only)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    # save
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df["y"] = y_train

    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df["y"] = y_test

    train_path = os.path.join(PROJECT_ROOT, "data", "mlp_train.csv")
    test_path = os.path.join(PROJECT_ROOT, "data", "mlp_test.csv")
    feat_path = os.path.join(PROJECT_ROOT, "data", "mlp_features.txt")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    with open(feat_path, "w") as f:
        for name in feature_names:
            f.write(name + "\n")

    print("Saved MLP data")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "features": feature_names
    }

# Prepare data for Cox model.
def preprocess_for_cox(df, n_gene_features=400):
    print(" Cox preprocessing")

    y = df["OS"].values.astype(float)
    t = df["OS.time"].values

    X_df, gene_cols, protein_cols = split_features(df)

    X_gene = X_df[gene_cols].values
    X_protein = X_df[protein_cols].values

    # gene selection (use all data here)
    var = np.var(X_gene, axis=0)
    idx = np.argsort(var)[-n_gene_features:][::-1]

    X_gene = X_gene[:, idx]
    selected_gene_names = list(np.array(gene_cols)[idx])

    # combine
    X = np.concatenate([X_gene, X_protein], axis=1)
    feature_names = selected_gene_names + protein_cols

    print(f"Gene selected: {len(selected_gene_names)}")
    print(f"Protein kept: {len(protein_cols)}")

    # normalization
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X = (X - mean) / std

    cox_df = pd.DataFrame(X, columns=feature_names)
    cox_df["OS"] = y
    cox_df["OS.time"] = t

    path = os.path.join(PROJECT_ROOT, "data", "cox_data.csv")
    feat_path = os.path.join(PROJECT_ROOT, "data", "cox_features.txt")

    cox_df.to_csv(path, index=False)

    with open(feat_path, "w") as f:
        for name in feature_names:
            f.write(name + "\n")

    print("Saved Cox data")

    return {
        "X": X,
        "y": y,
        "time": t,
        "features": feature_names
    }

# Main function
def main():
    df = load_raw_data()
    preprocess_for_mlp(df)
    preprocess_for_cox(df)
    print("\nDone.")

# Main function
if __name__ == "__main__":
    main()