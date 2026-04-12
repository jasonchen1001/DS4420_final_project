import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def make_binary_label(event, time, threshold):
    labels = np.full(len(event), np.nan)
    labels[(event == 1) & (time >= threshold)] = 1
    labels[(event == 1) & (time <  threshold)] = 0
    labels[(event == 0) & (time >= threshold)] = 1
    return labels


def main():
    df = pd.read_csv("/Users/chenyanzhen/Documents/DS4420_final_project-2/data/brca_multimodal.csv")

    # drop
    drop_cols = ["sample", "sampleID_x", "sampleID_y", "sampleID",
                 "ER.Status", "PR.Status", "HER2.Final.Status"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # clean survival columns
    df["OS"] = pd.to_numeric(df["OS"], errors="coerce")
    df["OS.time"] = pd.to_numeric(df["OS.time"], errors="coerce")
    df = df.dropna(subset=["OS", "OS.time"])
    df = df[df["OS.time"] > 0].copy()

    y_event = df["OS"].values.astype(float)
    y_time = df["OS.time"].values.astype(float)

    # binary threshold = median survival of deceased patients
    threshold = np.percentile(y_time[y_event == 1], 50)
    y_binary = make_binary_label(y_event, y_time, threshold)

    # build feature matrix
    X_df = df.drop(columns=["OS", "OS.time"])
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.select_dtypes(include=[np.number])
    X_df = X_df.dropna(axis=1, how="all")
    X_df = X_df.loc[:, X_df.nunique() > 1]  

    feature_names = X_df.columns.tolist()
    clinical_cols = ["age_at_initial_pathologic_diagnosis"]
    clinical_cols = [c for c in clinical_cols if c in feature_names]
    X = X_df.values.astype(float)

    # train/test split 
    X_tr, X_te, ye_tr, ye_te, yt_tr, yt_te, yb_tr, yb_te = \
        train_test_split(X, y_event, y_time, y_binary,
                         test_size=0.2, random_state=42, stratify=y_event)

    # imputation with train means
    col_means = np.nanmean(X_tr, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    for j in range(X_tr.shape[1]):
        X_tr[np.isnan(X_tr[:, j]), j] = col_means[j]
        X_te[np.isnan(X_te[:, j]), j] = col_means[j]

    # variance filter
    variances = np.var(X_tr, axis=0)
    clin_mask = np.array([f in clinical_cols for f in feature_names])
    gene_mask = ~clin_mask
    gene_vars = variances[gene_mask]
    var_cutoff = np.percentile(gene_vars, 50)
    gene_pass = gene_vars >= var_cutoff

    keep = clin_mask.copy()
    keep[gene_mask] = gene_pass
    X_tr = X_tr[:, keep]
    X_te = X_te[:, keep]
    feature_names = [f for f, k in zip(feature_names, keep) if k]
    clinical_idx = [i for i, f in enumerate(feature_names) if f in clinical_cols]

    # standardize with train stats
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0) + 1e-8
    X_tr = (X_tr - mu) / sd
    X_te = (X_te - mu) / sd

    # save cox data 
    cox_train = pd.DataFrame(X_tr, columns=feature_names)
    cox_train["OS"] = ye_tr
    cox_train["OS.time"] = yt_tr
    cox_train.to_csv("/Users/chenyanzhen/Documents/DS4420_final_project-2/data/cox_train.csv", index=False)

    cox_test = pd.DataFrame(X_te, columns=feature_names)
    cox_test["OS"] = ye_te
    cox_test["OS.time"] = yt_te
    cox_test.to_csv("/Users/chenyanzhen/Documents/DS4420_final_project-2/data/cox_test.csv", index=False)

    # save mlp data 
    tr_ok = ~np.isnan(yb_tr)
    te_ok = ~np.isnan(yb_te)

    mlp_train = pd.DataFrame(X_tr[tr_ok], columns=feature_names)
    mlp_train["survival_label"] = yb_tr[tr_ok].astype(int)
    mlp_train.to_csv("/Users/chenyanzhen/Documents/DS4420_final_project-2/data/mlp_train.csv", index=False)

    mlp_test = pd.DataFrame(X_te[te_ok], columns=feature_names)
    mlp_test["survival_label"] = yb_te[te_ok].astype(int)
    mlp_test.to_csv("/Users/chenyanzhen/Documents/DS4420_final_project-2/data/mlp_test.csv", index=False)

    # save feature info
    pd.DataFrame({
        "feature": feature_names,
        "is_clinical": [i in clinical_idx for i in range(len(feature_names))]
    }).to_csv("/Users/chenyanzhen/Documents/DS4420_final_project-2/data/features.csv", index=False)


if __name__ == "__main__":
    main()