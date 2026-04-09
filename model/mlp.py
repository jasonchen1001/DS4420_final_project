"""
MLP Binary Classifier — hand-written, numpy only
Uses Cox-selected features from bayesian_cox_summary.csv
Predicts survival_label (0=short, 1=long) from mlp_train/test.csv

Classes (PyTorch style):
  Layers:  Linear, ReLU, Dropout, BatchNorm, Sigmoid
  Model:   MLPClassifier
  Loss:    BCELoss
  Optim:   Adam
"""

import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR)) \
    if "mlp" in SCRIPT_DIR else os.path.dirname(SCRIPT_DIR)
for candidate in [
    os.path.join(SCRIPT_DIR, "..", "..", "data"),
    os.path.join(SCRIPT_DIR, "..", "data"),
]:
    if os.path.isdir(candidate):
        PROJECT_ROOT = os.path.dirname(os.path.abspath(candidate))
        break

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
COX_DIR = os.path.join(PROJECT_ROOT, "results", "cox")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "mlp")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==============================================================
# Layers
# ==============================================================

class Linear:
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
        self.b = np.zeros((1, out_dim))
        self.dW = None
        self.db = None
        self._x = None
        self.training = True

    def forward(self, x):
        self._x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self._x.T @ dout
        self.db = dout.sum(axis=0, keepdims=True)
        return dout @ self.W.T

    def params(self):
        return [(self.W, self.dW, True), (self.b, self.db, False)]


class ReLU:
    def __init__(self):
        self._mask = None
        self.training = True

    def forward(self, x):
        self._mask = (x > 0).astype(np.float64)
        return x * self._mask

    def backward(self, dout):
        return dout * self._mask

    def params(self):
        return []


class Sigmoid:
    def __init__(self):
        self._out = None
        self.training = True

    def forward(self, x):
        self._out = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return self._out

    def backward(self, dout):
        return dout * self._out * (1.0 - self._out)

    def params(self):
        return []


class Dropout:
    def __init__(self, p=0.3):
        self.p = p
        self._mask = None
        self.training = True

    def forward(self, x):
        if self.training and self.p > 0:
            self._mask = (np.random.rand(*x.shape) > self.p) / (1.0 - self.p)
            return x * self._mask
        return x

    def backward(self, dout):
        if self.training and self._mask is not None:
            return dout * self._mask
        return dout

    def params(self):
        return []


class BatchNorm:
    def __init__(self, dim, momentum=0.1, eps=1e-5):
        self.gamma = np.ones((1, dim))
        self.beta = np.zeros((1, dim))
        self.running_mean = np.zeros((1, dim))
        self.running_var = np.ones((1, dim))
        self.momentum = momentum
        self.eps = eps
        self.dgamma = None
        self.dbeta = None
        self._cache = None
        self.training = True

    def forward(self, x):
        if self.training:
            mu = x.mean(axis=0, keepdims=True)
            var = x.var(axis=0, keepdims=True)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mu = self.running_mean
            var = self.running_var
        inv_std = 1.0 / np.sqrt(var + self.eps)
        x_hat = (x - mu) * inv_std
        self._cache = (x, mu, var, inv_std, x_hat)
        return self.gamma * x_hat + self.beta

    def backward(self, dout):
        x, mu, var, inv_std, x_hat = self._cache
        n = x.shape[0]
        self.dgamma = (dout * x_hat).sum(axis=0, keepdims=True)
        self.dbeta = dout.sum(axis=0, keepdims=True)
        dx_hat = dout * self.gamma
        dvar = (dx_hat * (x - mu) * -0.5 * inv_std**3).sum(axis=0, keepdims=True)
        dmu = (dx_hat * -inv_std).sum(axis=0, keepdims=True) + \
              dvar * (-2.0 / n) * (x - mu).sum(axis=0, keepdims=True)
        return dx_hat * inv_std + dvar * 2.0 * (x - mu) / n + dmu / n

    def params(self):
        return [(self.gamma, self.dgamma, False), (self.beta, self.dbeta, False)]


# ==============================================================
# Model
# ==============================================================

class MLPClassifier:
    def __init__(self, in_dim, hidden=(32, 16), dropout=0.3):
        self.layers = []
        prev = in_dim
        for h in hidden:
            self.layers.append(Linear(prev, h))
            self.layers.append(BatchNorm(h))
            self.layers.append(ReLU())
            self.layers.append(Dropout(dropout))
            prev = h
        self.layers.append(Linear(prev, 1))
        self.layers.append(Sigmoid())          # output probability

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def backward(self, grad):
        for l in reversed(self.layers):
            grad = l.backward(grad)

    def train_mode(self):
        for l in self.layers:
            l.training = True

    def eval_mode(self):
        for l in self.layers:
            l.training = False

    def all_params(self):
        p = []
        for l in self.layers:
            p.extend(l.params())
        return p

    def n_params(self):
        return sum(w.size for w, _, _ in self.all_params())

    def save_params(self):
        return [(w.copy(), g, iw) for w, g, iw in self.all_params()]

    def load_params(self, saved):
        for (wc, _, _), (ws, _, _) in zip(self.all_params(), saved):
            wc[:] = ws


# ==============================================================
# Loss
# ==============================================================

class BCELoss:
    """Weighted Binary Cross-Entropy Loss (handles class imbalance)"""
    def __init__(self, pos_weight=None):
        """
        pos_weight: weight for positive class (y=1).
                    If None, auto-computed from first batch as n_neg / n_pos.
        """
        self._pred = None
        self._target = None
        self._w = None
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        p = np.clip(pred.ravel(), 1e-7, 1 - 1e-7)
        t = target.ravel()
        self._pred = p
        self._target = t

        # Auto-compute weight from class ratio if not set
        if self.pos_weight is None:
            n_pos = t.sum()
            n_neg = len(t) - n_pos
            self.pos_weight = n_neg / max(n_pos, 1)

        # Per-sample weight: minority class gets higher weight
        w = np.where(t == 1, self.pos_weight, 1.0)
        self._w = w

        loss = -(w * (t * np.log(p) + (1 - t) * np.log(1 - p))).mean()
        return loss

    def backward(self):
        p = self._pred
        t = self._target
        w = self._w
        n = len(t)
        grad = w * (-t / (p + 1e-7) + (1 - t) / (1 - p + 1e-7)) / n
        return grad.reshape(-1, 1)


# ==============================================================
# Optimizer
# ==============================================================

class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=1e-4):
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = wd
        self.t = 0
        self.m = [np.zeros_like(w) for w, _, _ in params]
        self.v = [np.zeros_like(w) for w, _, _ in params]

    def step(self, params):
        self.t += 1
        for i, (w, g, is_w) in enumerate(params):
            if g is None:
                continue
            gd = g.copy()
            if self.wd > 0 and is_w:
                gd += self.wd * w
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * gd
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * gd**2
            mh = self.m[i] / (1 - self.b1**self.t)
            vh = self.v[i] / (1 - self.b2**self.t)
            w -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ==============================================================
# Metrics
# ==============================================================

def accuracy(pred, target):
    return ((pred.ravel() >= 0.5) == target.ravel()).mean()


def auc_roc(pred, target):
    """Hand-written AUC-ROC (Mann-Whitney U statistic)"""
    p = pred.ravel()
    t = target.ravel()
    pos = p[t == 1]
    neg = p[t == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Count concordant pairs
    conc = 0.0
    for pi in pos:
        conc += (pi > neg).sum() + 0.5 * (pi == neg).sum()
    return conc / (len(pos) * len(neg))


def confusion_matrix(pred, target):
    """Returns (TP, FP, TN, FN)"""
    p = (pred.ravel() >= 0.5).astype(int)
    t = target.ravel().astype(int)
    tp = ((p == 1) & (t == 1)).sum()
    fp = ((p == 1) & (t == 0)).sum()
    tn = ((p == 0) & (t == 0)).sum()
    fn = ((p == 0) & (t == 1)).sum()
    return tp, fp, tn, fn


def classification_report(pred, target):
    tp, fp, tn, fn = confusion_matrix(pred, target)
    acc = (tp + tn) / (tp + fp + tn + fn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    auc = auc_roc(pred, target)
    return {"accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "auc": auc, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def permutation_importance(model, X, y, metric_fn, n_rep=10):
    """Permutation importance based on AUC drop"""
    model.eval_mode()
    base = metric_fn(model.forward(X), y)
    imp = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        drops = []
        for _ in range(n_rep):
            Xp = X.copy()
            np.random.shuffle(Xp[:, j])
            drops.append(base - metric_fn(model.forward(Xp), y))
        imp[j] = np.mean(drops)
    return imp


# ==============================================================
# SMOTE Oversampling (numpy only)
# ==============================================================

def smote(X, y, k=5, seed=42):
    """
    SMOTE: generate synthetic minority samples.
    For each minority sample, pick a random neighbor among k nearest,
    then create a new point on the line between them.

    Only applied to TRAINING set, never test set.
    """
    np.random.seed(seed)
    minority_class = 0 if (y == 0).sum() < (y == 1).sum() else 1
    majority_class = 1 - minority_class

    X_min = X[y == minority_class]
    X_maj = X[y == majority_class]
    n_min = len(X_min)
    n_maj = len(X_maj)
    n_to_generate = n_maj - n_min  # balance to 1:1

    if n_to_generate <= 0 or n_min < 2:
        return X, y

    # k nearest neighbors for each minority sample
    k_use = min(k, n_min - 1)
    synthetic = []

    for _ in range(n_to_generate):
        # Pick a random minority sample
        idx = np.random.randint(n_min)
        x_i = X_min[idx]

        # Find k nearest neighbors (euclidean distance)
        dists = np.sqrt(((X_min - x_i) ** 2).sum(axis=1))
        dists[idx] = np.inf  # exclude itself
        nn_idx = np.argsort(dists)[:k_use]

        # Pick one neighbor randomly
        nn = X_min[nn_idx[np.random.randint(k_use)]]

        # Interpolate: new point = x_i + rand * (nn - x_i)
        lam = np.random.rand()
        synthetic.append(x_i + lam * (nn - x_i))

    X_syn = np.array(synthetic)
    y_syn = np.full(len(X_syn), minority_class)

    X_out = np.vstack([X, X_syn])
    y_out = np.concatenate([y, y_syn])

    # Shuffle
    perm = np.random.permutation(len(X_out))
    return X_out[perm], y_out[perm]


# ==============================================================
# Training
# ==============================================================

class FocalLoss:
    """
    Focal Loss — down-weights easy examples, focuses on hard ones.
    FL = -alpha * (1-p_t)^gamma * log(p_t)
    """
    def __init__(self, gamma=2.0, alpha=None):
        self.gamma = gamma
        self.alpha = alpha
        self._pred = None
        self._target = None
        self._alpha_vec = None

    def forward(self, pred, target):
        p = np.clip(pred.ravel(), 1e-7, 1 - 1e-7)
        t = target.ravel()
        self._pred = p
        self._target = t

        if self.alpha is None:
            n_pos = t.sum()
            n_neg = len(t) - n_pos
            self.alpha = n_neg / (n_pos + n_neg)

        alpha_vec = np.where(t == 1, self.alpha, 1 - self.alpha)
        self._alpha_vec = alpha_vec

        p_t = np.where(t == 1, p, 1 - p)
        focal_weight = (1 - p_t) ** self.gamma
        loss = -(alpha_vec * focal_weight * np.log(p_t + 1e-7)).mean()
        return loss

    def backward(self):
        p = self._pred
        t = self._target
        g = self.gamma
        a = self._alpha_vec
        n = len(t)

        p_t = np.where(t == 1, p, 1 - p)
        fw = (1 - p_t) ** g

        grad_pt = -a * (g * (1 - p_t) ** (g - 1) * np.log(p_t + 1e-7) +
                        fw / (p_t + 1e-7))
        dp = np.where(t == 1, 1.0, -1.0)
        grad = grad_pt * dp / n
        return grad.reshape(-1, 1)


def train_model(model, Xtr, ytr, Xv, yv,
                epochs=300, lr=1e-3, wd=1e-4, patience=30):
    loss_fn = BCELoss()
    opt = Adam(model.all_params(), lr=lr, wd=wd)
    best_auc = 0.0
    best_ep = 0
    best_p = None
    hist = {"loss": [], "val_auc": [], "val_acc": []}

    for ep in range(1, epochs + 1):
        # ---- Train step ----
        model.train_mode()
        pred = model.forward(Xtr)
        loss = loss_fn.forward(pred, ytr)
        model.backward(loss_fn.backward())
        opt.step(model.all_params())

        # ---- Validation ----
        model.eval_mode()
        vpred = model.forward(Xv)
        vauc = auc_roc(vpred, yv)
        vacc = accuracy(vpred, yv)
        hist["loss"].append(loss)
        hist["val_auc"].append(vauc)
        hist["val_acc"].append(vacc)

        # ---- Early stopping (based on AUC) ----
        if vauc > best_auc:
            best_auc = vauc
            best_ep = ep
            best_p = model.save_params()
        if ep - best_ep >= patience:
            print(f"  Early stop at epoch {ep}")
            break
        if ep % 50 == 0:
            print(f"  Epoch {ep:4d} | loss={loss:.4f} | "
                  f"val_auc={vauc:.4f} | val_acc={vacc:.4f}")

    if best_p:
        model.load_params(best_p)
    print(f"  Best: epoch {best_ep}, val_auc={best_auc:.4f}")
    return hist, best_auc, best_ep


# ==============================================================
# Main
# ==============================================================

def main():
    print("=" * 60)
    print("MLP Binary Classifier (Cox-selected features)")
    print("  Task: predict survival_label (0=short, 1=long)")
    print("=" * 60)

    # ---- Load Cox feature list ----
    cox_df = pd.read_csv(os.path.join(COX_DIR, "bayesian_cox_summary.csv"))
    all_cox_feats = cox_df["feature"].tolist()
    sig_feats = cox_df[cox_df["signif"] == True]["feature"].tolist()
    print(f"Cox features: {len(all_cox_feats)} total | "
          f"{len(sig_feats)} significant")
    print(f"  Significant: {sig_feats}")

    # Use significant features only (5 features)
    # 24-feature version was tested but showed severe overfitting
    # (Train AUC=0.92, Test AUC=0.62, gap=0.30)
    features = sig_feats if len(sig_feats) >= 3 else all_cox_feats
    print(f"  Using: {len(features)} features")

    # ---- Load MLP data (binary labels) ----
    train_df = pd.read_csv(os.path.join(DATA_DIR, "mlp_train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "mlp_test.csv"))

    X_all = train_df[features].values.astype(np.float64)
    y_all = train_df["survival_label"].values.astype(np.float64)
    X_test = test_df[features].values.astype(np.float64)
    y_test = test_df["survival_label"].values.astype(np.float64)

    print(f"\nTrain: {X_all.shape[0]} samples | "
          f"y=1: {int(y_all.sum())} | y=0: {int((y_all==0).sum())}")
    print(f"Test:  {X_test.shape[0]} samples | "
          f"y=1: {int(y_test.sum())} | y=0: {int((y_test==0).sum())}")

    # ---- Stratified train/val split ----
    np.random.seed(42)
    pos_i = np.where(y_all == 1)[0]
    neg_i = np.where(y_all == 0)[0]
    np.random.shuffle(pos_i)
    np.random.shuffle(neg_i)
    nv_p = max(1, int(len(pos_i) * 0.2))
    nv_n = max(1, int(len(neg_i) * 0.2))
    vi = np.concatenate([pos_i[:nv_p], neg_i[:nv_n]])
    ti = np.concatenate([pos_i[nv_p:], neg_i[nv_n:]])

    Xtr, Xv = X_all[ti], X_all[vi]
    ytr, yv = y_all[ti], y_all[vi]
    print(f"  Split: train {len(ti)} (y=1:{int(ytr.sum())}) | "
          f"val {len(vi)} (y=1:{int(yv.sum())})")

    # ---- Build & Train ----
    n_pos = ytr.sum()
    n_neg = len(ytr) - n_pos
    pos_w = n_neg / max(n_pos, 1)
    print(f"\n  Class weight: pos_weight={pos_w:.3f} "
          f"(neg:{int(n_neg)} / pos:{int(n_pos)})")

    model = MLPClassifier(in_dim=len(features), hidden=(32, 16), dropout=0.3)
    print(f"  Model: {len(features)} -> 32 -> 16 -> 1(sigmoid) "
          f"({model.n_params()} params)\n")

    hist, best_auc, best_ep = train_model(
        model, Xtr, ytr, Xv, yv,
        epochs=300, lr=1e-3, wd=1e-4, patience=30
    )

    # ---- Evaluate on test set ----
    print(f"\n{'='*60}")
    print("Test Set Evaluation")
    print(f"{'='*60}")
    model.eval_mode()

    test_pred = model.forward(X_test)
    report = classification_report(test_pred, y_test)

    print(f"Accuracy:  {report['accuracy']:.4f}")
    print(f"Precision: {report['precision']:.4f}")
    print(f"Recall:    {report['recall']:.4f}")
    print(f"F1 Score:  {report['f1']:.4f}")
    print(f"AUC-ROC:   {report['auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP={report['tp']}  FP={report['fp']}")
    print(f"  FN={report['fn']}  TN={report['tn']}")

    # Also evaluate on train set (for overfitting check)
    train_pred = model.forward(X_all)
    train_report = classification_report(train_pred, y_all)
    print(f"\nTrain AUC: {train_report['auc']:.4f} | "
          f"Test AUC: {report['auc']:.4f} | "
          f"Gap: {train_report['auc'] - report['auc']:.4f}")

    # ---- Permutation importance ----
    print(f"\nPermutation Importance (AUC drop)...")
    imp = permutation_importance(model, X_test, y_test,
                                 metric_fn=auc_roc, n_rep=10)
    imp_df = pd.DataFrame({
        "feature": features, "importance": imp
    }).sort_values("importance", ascending=False)
    print(imp_df.to_string(index=False))

    # ---- Cox vs MLP feature comparison ----
    cox_rank = cox_df.set_index("feature").loc[features]
    comp = pd.DataFrame({
        "feature": features,
        "cox_HR": cox_rank["HR"].values,
        "cox_signif": cox_rank["signif"].values,
        "mlp_importance": imp
    }).sort_values("mlp_importance", ascending=False)

    mlp_top = set(imp_df.head(min(5, len(features)))["feature"])
    cox_sig = set(sig_feats)
    print(f"\nCox significant: {cox_sig}")
    print(f"MLP top features: {mlp_top}")
    print(f"Overlap: {mlp_top & cox_sig if mlp_top & cox_sig else 'none'}")

    # ---- Save results ----
    perf_df = pd.DataFrame([{
        "metric": k, "value": v
    } for k, v in report.items() if isinstance(v, float)])
    perf_df.to_csv(os.path.join(RESULTS_DIR, "test_performance.csv"), index=False)
    imp_df.to_csv(os.path.join(RESULTS_DIR, "feature_importance.csv"), index=False)
    comp.to_csv(os.path.join(RESULTS_DIR, "cox_vs_mlp.csv"), index=False)

    # Save training history
    hist_df = pd.DataFrame(hist)
    hist_df.to_csv(os.path.join(RESULTS_DIR, "training_history.csv"), index=False)

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()