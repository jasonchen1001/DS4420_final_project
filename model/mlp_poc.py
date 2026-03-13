import math
import pandas as pd
import numpy as np

def load_data(csv_path, test_ratio=0.2, seed=42):
    df = pd.read_csv(csv_path)
    y = df["OS"].values.astype(float)

    # drop non-feature columns
    drop = ["sample", "OS", "OS.time", "sampleID_x", "sampleID_y", "sampleID",
            "ER.Status", "PR.Status", "HER2.Final.Status",
            "age_at_initial_pathologic_diagnosis"]
    X = df.drop(columns=[c for c in drop if c in df.columns])
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(X.mean())

    # standardize
    X = (X - X.mean()) / (X.std() + 1e-8)
    X = X.to_numpy(dtype=float)

    # shuffle and split
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    n_test = int(len(X) * test_ratio)
    print(f"train: {len(X)-n_test}, test: {n_test}")
    return X[n_test:], y[n_test:], X[:n_test], y[:n_test]


class MLP:
    def __init__(self, d_in, d_hid=64, lr=1e-3):
        self.W1 = np.random.randn(d_in, d_hid) * math.sqrt(2 / (d_in + d_hid))
        self.b1 = np.zeros((1, d_hid))
        self.W2 = np.random.randn(d_hid, 1) * math.sqrt(2 / (d_hid + 1))
        self.b2 = np.zeros((1, 1))
        self.lr = lr

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

    def forward(self, X):
        h = np.maximum(X @ self.W1 + self.b1, 0)  # relu
        p = self.sigmoid(h @ self.W2 + self.b2)
        return h, p

    def loss(self, p, y):
        y = y.reshape(-1, 1)
        p = np.clip(p, 1e-8, 1 - 1e-8)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def step(self, X, y):
        n = X.shape[0]
        h, p = self.forward(X)
        y = y.reshape(-1, 1)

        # backprop
        dout = (p - y) / n
        dW2 = h.T @ dout
        db2 = dout.sum(axis=0, keepdims=True)

        dh = dout @ self.W2.T
        dh[h <= 0] = 0  # relu gradient

        dW1 = X.T @ dh
        db1 = dh.sum(axis=0, keepdims=True)

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        return self.loss(p, y)

    def predict(self, X):
        _, p = self.forward(X)
        return (p >= 0.5).astype(int).flatten()


def train():
    X_tr, y_tr, X_te, y_te = load_data("/Users/chenyanzhen/Documents/DS4420_Final_Project/data/brca_multimodal.csv")
    model = MLP(X_tr.shape[1])

    for ep in range(20):
        idx = np.random.permutation(len(X_tr))
        X_tr, y_tr = X_tr[idx], y_tr[idx]

        total_loss, cnt = 0, 0
        for i in range(0, len(X_tr), 64):
            xb = X_tr[i:i+64]
            yb = y_tr[i:i+64]
            total_loss += model.step(xb, yb)
            cnt += 1

        tr_acc = (model.predict(X_tr) == y_tr).mean()
        te_acc = (model.predict(X_te) == y_te).mean()
        print(f"epoch {ep+1:02d} | loss={total_loss/cnt:.3f} | train={tr_acc:.3f} | test={te_acc:.3f}")


if __name__ == "__main__":
    train()