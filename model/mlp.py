import numpy as np
import pandas as pd

# layers
class Linear:
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
        self.b = np.zeros((1, out_dim))
        self.dW = None; self.db = None; self._x = None
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
        self._mask = None; self.training = True
    def forward(self, x):
        self._mask = (x > 0).astype(np.float64)
        return x * self._mask
    def backward(self, dout):
        return dout * self._mask
    def params(self): return []


class Sigmoid:
    def __init__(self):
        self._out = None; self.training = True
    def forward(self, x):
        self._out = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return self._out
    def backward(self, dout):
        return dout * self._out * (1.0 - self._out)
    def params(self): return []


class Dropout:
    def __init__(self, p=0.3):
        self.p = p; self._mask = None; self.training = True
    def forward(self, x):
        if self.training and self.p > 0:
            self._mask = (np.random.rand(*x.shape) > self.p) / (1.0 - self.p)
            return x * self._mask
        return x
    def backward(self, dout):
        if self.training and self._mask is not None:
            return dout * self._mask
        return dout
    def params(self): return []


class BatchNorm:
    def __init__(self, dim, momentum=0.1, eps=1e-5):
        self.gamma = np.ones((1, dim))
        self.beta = np.zeros((1, dim))
        self.running_mean = np.zeros((1, dim))
        self.running_var = np.ones((1, dim))
        self.momentum = momentum; self.eps = eps
        self.dgamma = None; self.dbeta = None
        self._cache = None; self.training = True

    def forward(self, x):
        if self.training:
            mu = x.mean(axis=0, keepdims=True)
            var = x.var(axis=0, keepdims=True)
            self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mu
            self.running_var = (1-self.momentum)*self.running_var + self.momentum*var
        else:
            mu = self.running_mean; var = self.running_var
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
              dvar * (-2.0/n) * (x - mu).sum(axis=0, keepdims=True)
        return dx_hat * inv_std + dvar * 2.0*(x - mu)/n + dmu/n

    def params(self):
        return [(self.gamma, self.dgamma, False), (self.beta, self.dbeta, False)]


# model
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
        self.layers.append(Sigmoid())

    def forward(self, x):
        for l in self.layers: x = l.forward(x)
        return x
    def backward(self, grad):
        for l in reversed(self.layers): grad = l.backward(grad)
    def train_mode(self):
        for l in self.layers: l.training = True
    def eval_mode(self):
        for l in self.layers: l.training = False
    def all_params(self):
        p = []
        for l in self.layers: p.extend(l.params())
        return p
    def save_params(self):
        return [(w.copy(), g, iw) for w, g, iw in self.all_params()]
    def load_params(self, saved):
        for (wc, _, _), (ws, _, _) in zip(self.all_params(), saved):
            wc[:] = ws


# weighted BCE loss
class BCELoss:
    def __init__(self, pos_weight=None):
        self.pos_weight = pos_weight
        self._pred = None; self._target = None; self._w = None

    def forward(self, pred, target):
        p = np.clip(pred.ravel(), 1e-7, 1-1e-7)
        t = target.ravel()
        self._pred = p; self._target = t
        if self.pos_weight is None:
            self.pos_weight = (len(t) - t.sum()) / max(t.sum(), 1)
        self._w = np.where(t == 1, self.pos_weight, 1.0)
        return -(self._w * (t*np.log(p) + (1-t)*np.log(1-p))).mean()

    def backward(self):
        p, t, w = self._pred, self._target, self._w
        grad = w * (-t/(p+1e-7) + (1-t)/(1-p+1e-7)) / len(t)
        return grad.reshape(-1, 1)


# adam optimizer
class Adam:
    def __init__(self, params, lr=1e-3, wd=1e-4):
        self.lr = lr; self.wd = wd; self.t = 0
        self.m = [np.zeros_like(w) for w, _, _ in params]
        self.v = [np.zeros_like(w) for w, _, _ in params]

    def step(self, params):
        self.t += 1
        for i, (w, g, is_w) in enumerate(params):
            if g is None: continue
            gd = g + self.wd*w if self.wd > 0 and is_w else g.copy()
            self.m[i] = 0.9*self.m[i] + 0.1*gd
            self.v[i] = 0.999*self.v[i] + 0.001*gd**2
            mh = self.m[i] / (1 - 0.9**self.t)
            vh = self.v[i] / (1 - 0.999**self.t)
            w -= self.lr * mh / (np.sqrt(vh) + 1e-8)


# metrics

def auc_roc(pred, target):
    p = pred.ravel(); t = target.ravel()
    pos = p[t == 1]; neg = p[t == 0]
    if len(pos) == 0 or len(neg) == 0: return 0.5
    conc = sum((pi > neg).sum() + 0.5*(pi == neg).sum() for pi in pos)
    return conc / (len(pos) * len(neg))

def classification_report(pred, target):
    p = (pred.ravel() >= 0.5).astype(int)
    t = target.ravel().astype(int)
    tp = ((p==1)&(t==1)).sum(); fp = ((p==1)&(t==0)).sum()
    tn = ((p==0)&(t==0)).sum(); fn = ((p==0)&(t==1)).sum()
    acc = (tp+tn) / (tp+fp+tn+fn)
    prec = tp/(tp+fp) if tp+fp > 0 else 0.0
    rec = tp/(tp+fn) if tp+fn > 0 else 0.0
    f1 = 2*prec*rec/(prec+rec) if prec+rec > 0 else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "auc": auc_roc(pred, target),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}

def perm_importance(model, X, y, n_rep=10):
    model.eval_mode()
    base = auc_roc(model.forward(X), y)
    imp = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        drops = []
        for _ in range(n_rep):
            Xp = X.copy()
            np.random.shuffle(Xp[:, j])
            drops.append(base - auc_roc(model.forward(Xp), y))
        imp[j] = np.mean(drops)
    return imp


# training loop
def train(model, Xtr, ytr, Xv, yv, epochs=300, lr=1e-3, wd=1e-4, patience=30):
    loss_fn = BCELoss()
    opt = Adam(model.all_params(), lr=lr, wd=wd)
    best_auc = 0.0; best_ep = 0; best_p = None
    hist = {"loss": [], "val_auc": []}

    for ep in range(1, epochs+1):
        model.train_mode()
        pred = model.forward(Xtr)
        loss = loss_fn.forward(pred, ytr)
        model.backward(loss_fn.backward())
        opt.step(model.all_params())

        model.eval_mode()
        vauc = auc_roc(model.forward(Xv), yv)
        hist["loss"].append(loss)
        hist["val_auc"].append(vauc)

        if vauc > best_auc:
            best_auc = vauc; best_ep = ep
            best_p = model.save_params()
        if ep - best_ep >= patience:
            break

    if best_p: model.load_params(best_p)
    return hist


# main
def main():
    # load cox results
    cox_df = pd.read_csv("/Users/chenyanzhen/Documents/DS4420_final_project-2/results/cox/bayesian_cox_summary.csv")
    sig_feats = cox_df[cox_df["signif"] == True]["feature"].tolist()
    feats = sig_feats if len(sig_feats) >= 3 else cox_df["feature"].tolist()

    # load data
    train_df = pd.read_csv("/Users/chenyanzhen/Documents/DS4420_final_project-2/data/mlp_train.csv")
    test_df = pd.read_csv("/Users/chenyanzhen/Documents/DS4420_final_project-2/data/mlp_test.csv")
    X_all = train_df[feats].values.astype(np.float64)
    y_all = train_df["survival_label"].values.astype(np.float64)
    X_test = test_df[feats].values.astype(np.float64)
    y_test = test_df["survival_label"].values.astype(np.float64)

    # stratified train/val split
    np.random.seed(42)
    pos_i = np.where(y_all == 1)[0]; neg_i = np.where(y_all == 0)[0]
    np.random.shuffle(pos_i); np.random.shuffle(neg_i)
    nv_p = max(1, int(len(pos_i)*0.2))
    nv_n = max(1, int(len(neg_i)*0.2))
    vi = np.concatenate([pos_i[:nv_p], neg_i[:nv_n]])
    ti = np.concatenate([pos_i[nv_p:], neg_i[nv_n:]])
    Xtr, Xv = X_all[ti], X_all[vi]
    ytr, yv = y_all[ti], y_all[vi]

    # train
    model = MLPClassifier(in_dim=len(feats), hidden=(32, 16), dropout=0.3)
    hist = train(model, Xtr, ytr, Xv, yv, epochs=300, lr=1e-3, wd=1e-4, patience=30)

    # evaluate
    model.eval_mode()
    report = classification_report(model.forward(X_test), y_test)
    train_report = classification_report(model.forward(X_all), y_all)

    # feature importance
    imp = perm_importance(model, X_test, y_test, n_rep=10)
    imp_df = pd.DataFrame({"feature": feats, "importance": imp}) \
             .sort_values("importance", ascending=False)

    # cox vs mlp comparison
    cox_rank = cox_df.set_index("feature").loc[feats]
    comp = pd.DataFrame({
        "feature": feats,
        "cox_HR": cox_rank["HR"].values,
        "cox_signif": cox_rank["signif"].values,
        "mlp_importance": imp
    }).sort_values("mlp_importance", ascending=False)

    # save
    perf = pd.DataFrame([{"metric": k, "value": v}
                          for k, v in report.items() if isinstance(v, float)])
    perf.to_csv("/Users/chenyanzhen/Documents/DS4420_final_project-2/results/mlp/test_performance.csv", index=False)
    imp_df.to_csv("/Users/chenyanzhen/Documents/DS4420_final_project-2/results/mlp/feature_importance.csv", index=False)
    comp.to_csv("/Users/chenyanzhen/Documents/DS4420_final_project-2/results/mlp/cox_vs_mlp.csv", index=False)
    pd.DataFrame(hist).to_csv("/Users/chenyanzhen/Documents/DS4420_final_project-2/results/mlp/training_history.csv", index=False)


if __name__ == "__main__":
    main()