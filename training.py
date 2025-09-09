# train_once.py
import json, joblib, numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity

# ---------- Model ----------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)  # single target output
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)  # (batch, 1)

def make_sequences(X, y, window=42, horizon=1):
    Xs, ys = [], []
    for i in range(len(X) - window - horizon + 1):
        Xs.append(X[i:i+window])
        ys.append(y[i+window+horizon-1])
    return np.array(Xs), np.array(ys)

# ---------- Residual UCL ----------
def residual_ucl(residuals, alpha=0.99):
    return float(np.quantile(np.abs(residuals), alpha))

# ---------- CVA ----------
def sym_inv_sqrt(S, eps=1e-6):
    w, V = np.linalg.eigh(S)
    w = np.clip(w, eps, None)
    return V @ np.diag(1.0/np.sqrt(w)) @ V.T

def build_pf_blocks(X, p, f):
    T, m = X.shape
    N = T - (p + f) + 1
    if N <= 0:
        raise ValueError("Not enough samples for chosen p_lags + f_lags.")
    Yp, Yf = np.zeros((m*p, N)), np.zeros((m*f, N))
    for k in range(N):
        past = X[k:k+p, :]
        fut  = X[k+p:k+p+f, :]
        Yp[:, k] = past[::-1].reshape(-1)
        Yf[:, k] = fut.reshape(-1)
    return Yp, Yf

def fit_cva(Yp, Yf, energy_keep=0.9):
    M = Yp.shape[1]
    Spp = (Yp @ Yp.T) / (M - 1)
    Sff = (Yf @ Yf.T) / (M - 1)
    Sfp = (Yf @ Yp.T) / (M - 1)
    Spp_m12 = sym_inv_sqrt(Spp)
    Sff_m12 = sym_inv_sqrt(Sff)
    H = Sff_m12 @ Sfp @ Spp_m12
    U, D, Vt = np.linalg.svd(H, full_matrices=False)
    r = max(1, int(np.searchsorted(np.cumsum(D**2)/np.sum(D**2), energy_keep) + 1))
    print("CVA r =", r)
    # r=25
    Vr = Vt[:r, :].T
    J = Vr.T @ Spp_m12
    L = (np.eye(Spp_m12.shape[0]) - Vr @ Vr.T) @ Spp_m12
    return J, L

# def ucl_kde(values, alpha=0.99):
#     x = np.asarray(values).ravel()
#     if x.size == 0:
#         return 0.0
#     std = np.std(x)
#     if std == 0:
#         return float(np.max(x))
#     bw = 1.06 * std * (len(x) ** (-1/5))
#     kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(x.reshape(-1, 1))
#     grid = np.linspace(0, max(1e-6, x.max() * 1.5), 5000).reshape(-1, 1)
#     pdf = np.exp(kde.score_samples(grid))
#     cdf = np.cumsum(pdf); cdf /= cdf[-1]
#     idx = np.searchsorted(cdf, alpha)
#     return float(grid[min(idx, len(grid)-1)])

def ucl_kde(values, alpha=0.99):
    return float(np.quantile(values, alpha))

# ---------- Config (single target) ----------
key_targets = ['PT501']
all_cols = [
    'PT312','PT401','PT408','PT403','PT501','PT408_diff','PT403_diff',
    'FT305','FT104','FT407','LI405','FT406','FT407_density','FT406_density',
    'FT104_density','FT407_temp','FT406_temp','FT104_temp','LI504',
    'VC501','VC302','VC101','PO1','PT417'
]
window, horizon = 42, 1
hidden_size, epochs, batch_size, lr = 64, 50, 64, 1e-3
p_lags, f_lags = 15, 15

# ---------- Load baseline (normal = T2+T3) ----------
mat = loadmat("data/mat/Training.mat")  # adjust path if needed
df2 = pd.DataFrame(mat['T2'], columns=all_cols)
df3 = pd.DataFrame(mat['T3'], columns=all_cols)
df_train = pd.concat([df2, df3], ignore_index=True)

features = [c for c in all_cols if c not in key_targets]

# ---------- Scale & sequences ----------
scalerX, scalery = StandardScaler(), StandardScaler()
Xtr = scalerX.fit_transform(df_train[features].values)
ytr = scalery.fit_transform(df_train[key_targets].values)  # shape (T,1)
Xtr_seq, ytr_seq = make_sequences(Xtr, ytr, window, horizon)  # (N, win, m), (N,1)

# ---------- Train ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LSTMModel(input_size=Xtr_seq.shape[-1], hidden_size=hidden_size).to(device)
opt = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

loader = DataLoader(
    TensorDataset(torch.tensor(Xtr_seq, dtype=torch.float32),
                  torch.tensor(ytr_seq, dtype=torch.float32)),
    batch_size=batch_size, shuffle=True
)

model.train()
for _ in range(epochs):
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)                 # (B,1)
        loss = loss_fn(pred, yb)         # yb is (B,1)
        opt.zero_grad(); loss.backward(); opt.step()

# ---------- Residual UCLs on training ----------
model.eval()
with torch.no_grad():
    y_hat_std = model(torch.tensor(Xtr_seq, dtype=torch.float32).to(device)).cpu().numpy()
y_true = scalery.inverse_transform(ytr_seq)   # (N,1)
y_pred = scalery.inverse_transform(y_hat_std) # (N,1)
resid = y_true - y_pred                       # (N,1)

per_target = {'PT501': {"resid_ucl": residual_ucl(resid[:, 0], alpha=0.99)}}

# ---------- CVA fit & UCLs (inputs only) ----------
Yp_tr, Yf_tr = build_pf_blocks(Xtr, p_lags, f_lags)
J, L = fit_cva(Yp_tr, Yf_tr)
T2_tr = np.sum((J @ Yp_tr)**2, axis=0)
Q_tr  = np.sum((L @ Yp_tr)**2, axis=0)
T2_UCL, Q_UCL = ucl_kde(T2_tr, 0.995), ucl_kde(Q_tr, 0.995)

# ---------- Save artifacts ----------
joblib.dump(scalerX, "scalerX.pkl")
joblib.dump(scalery, "scalery.pkl")
torch.save(model.state_dict(), "model.pt")
np.save("cva_J.npy", J)
np.save("cva_L.npy", L)

with open("thresholds.json", "w") as f:
    json.dump({"per_target": per_target, "T2_UCL": T2_UCL, "Q_UCL": Q_UCL}, f, indent=2)

with open("config.json", "w") as f:
    json.dump({
        "targets": key_targets,          # ["PT501"]
        "features": features,
        "all_cols": all_cols,
        "window": window,
        "horizon": horizon,
        "hidden_size": hidden_size,
        "p_lags": p_lags,
        "f_lags": f_lags
    }, f, indent=2)

print("✅ Saved: model.pt, scalerX.pkl, scalery.pkl, cva_J.npy, cva_L.npy, thresholds.json, config.json")
