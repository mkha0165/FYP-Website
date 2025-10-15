# train_once.py  (Transformer-based version)
import json, joblib, numpy as np, pandas as pd, torch, math
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================================
# === Positional Encoding ===
# ==========================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ==========================================================
# === Transformer Regressor ===
# ==========================================================
class TransformerRegressor(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.2):
        super().__init__()
        self.input_fc = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        x = self.input_fc(x)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)
        last = out[:, -1, :]
        return self.fc_out(last)

# ==========================================================
# === Helper Functions ===
# ==========================================================
def make_sequences(X, y, window=42, horizon=1):
    Xs, ys = [], []
    for i in range(len(X) - window - horizon + 1):
        Xs.append(X[i:i+window])
        ys.append(y[i+window+horizon-1])
    return np.array(Xs), np.array(ys)

def residual_ucl(residuals, alpha=0.99):
    return float(np.quantile(np.abs(residuals), alpha))

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
    Vr = Vt[:r, :].T
    J = Vr.T @ Spp_m12
    L = (np.eye(Spp_m12.shape[0]) - Vr @ Vr.T) @ Spp_m12
    return J, L

def ucl_kde(values, alpha=0.99):
    return float(np.quantile(values, alpha))

# ==========================================================
# === Config ===
# ==========================================================
key_targets = ['PT501']
all_cols = [
    'PT312','PT401','PT408','PT403','PT501','PT408_diff','PT403_diff',
    'FT305','FT104','FT407','LI405','FT406','FT407_density','FT406_density',
    'FT104_density','FT407_temp','FT406_temp','FT104_temp','LI504',
    'VC501','VC302','VC101','PO1','PT417'
]
window, horizon = 42, 1
epochs, batch_size, lr = 80, 64, 1e-3
p_lags, f_lags = 15, 15

# ==========================================================
# === Load Training Data ===
# ==========================================================
mat = loadmat("data/mat/Training.mat")  # adjust path if needed
df2 = pd.DataFrame(mat['T2'], columns=all_cols)
df3 = pd.DataFrame(mat['T3'], columns=all_cols)
df_train = pd.concat([df2, df3], ignore_index=True)
features = [c for c in all_cols if c not in key_targets]

# ==========================================================
# === Scale and Sequence Data ===
# ==========================================================
scalerX, scalery = StandardScaler(), StandardScaler()
Xtr = scalerX.fit_transform(df_train[features].values)
ytr = scalery.fit_transform(df_train[key_targets].values)
Xtr_seq, ytr_seq = make_sequences(Xtr, ytr, window, horizon)

# ==========================================================
# === Train Transformer ===
# ==========================================================
model = TransformerRegressor(n_features=Xtr_seq.shape[-1]).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

loader = DataLoader(
    TensorDataset(torch.tensor(Xtr_seq, dtype=torch.float32),
                  torch.tensor(ytr_seq, dtype=torch.float32)),
    batch_size=batch_size, shuffle=True
)

print(f"🚀 Training Transformer model for target {key_targets[0]} on {DEVICE}...")
model.train()
for epoch in range(1, epochs + 1):
    epoch_losses = []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_losses.append(loss.item())
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | MSE: {np.mean(epoch_losses):.6f}")

# ==========================================================
# === Residual UCLs ===
# ==========================================================
model.eval()
with torch.no_grad():
    y_hat_std = model(torch.tensor(Xtr_seq, dtype=torch.float32).to(DEVICE)).cpu().numpy()
y_true = scalery.inverse_transform(ytr_seq)
y_pred = scalery.inverse_transform(y_hat_std)
resid = y_true - y_pred
per_target = {'PT501': {"resid_ucl": residual_ucl(resid[:, 0], alpha=0.99)}}

# ==========================================================
# === CVA (input-only) ===
# ==========================================================
Yp_tr, Yf_tr = build_pf_blocks(Xtr, p_lags, f_lags)
J, L = fit_cva(Yp_tr, Yf_tr)
T2_tr = np.sum((J @ Yp_tr)**2, axis=0)
Q_tr  = np.sum((L @ Yp_tr)**2, axis=0)
T2_UCL, Q_UCL = ucl_kde(T2_tr, 0.995), ucl_kde(Q_tr, 0.995)

# ==========================================================
# === Save Artifacts ===
# ==========================================================
joblib.dump(scalerX, "scalerX.pkl")
joblib.dump(scalery, "scalery.pkl")
torch.save(model.state_dict(), "model.pt")
np.save("cva_J.npy", J)
np.save("cva_L.npy", L)

with open("thresholds.json", "w") as f:
    json.dump({"per_target": per_target, "T2_UCL": T2_UCL, "Q_UCL": Q_UCL}, f, indent=2)

with open("config.json", "w") as f:
    json.dump({
        "targets": key_targets,
        "features": features,
        "all_cols": all_cols,
        "window": window,
        "horizon": horizon,
        "p_lags": p_lags,
        "f_lags": f_lags
    }, f, indent=2)

print("✅ Saved: model.pt, scalerX.pkl, scalery.pkl, cva_J.npy, cva_L.npy, thresholds.json, config.json")
