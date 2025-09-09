# shap_pt501.py
import json, joblib, numpy as np, pandas as pd, torch
from torch import nn
from scipy.io import loadmat

# ---- 0) Model (must match your training code) ----
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
            nn.Linear(hidden_size // 2, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)            # (B, T, H)
        out = out[:, -1, :]              # last time step
        out = self.dropout(out)
        return self.fc(out)              # (B, 1)

def make_sequences(X, window=42, horizon=1):
    Xs = []
    for i in range(len(X) - window - horizon + 1):
        Xs.append(X[i:i+window])
    return np.array(Xs)

# ---- 1) Load config & artifacts ----
with open("config.json") as f:
    CFG = json.load(f)
features   = CFG["features"]           # inputs only (PT501 excluded)
window     = CFG["window"]
hidden_size= CFG["hidden_size"]

scalerX = joblib.load("scalerX.pkl")
device  = "cuda" if torch.cuda.is_available() else "cpu"

model = LSTMModel(input_size=len(features), hidden_size=hidden_size).to(device)
state = torch.load("model.pt", map_location=device)
model.load_state_dict(state)
model.eval()

# ---- 2) Build data for SHAP (use normal data T2+T3; eval on T1) ----
# Adjust path/name if needed
mat = loadmat("data/mat/Training.mat")
cols = CFG["all_cols"]
df2 = pd.DataFrame(mat["T2"], columns=cols)  # normal background
df3 = pd.DataFrame(mat["T3"], columns=cols)  # normal background
df1 = pd.DataFrame(mat["T1"], columns=cols)  # evaluation split (could be normal/faulty)

# Scale inputs
Z_bg = scalerX.transform(pd.concat([df2, df3], ignore_index=True)[features].values)
Z_ev = scalerX.transform(df1[features].values)

# Sequences
X_bg = make_sequences(Z_bg, window=window, horizon=1)  # (Nb, T, m)
X_ev = make_sequences(Z_ev, window=window, horizon=1)  # (Ne, T, m)

# Keep SHAP reasonably light
Nb = min(200, len(X_bg))     # background samples for SHAP
Ne = min(200, len(X_ev))     # evaluation samples for SHAP
if Nb == 0 or Ne == 0:
    raise RuntimeError("Not enough rows to form sequences. Reduce 'window' or provide more data.")

bg  = torch.tensor(X_bg[:Nb], dtype=torch.float32).to(device)
ev  = torch.tensor(X_ev[:Ne], dtype=torch.float32).to(device)

# ---- 3) SHAP DeepExplainer ----
# Note: install shap if needed: `pip install shap`
import shap
# Some PyTorch models need hooks; DeepExplainer supports many standard graphs.
explainer = shap.GradientExplainer(model, bg)
shap_vals = explainer.shap_values(ev)
# For a single-output regression, DeepExplainer returns a list with one array.
if isinstance(shap_vals, list):
    shap_vals = shap_vals[0]  # shape: (Ne, T, m)

# ---- 4) Aggregate |SHAP| over time window to get per-feature impact ----
# abs over samples & time, then mean per feature
# shap_vals: (Ne, T, m)
abs_vals = np.abs(shap_vals)                  # (Ne, T, m)
per_feat = abs_vals.mean(axis=(0,1))          # (m, )

# Rank features
feat_scores = sorted(
    [{"feature": f, "score": float(s)} for f, s in zip(features, per_feat)],
    key=lambda d: d["score"], reverse=True
)

# ---- 5) Save results ----
with open("shap_importance_pt501.json", "w") as f:
    json.dump({
        "target": "PT501",
        "window": window,
        "num_background": int(Nb),
        "num_eval": int(Ne),
        "feature_importance": feat_scores
    }, f, indent=2)

print("✅ Saved shap_importance_pt501.json")
# Optional: also save raw SHAP for a few eval sequences (for waterfall/heatmaps)
np.save("shap_values_eval.npy", shap_vals)     # (Ne, T, m)
np.save("eval_sequences.npy", X_ev[:Ne])       # (Ne, T, m)
