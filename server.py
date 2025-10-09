# server.py
import json, io, os, re, warnings
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
from sklearn.exceptions import InconsistentVersionWarning

# Silence version chatter from sklearn joblib loads
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ---------- Load artifacts at startup ----------
with open("config.json") as f:
    CFG = json.load(f)
with open("thresholds.json") as f:
    THR = json.load(f)

SCALER_X = joblib.load("scalerX.pkl")
SCALER_Y = joblib.load("scalery.pkl")

# CVA matrices (case-sensitive!)
J = np.load("cva_J.npy")                      # (r, m*p)
L = np.load("cva_L.npy")                      # (m*p, m*p)
D = np.load("cva_D.npy") if os.path.exists("cva_D.npy") else np.ones(J.shape[0])  # (r,)

T2_UCL = float(THR["T2_UCL"])
Q_UCL  = float(THR["Q_UCL"])
RESID_UCLS = {k: float(v["resid_ucl"]) for k, v in THR["per_target"].items()}

TARGETS  = CFG["targets"]           # e.g. ["PT501"]
FEATURES = CFG["features"]          # inputs only (target excluded)
ALL_COLS = CFG["all_cols"]
WINDOW   = int(CFG["window"])
HORIZON  = int(CFG["horizon"])
HIDDEN   = int(CFG["hidden_size"])
P_LAGS   = int(CFG["p_lags"])
F_LAGS   = int(CFG["f_lags"])

# ---------- Inference LSTM ----------
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size // 2, output_size),
        )
    def forward(self, x):
        y, _ = self.lstm(x)
        y = y[:, -1, :]
        return self.fc(y)

MODEL = LSTMModel(
    input_size=len(FEATURES),
    output_size=len(TARGETS),
    hidden_size=HIDDEN
)
MODEL.load_state_dict(torch.load("model.pt", map_location="cpu"))
MODEL.eval()

# ---------- Helpers ----------
def make_sequences_X(X, window, horizon):
    Xs = []
    for i in range(len(X) - window - horizon + 1):
        Xs.append(X[i:i+window])
    return np.array(Xs)

def build_pf_blocks(X, p, f):
    T, m = X.shape
    N = T - (p + f) + 1
    if N <= 0:
        return np.zeros((m*p, 0)), np.zeros((m*f, 0))
    Yp, Yf = np.zeros((m*p, N)), np.zeros((m*f, N))
    for k in range(N):
        past = X[k:k+p, :]
        fut  = X[k+p:k+p+f, :]
        Yp[:, k] = past[::-1].reshape(-1)
        Yf[:, k] = fut.reshape(-1)
    return Yp, Yf

def variable_lag_indices(m: int, p: int):
    # rows for each variable across all p lags in Yp
    idxs = []
    for j in range(m):
        rows = []
        for lag in range(p):
            rows.append(lag*m + j)
        idxs.append(np.asarray(rows, dtype=int))
    return idxs

def read_ordered_csv_assign_names(file_bytes: bytes, expected_cols: list[str], targets: list[str], features: list[str]) -> pd.DataFrame:
    # Read with sniffed delimiter, no header
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), header=None, engine="python", sep=None)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unable to read CSV: {e}")

    n_expected = len(expected_cols)
    if df.shape[1] < n_expected:
        raise HTTPException(status_code=400, detail=f"CSV has {df.shape[1]} columns but expected {n_expected}.")
    if df.shape[1] > n_expected:
        df = df.iloc[:, :n_expected]

    def _clean(s): return str(s).replace("\ufeff", "").strip()
    def is_numbering_row(values, n_expected):
        import re
        toks = [_clean(v) for v in values]
        ints = []
        for t in toks:
            m = re.match(r"^(\d+)(?:\.0+)?\s*[\.\)]?$", t)
            if not m:
                return False
            ints.append(int(m.group(1)))
        return (ints == list(range(1, n_expected + 1))) or (ints == list(range(0, n_expected)))
    def is_expected_header(values, expected):
        return [_clean(v) for v in values] == [str(x).strip() for x in expected]

    if len(df) and is_numbering_row(df.iloc[0].tolist(), n_expected):
        df = df.iloc[1:].reset_index(drop=True)
    if len(df) and is_expected_header(df.iloc[0].tolist(), expected_cols):
        df = df.iloc[1:].reset_index(drop=True)
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV contained no usable rows after cleaning.")

    df.columns = expected_cols

    # Features numeric and imputed; targets numeric but allow NaN
    for c in features:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[features] = df[features].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    if df[features].isna().any().any():
        raise HTTPException(status_code=400, detail="Missing values remain in input features after ffill/bfill.")
    for tcol in targets:
        if tcol in df.columns:
            df[tcol] = pd.to_numeric(df[tcol], errors="coerce")
    return df

# ---------- API ----------
app = FastAPI(title="Soft Sensing API", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500", "http://localhost:5500",
        "http://127.0.0.1:3000", "http://localhost:3000",
        "http://127.0.0.1:8000", "http://localhost:8000",
        "http://localhost:5000",  # your Node/Express dev
        "*"                       # loosen for quick tests; tighten in prod
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LAST_RESULT = None  # cached latest prediction (small pointer, not huge history)

@app.get("/")
def root(): return {"message": "Soft Sensing API is running", "status": "OK"}

@app.get("/health")
def health(): return {"status": "healthy"}

class PredictResponse(BaseModel):
    targets: list[str]
    time_index: list[int]
    y_pred: list[list[float]]
    y_true: list[list[float]] | None
    resid_abs: list[list[float]] | None
    resid_ucl: dict[str, float]
    metrics: dict[str, dict] | None
    cva: dict

@app.get("/result/latest")
def result_latest():
    if LAST_RESULT is None:
        raise HTTPException(status_code=404, detail="No result cached yet")
    return JSONResponse(LAST_RESULT)

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    global LAST_RESULT

    # 1) Read CSV
    try:
        content = await file.read()
        df = read_ordered_csv_assign_names(content, ALL_COLS, TARGETS, FEATURES)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")

    # 2) Inputs
    X = SCALER_X.transform(df[FEATURES].values)

    # 3) LSTM sequences & predict
    X_seq = make_sequences_X(X, WINDOW, HORIZON)
    if X_seq.size == 0:
        raise HTTPException(status_code=400, detail=f"Not enough rows for window={WINDOW}, horizon={HORIZON}.")
    with torch.no_grad():
        y_pred_std = MODEL(torch.tensor(X_seq, dtype=torch.float32)).cpu().numpy()
    y_pred = SCALER_Y.inverse_transform(y_pred_std)

    # 4) Ground truth (optional)
    y_true_seq = None
    resid = None
    metrics = None
    if all(t in df.columns for t in TARGETS):
        y_full = df[TARGETS].values
        if not np.isnan(y_full).all():
            seq_truth = []
            T = len(y_full)
            for i in range(T - WINDOW - HORIZON + 1):
                seq_truth.append(y_full[i + WINDOW + HORIZON - 1])
            y_true_seq = np.array(seq_truth)
            if np.isnan(y_true_seq).any():
                mask = ~np.isnan(y_true_seq).any(axis=1)
                y_true_seq = y_true_seq[mask]
                y_pred = y_pred[mask]
            resid = np.abs(y_true_seq - y_pred)
            metrics = {}
            for j, name in enumerate(TARGETS):
                yj = y_true_seq[:, j]; yhat = y_pred[:, j]
                rmse = float(np.sqrt(np.mean((yj - yhat) ** 2)))
                mae  = float(np.mean(np.abs(yj - yhat)))
                ss_res = float(np.sum((yj - yhat) ** 2))
                ss_tot = float(np.sum((yj - np.mean(yj)) ** 2))
                r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
                metrics[name] = {"rmse": rmse, "mae": mae, "r2": r2}

    # 5) CVA (T2/Q) + Contributions
    Yp_te, _ = build_pf_blocks(X, P_LAGS, F_LAGS)
    if Yp_te.shape[1] > 0:
        m = len(FEATURES)
        mp = m * P_LAGS
        if Yp_te.shape[0] != mp or J.shape[1] != mp or L.shape != (mp, mp):
            raise HTTPException(
                status_code=400,
                detail=f"CVA mismatch: Yp[{Yp_te.shape[0]} x N], J{J.shape}, L{L.shape}, expected m*p={mp}."
            )
        Z = J @ Yp_te               # (r, N)
        E = L @ Yp_te               # (mp, N)
        T2 = np.sum(Z * Z, axis=0)
        Q  = np.sum(E * E, axis=0)

        # ---- Contributions ----
        # Retained space contribution: C_retained = ( (Z / D)ᵀ @ J )ᵀ  -> (mp x N)
        invD = 1.0 / np.clip(D, 1e-12, None)       # (r,)
        W = (Z.T * invD).T                          # (r, N) divide rows by γ_i
        C_retained = (W.T @ J).T                    # (mp, N)

        # Fold lag rows (mp) into variables (m) by summing over p lags
        idxs = variable_lag_indices(m, P_LAGS)
        C_total = np.zeros((m, C_retained.shape[1]))
        for jv, rows in enumerate(idxs):
            C_total[jv, :] = np.sum(C_retained[rows, :], axis=0)

        # Residual-space contribution for Q (sum of squared residual lag rows per variable)
        E2 = E * E
        C_q = np.zeros((m, E2.shape[1]))
        for jv, rows in enumerate(idxs):
            C_q[jv, :] = np.sum(E2[rows, :], axis=0)

        breach_rate = float(np.mean((T2 > T2_UCL) | (Q > Q_UCL)))
        cva = {
            "t": list(range(len(T2))),
            "T2": T2.tolist(),
            "Q": Q.tolist(),
            "T2_UCL": T2_UCL,
            "Q_UCL": Q_UCL,
            "breach_rate": breach_rate,
            "contrib": {
                "features": FEATURES,
                "C_total": C_total.tolist(),
                "C_q": C_q.tolist()
            }
        }
    else:
        cva = {
            "t": [], "T2": [], "Q": [],
            "T2_UCL": T2_UCL, "Q_UCL": Q_UCL, "breach_rate": 0.0,
            "contrib": { "features": FEATURES, "C_total": [], "C_q": [] }
        }

    # 6) Build response
    t = list(range(len(y_pred)))
    result = {
        "targets": TARGETS,
        "time_index": t,
        "y_pred": y_pred.tolist(),
        "y_true": None if y_true_seq is None else y_true_seq.tolist(),
        "resid_abs": None if resid is None else resid.tolist(),
        "resid_ucl": RESID_UCLS,
        "metrics": metrics,
        "cva": cva
    }
    LAST_RESULT = result
    return JSONResponse(result)
