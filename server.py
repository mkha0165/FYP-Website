# server.py
import json, io
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import re, io

# ---------- Load artifacts at startup ----------
with open("config.json") as f: CFG = json.load(f)
with open("thresholds.json") as f: THR = json.load(f)

SCALER_X = joblib.load("scalerX.pkl")
SCALER_Y = joblib.load("scalery.pkl")
J = np.load("cva_J.npy")
L = np.load("cva_L.npy")
T2_UCL = THR["T2_UCL"]
Q_UCL = THR["Q_UCL"]
RESID_UCLS = {k: v["resid_ucl"] for k, v in THR["per_target"].items()}

# Small inference-only LSTM to match training config
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
    input_size=len(CFG["features"]),
    output_size=len(CFG["targets"]),
    hidden_size=CFG["hidden_size"]
)
MODEL.load_state_dict(torch.load("model.pt", map_location="cpu"))
MODEL.eval()

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

# ---------- API ----------
app = FastAPI(title="Soft Sensing API", version="1.0.0")

def is_numbering_row(values, n_expected):
    """
    True if row looks like 1..n (accepts '1', '1.', '1)', with spaces).
    """
    if len(values) != n_expected:
        return False
    pat = re.compile(r'^\s*(\d+)\s*[\.\)]?\s*$')
    nums = []
    for v in values:
        s = str(v).strip()
        m = pat.match(s)
        if not m:
            return False
        nums.append(int(m.group(1)))
    return nums == list(range(1, n_expected + 1))

def read_ordered_csv_assign_names(file_bytes: bytes, expected_cols: list[str]) -> pd.DataFrame:
    """
    Read a CSV where users guarantee column order.
    - Treat file as headerless.
    - If first row is numbering (1..n), drop it.
    - Assign expected column names.
    - Coerce to numeric and simple NaN fills.
    """
    n_expected = len(expected_cols)

    # Read raw (no header)
    df = pd.read_csv(io.BytesIO(file_bytes), header=None)

    # Validate column count early
    if df.shape[1] != n_expected:
        raise HTTPException(
            status_code=400,
            detail=f"CSV has {df.shape[1]} columns but expected {n_expected}. "
                   f"Please export with exactly {n_expected} columns in the correct order."
        )

    # If first row is numbering 1..n → drop it
    first_row_vals = df.iloc[0].tolist()
    if is_numbering_row(first_row_vals, n_expected):
        df = df.iloc[1:].reset_index(drop=True)

    # Assign names & clean
    df.columns = expected_cols
    df = df.apply(pd.to_numeric, errors="coerce").fillna(method="ffill").fillna(method="bfill")

    # Final sanity: need at least window+horizon rows
    return df

# Allow your frontend origin(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],        # change to ["https://yourdomain.com"] in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictResponse(BaseModel):
    targets: list[str]
    time_index: list[int]
    y_pred: list[list[float]]
    y_true: list[list[float]] | None
    resid_abs: list[list[float]] | None
    resid_ucl: dict[str, float]
    metrics: dict[str, dict] | None
    cva: dict

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    # Read CSV into DataFrame
    # Read CSV (order-guaranteed, headerless or with numbering row)
    try:
        content = await file.read()
        df = read_ordered_csv_assign_names(content, CFG["all_cols"])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")

    # Validate / align columns
    missing = [c for c in CFG["all_cols"] if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")
    df = df[CFG["all_cols"]].apply(pd.to_numeric, errors="coerce").fillna(method="ffill").fillna(method="bfill")

    # Inputs
    X = SCALER_X.transform(df[CFG["features"]].values)

    # Sequence for LSTM
    W, H = CFG["window"], CFG["horizon"]
    X_seq = make_sequences_X(X, W, H)
    if X_seq.size == 0:
        raise HTTPException(status_code=400, detail="Not enough rows for the configured window/horizon.")

    with torch.no_grad():
        y_pred_std = MODEL(torch.tensor(X_seq, dtype=torch.float32)).cpu().numpy()
    y_pred = SCALER_Y.inverse_transform(y_pred_std)  # (N, T)

    # If CSV also contains true target columns, compute residuals/metrics
    has_truth = set(CFG["targets"]).issubset(df.columns)
    y_true_seq = None
    resid = None
    metrics = None

    if has_truth:
        y_full = df[CFG["targets"]].values
        y_true_seq = []
        T = len(y_full)
        for i in range(T - W - H + 1):
            y_true_seq.append(y_full[i+W+H-1])
        y_true_seq = np.array(y_true_seq)
        resid = np.abs(y_true_seq - y_pred)

        # Per-target metrics
        metrics = {}
        for j, name in enumerate(CFG["targets"]):
            yj = y_true_seq[:, j]; yhat = y_pred[:, j]
            rmse = float(np.sqrt(np.mean((yj - yhat)**2)))
            mae  = float(np.mean(np.abs(yj - yhat)))
            ss_res = float(np.sum((yj - yhat)**2))
            ss_tot = float(np.sum((yj - np.mean(yj))**2))
            r2 = float(1 - ss_res/ss_tot) if ss_tot > 0 else float("nan")
            metrics[name] = {"rmse": rmse, "mae": mae, "r2": r2}

    # CVA on inputs
    p, f = CFG["p_lags"], CFG["f_lags"]
    Yp_te, _ = build_pf_blocks(X, p, f)
    if Yp_te.shape[1] > 0:
        Z = J @ Yp_te
        E = L @ Yp_te
        T2 = np.sum(Z*Z, axis=0)
        Q  = np.sum(E*E, axis=0)
        breach_rate = float(np.mean((T2 > T2_UCL) | (Q > Q_UCL)))
        t_cva = list(range(len(T2)))
        cva = {"t": t_cva, "T2": T2.tolist(), "Q": Q.tolist(),
               "T2_UCL": float(T2_UCL), "Q_UCL": float(Q_UCL),
               "breach_rate": breach_rate}
    else:
        cva = {"t": [], "T2": [], "Q": [], "T2_UCL": float(T2_UCL), "Q_UCL": float(Q_UCL), "breach_rate": 0.0}

    # Build response
    t = list(range(len(y_pred)))  # time index for seq outputs
    return {
        "targets": CFG["targets"],
        "time_index": t,
        "y_pred": y_pred.tolist(),
        "y_true": None if y_true_seq is None else y_true_seq.tolist(),
        "resid_abs": None if resid is None else resid.tolist(),
        "resid_ucl": RESID_UCLS,
        "metrics": metrics,
        "cva": cva
    }
