# server.py
import json, io, os, re
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
import joblib

from report_generator import build_pdf_from_result

# ---------- Load artifacts at startup ----------
with open("config.json") as f:
    CFG = json.load(f)
with open("thresholds.json") as f:
    THR = json.load(f)

SCALER_X = joblib.load("scalerX.pkl")
SCALER_Y = joblib.load("scalery.pkl")
J = np.load("cva_J.npy")
L = np.load("cva_L.npy")
T2_UCL = float(THR["T2_UCL"])
Q_UCL = float(THR["Q_UCL"])
RESID_UCLS = {k: float(v["resid_ucl"]) for k, v in THR["per_target"].items()}

TARGETS  = CFG["targets"]           # e.g. ["PT501"]
FEATURES = CFG["features"]          # inputs only (target excluded)
ALL_COLS = CFG["all_cols"]
WINDOW   = int(CFG["window"])
HORIZON  = int(CFG["horizon"])
HIDDEN   = int(CFG["hidden_size"])
P_LAGS   = int(CFG["p_lags"])
F_LAGS   = int(CFG["f_lags"])

# ---------- Inference-only LSTM ----------
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

# ---------- Helpers to read CSV safely ----------
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

def read_ordered_csv_assign_names(file_bytes: bytes, expected_cols: list[str], targets: list[str], features: list[str]) -> pd.DataFrame:
    """
    Accept CSVs in any of these formats:
      A) No header (pure data rows)
      B) First row is numbering "1..n" (e.g., 1,2,...,24)
      C) First row is a named header that matches expected_cols
      D) First row is numbering, second row is named header

    Logic:
      - Always read with header=None (so we keep all rows).
      - If first row is numbering -> drop it.
      - Then, if (new) first row matches expected_cols -> drop it too.
      - Trim/validate column count.
      - Assign expected_cols as final names.
      - Coerce numeric; ONLY impute inputs (features); leave targets as-is.
    """
    import pandas as pd
    import numpy as np
    import re

    n_expected = len(expected_cols)
    df = pd.read_csv(io.BytesIO(file_bytes), header=None)

    # If there are more columns than expected, trim; if fewer, error
    if df.shape[1] < n_expected:
        raise HTTPException(
            status_code=400,
            detail=f"CSV has {df.shape[1]} columns but expected {n_expected}. "
                   f"Please export with exactly {n_expected} columns in the correct order."
        )
    if df.shape[1] > n_expected:
        df = df.iloc[:, :n_expected]

    def is_numbering_row(values, n_expected):
        pat = re.compile(r'^\s*(\d+)\s*[\.\)]?\s*$')
        nums = []
        for v in values:
            s = str(v).strip()
            m = pat.match(s)
            if not m:
                return False
            nums.append(int(m.group(1)))
        return nums == list(range(1, n_expected + 1))

    def normalize_tokens(vs):
        return [str(v).strip() for v in vs]

    def is_expected_header_row(values, expected):
        vals = normalize_tokens(values)
        exp  = [str(x).strip() for x in expected]
        return vals == exp

    # --- Drop first line if numbering 1..n ---
    if len(df) > 0 and is_numbering_row(df.iloc[0].tolist(), n_expected):
        df = df.iloc[1:].reset_index(drop=True)

    # --- Drop next line if it matches expected header names exactly ---
    if len(df) > 0 and is_expected_header_row(df.iloc[0].tolist(), expected_cols):
        df = df.iloc[1:].reset_index(drop=True)

    # Final sanity after possible drops
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV has no data rows after removing header/numbering lines.")

    # Assign final schema
    df.columns = expected_cols

    # Coerce numeric everywhere (targets may become NaN if blank)
    for c in expected_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Impute ONLY inputs; leave targets untouched
    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    df[features] = df[features].fillna(method="ffill").fillna(method="bfill")

    # Ensure inputs are clean
    if df[features].isna().any().any():
        raise HTTPException(status_code=400, detail="Missing values remain in input features after ffill/bfill.")

    return df


# ---------- API ----------
app = FastAPI(title="Soft Sensing API", version="1.0.0")

# CORS to allow your front-end (adjust for prod domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500", "http://localhost:5500",
        "http://127.0.0.1:3000", "http://localhost:3000",
        "https://fyp-website-xkq5.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LAST_RESULT = None  # optional cache for /report/pdf

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
    global LAST_RESULT

    # 1) Read CSV with strict schema; only impute inputs
    try:
        content = await file.read()
        df = read_ordered_csv_assign_names(content, ALL_COLS, TARGETS, FEATURES)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")

    # 2) Prepare inputs (features only)
    X = SCALER_X.transform(df[FEATURES].values)

    # 3) Sequences for LSTM
    X_seq = make_sequences_X(X, WINDOW, HORIZON)
    if X_seq.size == 0:
        raise HTTPException(status_code=400, detail=f"Not enough rows for window={WINDOW}, horizon={HORIZON}.")

    # 4) Predict target(s)
    with torch.no_grad():
        y_pred_std = MODEL(torch.tensor(X_seq, dtype=torch.float32)).cpu().numpy()  # (N, len(TARGETS))
    y_pred = SCALER_Y.inverse_transform(y_pred_std)

    # 5) Ground truth handling (ONLY if target column has real values)
    #    'has_truth' is true only if the target column exists and has at least some non-NaN values.
    y_true_seq = None
    resid = None
    metrics = None
    has_truth_col = all(t in df.columns for t in TARGETS)
    if has_truth_col:
        y_full = df[TARGETS].values
        # If ALL values are NaN, treat as no truth
        if not np.isnan(y_full).all():
            # Build aligned y_true for sequence outputs
            seq_truth = []
            T = len(y_full)
            for i in range(T - WINDOW - HORIZON + 1):
                seq_truth.append(y_full[i + WINDOW + HORIZON - 1])
            y_true_seq = np.array(seq_truth)  # (N, len(TARGETS))

            # Filter if any NaNs accidentally slipped in
            if np.isnan(y_true_seq).any():
                # drop rows with NaNs in truth to compute metrics cleanly
                mask = ~np.isnan(y_true_seq).any(axis=1)
                y_true_seq = y_true_seq[mask]
                y_pred = y_pred[mask]

            # Residuals & metrics
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

    # 6) CVA on inputs (scaled)
    Yp_te, _ = build_pf_blocks(X, P_LAGS, F_LAGS)
    if Yp_te.shape[1] > 0:
        Z = J @ Yp_te
        E = L @ Yp_te
        T2 = np.sum(Z * Z, axis=0)
        Q  = np.sum(E * E, axis=0)
        breach_rate = float(np.mean((T2 > T2_UCL) | (Q > Q_UCL)))
        t_cva = list(range(len(T2)))
        cva = {
            "t": t_cva, "T2": T2.tolist(), "Q": Q.tolist(),
            "T2_UCL": T2_UCL, "Q_UCL": Q_UCL, "breach_rate": breach_rate
        }
    else:
        cva = {"t": [], "T2": [], "Q": [], "T2_UCL": T2_UCL, "Q_UCL": Q_UCL, "breach_rate": 0.0}

    # 7) Build response (sequence output length)
    t = list(range(len(y_pred)))  # if each row = 1 second, you can convert to timestamps in frontend
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

# ---------- Optional: on-demand PDF report ----------
@app.post("/report/pdf")
def make_report(shap: dict | None = Body(default=None)):
    global LAST_RESULT
    if LAST_RESULT is None:
        return Response(content="No prediction result available. Run /predict first.", status_code=400)
    pdf_bytes = build_pdf_from_result(LAST_RESULT, shap_importance=shap, target=TARGETS[0] if TARGETS else "TARGET")
    return Response(content=pdf_bytes, media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="softsensor_report.pdf"'})
