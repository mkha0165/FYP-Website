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
import zipfile

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

def read_ordered_csv_assign_names(
    file_bytes: bytes,
    expected_cols: list[str],
    targets: list[str],
    features: list[str],
) -> pd.DataFrame:
    """
    Accept CSVs in formats:
      A) No header
      B) First row is numbering 1..n (or 0..n-1), e.g. 1,2,3,...,24 (also '1.' or '1)')
      C) First row exactly equals expected column names
      D) Numbering row then expected header row

    Steps:
      - Read with header=None (keep all rows)
      - Sniff delimiter
      - If first row is numbering -> drop it
      - If (new) first row matches expected header -> drop it
      - Trim to expected column count and assign names
      - Coerce numeric; ONLY impute inputs; leave targets as-is (can be blank)
    """
    import re
    # --- 1) Read with delimiter sniffing ---
    df = pd.read_csv(io.BytesIO(file_bytes), header=None, engine="python", sep=None)

    n_expected = len(expected_cols)

    # --- 2) Trim/validate column count ---
    if df.shape[1] < n_expected:
        raise HTTPException(
            status_code=400,
            detail=f"CSV has {df.shape[1]} columns but expected {n_expected}. "
                   f"Please export exactly {n_expected} columns in the trained order."
        )
    if df.shape[1] > n_expected:
        df = df.iloc[:, :n_expected]

    # --- helpers ---
    def _clean_token(s):
        return str(s).replace("\ufeff", "").strip()  # strip BOM + spaces

    def _row_tokens(row):
        return [_clean_token(v) for v in row.tolist()]

    def is_numbering_row(values, n_expected):
        """True if values equal 1..n or 0..n-1 (allow '1', '1.', '1)', '1.0')."""
        toks = [_clean_token(v) for v in values]
        ints = []
        for t in toks:
            m = re.match(r"^(\d+)(?:\.0+)?\s*[\.\)]?$", t)  # 1, 1., 1), 1.0
            if not m:
                return False
            ints.append(int(m.group(1)))
        return (ints == list(range(1, n_expected + 1))) or (ints == list(range(0, n_expected)))

    def is_expected_header_row(values, expected):
        toks = [_clean_token(v) for v in values]
        exp  = [str(x).strip() for x in expected]
        return toks == exp

    # --- 3) Drop numbered header if present ---
    if len(df) > 0 and is_numbering_row(df.iloc[0].values, n_expected):
        df = df.iloc[1:].reset_index(drop=True)

    # --- 4) Drop named header if present (after possible numbering drop) ---
    if len(df) > 0 and is_expected_header_row(df.iloc[0].values, expected_cols):
        df = df.iloc[1:].reset_index(drop=True)

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV has no data rows after removing header/numbering line(s).")

    # --- 5) Assign schema ---
    df.columns = expected_cols

    # --- 6) Coerce numeric everywhere (targets may become NaN; thats OK) ---
    for c in expected_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- 7) Impute ONLY inputs; leave targets untouched ---
    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    df[features] = df[features].fillna(method="ffill").fillna(method="bfill")

    if df[features].isna().any().any():
        raise HTTPException(status_code=400, detail="Missing values remain in input features after ffill/bfill.")

    return df


# ---------- API ----------
app = FastAPI(title="Soft Sensing API", version="1.0.0")

# CORS to allow your front-end (adjust for prod domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "https://fyp-website-xkq5.onrender.com",  # frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Hello! Your API is running"}

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

import zipfile

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    global LAST_RESULT
    import gc

    try:
        # --- 1) Ensure it's a .zip file ---
        if not file.filename.lower().endswith(".zip"):
            raise HTTPException(status_code=400, detail="Please upload a .zip file containing a single CSV.")

        content = await file.read()

        # --- 2) Open the zip ---
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                csv_files = [name for name in zf.namelist() if name.lower().endswith(".csv")]
                if len(csv_files) != 1:
                    raise HTTPException(status_code=400,
                        detail=f"Zip must contain exactly 1 CSV, found {len(csv_files)}.")
                csv_bytes = zf.read(csv_files[0])
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid zip archive.")

        # --- 3) Parse CSV with strict schema ---
        df = read_ordered_csv_assign_names(csv_bytes, ALL_COLS, TARGETS, FEATURES)

        # --- 4) Prepare inputs ---
        X = SCALER_X.transform(df[FEATURES].values)
        del df, content, csv_bytes  # free memory early

        # --- 5) Sequences for LSTM ---
        X_seq = make_sequences_X(X, WINDOW, HORIZON)
        if X_seq.size == 0:
            raise HTTPException(status_code=400,
                                detail=f"Not enough rows for window={WINDOW}, horizon={HORIZON}.")

        # --- 6) Predict ---
        with torch.no_grad():
            y_pred_std = MODEL(torch.tensor(X_seq, dtype=torch.float32)).cpu().numpy()
        y_pred = SCALER_Y.inverse_transform(y_pred_std)
        del y_pred_std, X_seq

        # --- 7) Handle ground truth if available ---
        y_true_seq, resid, metrics = None, None, None
        if all(t in ALL_COLS for t in TARGETS):
            y_full = df[TARGETS].values if "df" in locals() else None
            if y_full is not None and not np.isnan(y_full).all():
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
                    yj, yhat = y_true_seq[:, j], y_pred[:, j]
                    rmse = float(np.sqrt(np.mean((yj - yhat) ** 2)))
                    mae  = float(np.mean(np.abs(yj - yhat)))
                    ss_res = float(np.sum((yj - yhat) ** 2))
                    ss_tot = float(np.sum((yj - np.mean(yj)) ** 2))
                    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
                    metrics[name] = {"rmse": rmse, "mae": mae, "r2": r2}

        # --- 8) CVA ---
        Yp_te, _ = build_pf_blocks(X, P_LAGS, F_LAGS)
        cva = {"t": [], "T2": [], "Q": [], "T2_UCL": T2_UCL,
               "Q_UCL": Q_UCL, "breach_rate": 0.0}
        if Yp_te.shape[1] > 0:
            Z = J @ Yp_te
            E = L @ Yp_te
            T2, Q = np.sum(Z * Z, axis=0), np.sum(E * E, axis=0)
            breach_rate = float(np.mean((T2 > T2_UCL) | (Q > Q_UCL)))
            cva = {
                "t": list(range(len(T2))),
                "T2": T2.tolist(),
                "Q": Q.tolist(),
                "T2_UCL": T2_UCL,
                "Q_UCL": Q_UCL,
                "breach_rate": breach_rate
            }

        # --- 9) Build result ---
        result = {
            "targets": TARGETS,
            "time_index": list(range(len(y_pred))),
            "y_pred": y_pred.tolist(),
            "y_true": None if y_true_seq is None else y_true_seq.tolist(),
            "resid_abs": None if resid is None else resid.tolist(),
            "resid_ucl": RESID_UCLS,
            "metrics": metrics,
            "cva": cva
        }
        LAST_RESULT = result

        # cleanup
        del X, Yp_te, Z, E, T2, Q, y_pred, y_true_seq, resid
        gc.collect()
        torch.cuda.empty_cache()

        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print("Error in /predict:", e)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()}
        )


