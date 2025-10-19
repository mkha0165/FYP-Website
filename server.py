# server.py
import json, io, os, re, warnings, joblib, torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from sklearn.exceptions import InconsistentVersionWarning

# Suppress version warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

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

TARGETS  = CFG["targets"]
FEATURES = CFG["features"]
ALL_COLS = CFG["all_cols"]
WINDOW   = int(CFG["window"])
HORIZON  = int(CFG["horizon"])
P_LAGS   = int(CFG["p_lags"])
F_LAGS   = int(CFG["f_lags"])

# ==========================================================
# === TransformerRegressor (same as in train_once.py) ===
# ==========================================================
import torch.nn as nn
import math

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

class TransformerRegressor(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.2, output_dim=1):
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
            nn.Linear(d_model // 2, output_dim)
        )

    def forward(self, x):
        x = self.input_fc(x)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)
        last = out[:, -1, :]
        return self.fc_out(last)


# ---------- Load trained model ----------
# MODEL = TransformerRegressor(n_features=len(FEATURES))
# MODEL.load_state_dict(torch.load("model.pt", map_location="cpu"))
# MODEL.eval()

# ---------- Multi-model setup ----------
MODEL_DIR = "./models"  # folder where you store model files (e.g. model_PT501.pt)

# Load all available models at startup
MODELS = {}
for target in CFG["targets"]:
    model_path = os.path.join(MODEL_DIR, f"model_{target}.pt")
    if os.path.exists(model_path):
        m = TransformerRegressor(n_features=len(FEATURES))
        m.load_state_dict(torch.load(model_path, map_location="cpu"))
        m.eval()
        MODELS[target] = m
    else:
        print(f"⚠️ Warning: model for {target} not found at {model_path}")


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

    # --- 6) Coerce numeric everywhere ---
    for c in expected_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- 7) Impute ONLY inputs; leave targets untouched ---
    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    # FIXED: Replace deprecated fillna method
    df[features] = df[features].ffill().bfill()

    if df[features].isna().any().any():
        raise HTTPException(status_code=400, detail="Missing values remain in input features after ffill/bfill.")

    return df


# ---------- API ----------
app = FastAPI(title="Soft Sensing API", version="1.0.0")

# CORS to allow front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:5500", "http://localhost:5500",
        "http://127.0.0.1:3000", "http://localhost:3000",
        "http://localhost:8080"  # Added for Node.js server
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LAST_RESULT = None  # optional cache for /report/pdf

# Add root route to fix 404
@app.get("/")
async def read_root():
    return {"message": "Soft Sensing API is running", "status": "OK"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

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
async def predict(file: UploadFile = File(...), target: str = "PT501"):
    global LAST_RESULT

    # Ensure model exists
    if target not in MODELS:
        raise HTTPException(status_code=400, detail=f"No trained model found for target '{target}'.")

    model = MODELS[target]  # <-- Use the preloaded model directly
    model.eval()

    # Load corresponding scalers
    scalerX = joblib.load("scalerX.pkl")
    scalery = joblib.load(f"scalers/scalery_{target}.pkl")

    # 1) Read CSV strictly and parse columns
    try:
        content = await file.read()
        df = read_ordered_csv_assign_names(content, ALL_COLS, TARGETS, FEATURES)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")

    # 2) Prepare inputs (features only)
    X = SCALER_X.transform(df[FEATURES].values)

    # 3) Create input sequences
    X_seq = make_sequences_X(X, WINDOW, HORIZON)
    if X_seq.size == 0:
        raise HTTPException(status_code=400, detail=f"Not enough rows for window={WINDOW}, horizon={HORIZON}.")

    # 4) Predict
    with torch.no_grad():
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        y_pred_std = model(X_tensor).cpu().numpy()

    # Ensure shape is consistent
    if y_pred_std.ndim == 1:
        y_pred_std = y_pred_std.reshape(-1, 1)

    # Inverse transform prediction to original scale
    y_pred = SCALER_Y.inverse_transform(y_pred_std)

    # 5) If truth values are present, compute residuals and metrics
    y_true_seq = None
    resid = None
    metrics = None
    has_truth_col = all(t in df.columns for t in TARGETS)
    if has_truth_col:
        y_full = df[TARGETS].values
        if not np.isnan(y_full).all():
            seq_truth = []
            T = len(y_full)
            for i in range(T - WINDOW - HORIZON + 1):
                seq_truth.append(y_full[i + WINDOW + HORIZON - 1])
            y_true_seq = np.array(seq_truth)

            # Drop NaN rows
            if np.isnan(y_true_seq).any():
                mask = ~np.isnan(y_true_seq).any(axis=1)
                y_true_seq = y_true_seq[mask]
                y_pred = y_pred[mask]

            # Compute metrics only for the selected target
            resid = np.abs(y_true_seq[:, TARGETS.index(target)] - y_pred[:, 0])
            yj = y_true_seq[:, TARGETS.index(target)]
            yhat = y_pred[:, 0]

            rmse = float(np.sqrt(np.mean((yj - yhat) ** 2)))
            mae  = float(np.mean(np.abs(yj - yhat)))
            ss_res = float(np.sum((yj - yhat) ** 2))
            ss_tot = float(np.sum((yj - np.mean(yj)) ** 2))
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

            metrics = {target: {"rmse": rmse, "mae": mae, "r2": r2}}


    # 6) CVA on inputs (scaled)
    Yp_te, _ = build_pf_blocks(X, P_LAGS, F_LAGS)
    if Yp_te.shape[1] > 0:
        Z = J @ Yp_te
        E = L @ Yp_te
        T2 = np.sum(Z * Z, axis=0)
        Q = np.sum(E * E, axis=0)
        breach_rate = float(np.mean((T2 > T2_UCL) | (Q > Q_UCL)))
        t_cva = list(range(len(T2)))
        cva = {
            "t": t_cva,
            "T2": T2.tolist(),
            "Q": Q.tolist(),
            "T2_UCL": T2_UCL,
            "Q_UCL": Q_UCL,
            "breach_rate": breach_rate,
        }
    else:
        cva = {"t": [], "T2": [], "Q": [], "T2_UCL": T2_UCL, "Q_UCL": Q_UCL, "breach_rate": 0.0}

    # 7) Build response
    t = list(range(len(y_pred)))
    result = {
        "targets": [target],
        "time_index": t,
        "y_pred": y_pred.tolist(),
        "y_true": None if y_true_seq is None else y_true_seq.tolist(),
        "resid_abs": None if resid is None else resid.tolist(),
        "resid_ucl": RESID_UCLS,
        "metrics": metrics,
        "cva": cva,
    }

    LAST_RESULT = result
    return JSONResponse(result)


def build_pdf_from_result(result: dict, shap_importance=None, target="TARGET"):
    from io import BytesIO
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica", 12)
    c.drawString(100, 800, f"Soft Sensor Report for Target: {target}")
    c.drawString(100, 780, "Summary metrics:")

    metrics = result.get("metrics", {})
    y = 760
    for name, m in metrics.items():
        c.drawString(120, y, f"{name}: RMSE={m['rmse']:.4f}, MAE={m['mae']:.4f}, R²={m['r2']:.4f}")
        y -= 20

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()

# ---------- Optional: on-demand PDF report ----------
@app.post("/report/pdf")
def make_report(shap: dict | None = Body(default=None)):
    global LAST_RESULT
    if LAST_RESULT is None:
        return Response(content="No prediction result available. Run /predict first.", status_code=400)
    pdf_bytes = build_pdf_from_result(LAST_RESULT, shap_importance=shap, target=TARGETS[0] if TARGETS else "TARGET")
    return Response(content=pdf_bytes, media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="softsensor_report.pdf"'})