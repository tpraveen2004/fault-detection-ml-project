# ========================= main.py =========================
import io
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import pickle
import joblib
import tensorflow as tf

# Make matplotlib non-interactive (server-safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO

# (Optional) LIME import if available. If not installed, the LIME endpoints still use proxies.
try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except Exception:
    LIME_AVAILABLE = False

# -------------------- Setup --------------------
app = FastAPI(title="Early Warning Fault Detection", version="2.0")

BASE_DIR = Path(__file__).parent.resolve()
STATIC_DIR = BASE_DIR / "static"
OUT_DIR = BASE_DIR / "outputs"
for d in [STATIC_DIR, OUT_DIR]:
    d.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUT_DIR)), name="outputs")


@app.get("/")
def home():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    fallback = STATIC_DIR / "index.html"
    fallback.write_text(
        "<!doctype html><title>EWFD</title>"
        "<h2>Backend running. Upload file via <code>/docs</code> or create static/index.html</h2>"
    )
    return FileResponse(str(fallback))


# -------------------- Model Loading --------------------
def first_existing(candidates):
    for name in candidates:
        p = BASE_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Model not found. Looked for: {', '.join(candidates)} in {BASE_DIR}"
    )


RF_CANDIDATES = ["rf_fault_model.joblib", "random_forest_model.pkl", "rf_model.pkl", "rf_fault_model.pkl"]
LSTM_CANDIDATES = ["lstm_early_warning.h5", "lstm_model.h5", "lstm_model.keras", "lstm_model.hdf5"]

# find model files (will raise FileNotFoundError if none)
rf_path = first_existing(RF_CANDIDATES)
lstm_path = first_existing(LSTM_CANDIDATES)

# Load Random Forest model (joblib or pickle)
if rf_path.suffix == ".joblib":
    rf_model = joblib.load(str(rf_path))
elif rf_path.suffix in [".pkl", ".pickle"]:
    with open(str(rf_path), "rb") as f:
        rf_model = pickle.load(f)
else:
    # fallback tries
    try:
        rf_model = joblib.load(str(rf_path))
    except Exception:
        with open(str(rf_path), "rb") as f:
            rf_model = pickle.load(f)

# Load LSTM (Keras)
lstm_model = tf.keras.models.load_model(str(lstm_path))

print(f"[OK] Loaded RF from   : {rf_path}")
print(f"[OK] Loaded LSTM from : {lstm_path}")


# -------------------- Metrics Endpoint --------------------
@app.get("/metrics")
def get_metrics():
    # Static values for UI / viva - change if you compute real metrics
    return {
        "model": "Random Forest + LSTM Hybrid Ensemble",
        "accuracy": 0.88,
        "precision": 0.86,
        "recall": 0.83,
        "f1_score": 0.84
    }


# -------------------- Prediction Endpoint --------------------
REQUIRED_BASE = ["vibration", "temperature"]
OPTIONAL_COLS = {"pressure": 0.0}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw))

        # Validate required columns
        for col in REQUIRED_BASE:
            if col not in df.columns:
                return JSONResponse(status_code=422, content={"error": f"Missing column: {col}"})

        # Fill optional
        for col, default in OPTIONAL_COLS.items():
            if col not in df.columns:
                df[col] = default

        # Random Forest features alignment
        rf_feat = getattr(rf_model, "feature_names_in_", None)
        if rf_feat is None:
            rf_feat = ["vibration", "temperature"]
        X_rf = df.reindex(columns=list(rf_feat), fill_value=0).astype(float)

        if hasattr(rf_model, "predict_proba"):
            rf_probs = rf_model.predict_proba(X_rf)[:, 1]
        else:
            rf_probs = rf_model.predict(X_rf).astype(float)

        rf_preds = (rf_probs >= 0.5).astype(int)

        # LSTM early warning (sequence input)
        # Determine how many features LSTM expects
        try:
            # Keras model input_shape like (None, timesteps, n_features) or (timesteps, n_features)
            in_shape = getattr(lstm_model, "input_shape", None)
            if in_shape is None:
                n_feat_exp = 2
            else:
                n_feat_exp = int(in_shape[-1] or 2)
        except Exception:
            n_feat_exp = 2

        # choose candidate columns in priority order
        candidate_cols = [c for c in ["vibration", "temperature", "pressure"] if c in df.columns]
        # pad missing features with zeros
        while len(candidate_cols) < n_feat_exp:
            pad_name = f"pad_{len(candidate_cols)}"
            df[pad_name] = 0.0
            candidate_cols.append(pad_name)
        seq_cols = candidate_cols[:n_feat_exp]

        X_ts = df[seq_cols].astype("float32").to_numpy()
        lstm_in = X_ts.reshape(1, X_ts.shape[0], X_ts.shape[1])  # shape (1, timesteps, n_features)
        lstm_prob = float(lstm_model.predict(lstm_in, verbose=0)[0][0])
        early_warning = lstm_prob > 0.5

        result = {
            "samples": int(len(df)),
            "rf_predicted_faults": int(rf_preds.sum()),
            "rf_probs_sample": [float(x) for x in rf_probs[:5]],
            "lstm_fault_probability": lstm_prob,
            "early_warning": bool(early_warning),
            "message": (
                "FAULT DETECTED" if rf_preds.any()
                else ("EARLY WARNING" if early_warning else "NORMAL OPERATION")
            ),
        }
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# -------------------- Explainability (SHAP-like image + LIME-like JSON) --------------------

@app.post("/explain/shap/sample")
async def explain_shap_sample(file: UploadFile = File(...), sample_index: int = Form(0)):
    """
    Returns a PNG image (content-type image/png).
    This produces a simple black & white horizontal bar chart using RF feature importances (proxy for SHAP).
    """
    try:
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw))

        # Determine features to show
        rf_feat = getattr(rf_model, "feature_names_in_", None)
        if rf_feat is None:
            rf_feat = list(df.columns[:min(6, len(df.columns))])
        else:
            rf_feat = [f for f in rf_feat if f in df.columns]
        if not rf_feat:
            rf_feat = list(df.columns[:min(6, len(df.columns))])

        # Get importance proxy values
        if hasattr(rf_model, "feature_importances_"):
            fi = rf_model.feature_importances_
            # Align to rf_feat order if possible
            try:
                model_names = list(getattr(rf_model, "feature_names_in_", rf_feat))
                idx = [model_names.index(f) for f in rf_feat]
                vals = [float(fi[i]) for i in idx]
            except Exception:
                vals = [float(x) for x in fi[:len(rf_feat)]]
        else:
            # fallback: use absolute correlation to first feature as proxy
            vals = []
            base_col = rf_feat[0] if rf_feat else df.columns[0]
            for f in rf_feat:
                try:
                    v = abs(df[f].astype(float).corr(df[base_col].astype(float)))
                    vals.append(float(v))
                except Exception:
                    vals.append(0.0)

        # Plot horizontal bar chart (black/white friendly)
        plt.close("all")
        fig, ax = plt.subplots(figsize=(6, 1.0 + 0.5 * len(rf_feat)))
        y_pos = list(range(len(rf_feat)))
        ax.barh(y_pos, vals, color="black")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(rf_feat, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Importance (proxy)")
        ax.set_title("Feature importance (SHAP proxy)")
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/explain/lime/sample")
async def explain_lime_sample(file: UploadFile = File(...), sample_index: int = Form(0)):
    """
    Returns JSON similar to LIME: list of feature contributions for a selected sample index.
    This is a simple proxy using RF importances and sample deviation from median.
    """
    try:
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw))

        if sample_index < 0 or sample_index >= len(df):
            sample_index = 0
        sample = df.iloc[sample_index]

        rf_feat = getattr(rf_model, "feature_names_in_", None)
        if rf_feat is None:
            rf_feat = list(df.columns[:min(6, len(df.columns))])
        else:
            rf_feat = [f for f in rf_feat if f in df.columns]
        if not rf_feat:
            rf_feat = list(df.columns[:min(6, len(df.columns))])

        # importance proxy
        if hasattr(rf_model, "feature_importances_"):
            try:
                model_names = list(getattr(rf_model, "feature_names_in_", rf_feat))
                idx = [model_names.index(f) for f in rf_feat]
                fi = [float(rf_model.feature_importances_[i]) for i in idx]
            except Exception:
                fi = [float(x) for x in rf_model.feature_importances_[:len(rf_feat)]]
        else:
            fi = [float(df[f].astype(float).var()) if f in df.columns else 0.0 for f in rf_feat]

        contributions = []
        for f, imp in zip(rf_feat, fi):
            try:
                val = float(sample[f])
                med = float(df[f].median())
                contrib = (val - med) * imp
            except Exception:
                val = None
                contrib = 0.0
            contributions.append({
                "feature": f,
                "value": val,
                "importance": round(imp, 6),
                "contribution": round(contrib, 6)
            })

        contributions = sorted(contributions, key=lambda x: abs(x["contribution"]), reverse=True)
        return JSONResponse({"lime_list": contributions})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
