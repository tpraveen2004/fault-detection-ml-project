# app_api.py
import os
import io
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

app = FastAPI(title="Fault Detection API", version="1.0")

# ---- Configure paths (edit BASE if your folder is different) ----
BASE = r"B:\projects\early-warning-fault-detection"
RF_PATH = os.path.join(BASE, "outputs", "rf_fault_model.joblib")
LSTM_PATH = os.path.join(BASE, "outputs", "lstm_early_warning.h5")

# ---- Lazy-load models with helpful errors ----
rf_model = None
lstm_model = None
tf_ok = True
try:
    import tensorflow as tf  # noqa
except Exception as _e:
    tf_ok = False

def load_models_once():
    global rf_model, lstm_model
    errors = []
    if rf_model is None:
        if os.path.exists(RF_PATH):
            rf_model = joblib.load(RF_PATH)
        else:
            errors.append(f"Missing RF model: {RF_PATH}")
    if lstm_model is None:
        if not tf_ok:
            errors.append("TensorFlow not available in this environment.")
        elif os.path.exists(LSTM_PATH):
            from tensorflow.keras.models import load_model
            lstm_model = load_model(LSTM_PATH)
        else:
            errors.append(f"Missing LSTM model: {LSTM_PATH}")
    return errors

REQUIRED_COLS = ["vibration", "temperature"]  # 'pressure' optional

@app.post("/predict", tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Upload a CSV with columns: vibration, temperature, [pressure], [label]
    Returns JSON with RF predictions and LSTM (early-warning) probabilities.
    """
    # 1) Load models (once)
    load_errors = load_models_once()
    if load_errors:
        return JSONResponse(status_code=500, content={"error": " ; ".join(load_errors)})

    # 2) Read uploaded CSV safely
    try:
        raw = await file.read()
        # Works for text/csv and application/octet-stream
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Could not read CSV: {e}"})

    # 3) Validate columns
    for c in REQUIRED_COLS:
        if c not in df.columns:
            return JSONResponse(status_code=422, content={"error": f"Missing column: {c}"})
    if "pressure" not in df.columns:
        df["pressure"] = 0.0  # fill optional sensor

    # 4) RandomForest predictions
    try:
        X = df[["vibration", "temperature", "pressure"]].fillna(0)
        if hasattr(rf_model, "predict_proba"):
            rf_probs = rf_model.predict_proba(X)[:, 1]
        else:
            rf_probs = rf_model.predict(X)
        rf_preds = (rf_probs >= 0.5).astype(int)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"RF prediction failed: {e}"})

    # 5) LSTM early-warning (sequence window = 10)
    try:
        window = 10
        seqs = []
        vals = df[["vibration", "temperature"]].values
        for i in range(len(vals) - window):
            seqs.append(vals[i:i+window])
        if len(seqs) > 0:
            seqs = np.asarray(seqs, dtype="float32")
            lstm_seq_probs = lstm_model.predict(seqs, verbose=0).flatten().tolist()
            lstm_full = [0.0]*len(df)
            for i, p in enumerate(lstm_seq_probs):
                lstm_full[i+window] = float(p)
        else:
            lstm_seq_probs = []
            lstm_full = [0.0]*len(df)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"LSTM prediction failed: {e}"})

    # 6) Simple early-warning flag using threshold 0.5
    early_warning = any(p >= 0.5 for p in lstm_full)
    result = {
        "samples": int(len(df)),
        "rf_predicted_faults": int(rf_preds.sum()),
        "rf_probs_sample": [float(x) for x in rf_probs[:5]],
        "lstm_probs_sample": [float(x) for x in lstm_full[:10]],
        "early_warning": bool(early_warning),
        "message": "FAULT DETECTED" if rf_preds.any() else ("EARLY WARNING" if early_warning else "NORMAL OPERATION")
    }
    return JSONResponse(content=result)
