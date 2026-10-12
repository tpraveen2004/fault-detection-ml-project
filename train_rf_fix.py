# train_rf_fix.py
import os, joblib, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE = r"B:\projects\early-warning-fault-detection"
DATA = os.path.join(BASE, "sample_data", "mixed_sensor_data.csv")
OUT  = os.path.join(BASE, "outputs", "rf_fault_model.joblib")

df = pd.read_csv(DATA)
# Ensure all three features exist; create pressure if missing
for c in ["vibration","temperature","pressure"]:
    if c not in df.columns:
        df[c] = 0.0

X = df[["vibration","temperature","pressure"]]
y = df["label"] if "label" in df.columns else (df["temperature"]>50).astype(int)  # fallback

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(Xtr, ytr)
print("Accuracy:", accuracy_score(yte, rf.predict(Xte)))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
joblib.dump(rf, OUT)
print("Saved:", OUT)
