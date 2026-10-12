# make_sample_csvs.py
# Creates ready-to-use CSVs for your API/LSTM/RF demo.

import os
import numpy as np
import pandas as pd

BASE = r"B:\projects\early-warning-fault-detection"
OUT_DIR = os.path.join(BASE, "sample_data")
os.makedirs(OUT_DIR, exist_ok=True)

rng = np.random.default_rng(42)

def gen_segment(n, vib_base, temp_base, press_base, noise=0.02, label=0):
    t = np.arange(n)
    vibration = vib_base + 0.03*np.sin(2*np.pi*t/120) + rng.normal(0, noise, n)
    temperature = temp_base + 0.8*np.sin(2*np.pi*t/200) + rng.normal(0, 0.6, n)
    pressure = press_base + 0.6*np.sin(2*np.pi*t/180) + rng.normal(0, 0.5, n)
    df = pd.DataFrame({
        "vibration": vibration.round(3),
        "temperature": temperature.round(2),
        "pressure": pressure.round(2),
        "label": np.full(n, label, dtype=int)
    })
    return df

# 1) Normal-only data
normal_df = gen_segment(600, vib_base=0.10, temp_base=40.0, press_base=101.0, noise=0.015, label=0)
normal_path = os.path.join(OUT_DIR, "normal_data.csv")
normal_df.to_csv(normal_path, index=False)

# 2) Fault-only data (higher vibration/temperature drift)
fault_df = gen_segment(400, vib_base=0.35, temp_base=55.0, press_base=104.0, noise=0.03, label=1)
fault_path = os.path.join(OUT_DIR, "fault_data.csv")
fault_df.to_csv(fault_path, index=False)

# 3) Mixed data: 700 normal → 300 fault transition
mixed_normal = gen_segment(700, vib_base=0.11, temp_base=41.0, press_base=101.5, noise=0.015, label=0)
mixed_fault = gen_segment(300, vib_base=0.32, temp_base=54.0, press_base=103.5, noise=0.03, label=1)

mixed_df = pd.concat([mixed_normal, mixed_fault], ignore_index=True)

# Add a subtle warm-up drift before fault to make early warning believable
drift = np.linspace(0, 0.08, len(mixed_df))
mixed_df["vibration"] = (mixed_df["vibration"] + drift*0.2).round(3)
mixed_df["temperature"] = (mixed_df["temperature"] + drift*0.6).round(2)

mixed_path = os.path.join(OUT_DIR, "mixed_sensor_data.csv")
mixed_df.to_csv(mixed_path, index=False)

print("Created CSVs:")
print(" -", normal_path)
print(" -", fault_path)
print(" -", mixed_path)
