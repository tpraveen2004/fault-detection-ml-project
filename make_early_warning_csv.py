import numpy as np
import pandas as pd
from pathlib import Path

# ----------------- Tunable parameters -----------------
N = 1000             # total timesteps (rows)
FAULT_START = 720    # point where degradation begins (should trigger early warning)
SEED = 42            # reproducibility
# ------------------------------------------------------

rng = np.random.default_rng(SEED)
t = np.arange(N)

# ----- Normal region (0 .. FAULT_START-1) -----
# Small oscillations + tiny noise
vibration = 0.09 + 0.02*np.sin(2*np.pi*t/180) + rng.normal(0, 0.006, N)
temperature = 38.5 + 1.2*np.sin(2*np.pi*t/240) + rng.normal(0, 0.10, N)
pressure = 2.6 + 0.03*np.sin(2*np.pi*t/150) + rng.normal(0, 0.02, N)

# ----- Faulty trend region (FAULT_START .. end) -----
# Model-friendly pattern to push LSTM probability up:
# - vibration jumps in level and gets “sharper” oscillations
# - temperature ramps upward (thermal drift)
# - pressure dips slightly (load/flow change)
drift_len = N - FAULT_START
ramp = np.linspace(0, 1, drift_len)

vibration[FAULT_START:] += 0.10 + 0.06*np.sin(2*np.pi*(t[FAULT_START:])/60)      # level jump + higher freq
temperature[FAULT_START:] += 5.0*ramp + 0.5*np.sin(2*np.pi*(t[FAULT_START:])/120) # steady heating drift
pressure[FAULT_START:] -= 0.25*ramp + 0.04*np.sin(2*np.pi*(t[FAULT_START:])/90)   # small systematic drop

# Defensive bounds (keep values realistic)
vibration = np.clip(vibration, 0.00, None)     # non-negative RMS
temperature = np.clip(temperature, 20, 120)    # plausible plant temps
pressure = np.clip(pressure, 0.5, 5.0)         # plausible process pressure

# Assemble DataFrame with the EXACT column names your API expects
df = pd.DataFrame({
    "vibration": vibration.astype(float),
    "temperature": temperature.astype(float),
    "pressure": pressure.astype(float)  # optional in your API; included for realism
})

# Ensure output folder exists
out_dir = Path("sample_data")
out_dir.mkdir(exist_ok=True)

# Main file that’s likely to trigger EARLY WARNING
out_path = out_dir / "early_warning_trigger.csv"
df.to_csv(out_path, index=False)
print(f"[OK] Wrote: {out_path.resolve()}  (rows={len(df)})")

# (Optional) also write a 'normal_only' file for demo comparison
df_normal = df.copy()
df_normal.loc[FAULT_START:, ["vibration","temperature","pressure"]] = df.loc[FAULT_START-1, ["vibration","temperature","pressure"]].values
out_path2 = out_dir / "normal_reference.csv"
df_normal.to_csv(out_path2, index=False)
print(f"[OK] Wrote: {out_path2.resolve()}  (rows={len(df_normal)})")

print("\nTips:")
print(" - Upload 'early_warning_trigger.csv' first → should show 'EARLY WARNING' (if RF doesn’t already mark faults).")
print(" - Upload 'normal_reference.csv' → should show 'NORMAL OPERATION'.")
print(" - If your LSTM was trained on only 2 features, the backend will auto-select the first two (vibration, temperature).")
