# ============================================================
# ALL WINTER TSSC CASES → DEVIATION (%) TABLE
# Based on 15-min CSV energy integration
# ============================================================

import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
# USER INPUTS
# ------------------------------------------------------------
DATA_DIR = Path(r"C:\Users\muthumaniraj\Documents\me research\me research\tssc\deviation winter")   # folder with winter TSSC CSVs

# Reference winter energy (kWh) for comparison
# (From PVGIS / measured SPP / paper baseline)
REFERENCE_WINTER_ENERGY_KWH = 10000.0   # <-- CHANGE to your value

TIME_STEP_HOURS = 0.25  # 15-minute resolution

# ------------------------------------------------------------
# FUNCTION: compute winter energy from CSV
# ------------------------------------------------------------
def compute_winter_energy_kwh(csv_file):
    df = pd.read_csv(csv_file)
    energy_kwh = (df["Pm_W"] * TIME_STEP_HOURS).sum() / 1000
    return energy_kwh

# ------------------------------------------------------------
# PROCESS ALL FILES
# ------------------------------------------------------------
results = []

for csv_file in DATA_DIR.glob("*TSSC_Winter_15min.csv"):

    energy_model = compute_winter_energy_kwh(csv_file)

    deviation_pct = (
        (energy_model - REFERENCE_WINTER_ENERGY_KWH)
        / REFERENCE_WINTER_ENERGY_KWH
    ) * 100

    # Extract metadata from filename
    name = csv_file.stem

    if "Flat" in name:
        shape = "Flat"
    elif "VShape" in name:
        shape = "V-Shape"
    else:
        shape = "Inverted-V"

    spacing = int(name.split("_")[1].replace("cm", ""))

    results.append({
        "Season": "Winter",
        "Shape": shape,
        "Spacing_cm": spacing,
        "Connection": "TSSC",
        "Model_Energy_kWh": round(energy_model, 2),
        "Reference_Energy_kWh": REFERENCE_WINTER_ENERGY_KWH,
        "Deviation_%": round(deviation_pct, 2)
    })

# ------------------------------------------------------------
# CREATE TABLE
# ------------------------------------------------------------
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(
    ["Shape", "Spacing_cm"]
).reset_index(drop=True)

# Save table
out_file = DATA_DIR / "Winter_TSSC_Deviation_Table.csv"
df_results.to_csv(out_file, index=False)

print("✅ Winter TSSC deviation table created")
print(f"📁 Saved as: {out_file}")
print("\n--- TABLE PREVIEW ---")
print(df_results)
