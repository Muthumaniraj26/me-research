# ============================================================
# MONTH-WISE DEVIATION (NO MISSING VALUES)
# AUTO BASELINE PER MONTH
# ============================================================

import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
INPUT_DIR = Path(r"C:\Users\muthumaniraj\Documents\me research\results")
OUTPUT_DIR = Path(r"C:\Users\muthumaniraj\Documents\me research\results_full")
OUTPUT_DIR.mkdir(exist_ok=True)

TIME_STEP_HOURS = 15 / 60  # 15-min

# Preferred baseline
BASE_SHAPE = "Flat"
BASE_SPACING = 77
BASE_CONNECTION = "Se-P"

# ------------------------------------------------------------
# LOAD ALL CSV FILES
# ------------------------------------------------------------
records = []

for csv_file in INPUT_DIR.glob("*.csv"):
    try:
        df = pd.read_csv(csv_file)

        # Mandatory column check
        if "Pm_W" not in df.columns:
            continue

        df["Energy_kWh"] = df["Pm_W"] * TIME_STEP_HOURS / 1000

        records.append(
            df.groupby(
                ["Month", "Shape", "Spacing_cm", "Connection"],
                as_index=False
            )["Energy_kWh"].sum()
        )

    except Exception as e:
        print(f"⚠️ Skipped {csv_file.name}: {e}")

data = pd.concat(records, ignore_index=True)

# ------------------------------------------------------------
# BUILD MONTHLY BASELINE (AUTO)
# ------------------------------------------------------------
baseline_rows = []

for month in data["Month"].unique():

    # 1️⃣ Try exact baseline
    exact = data[
        (data["Month"] == month) &
        (data["Shape"] == BASE_SHAPE) &
        (data["Spacing_cm"] == BASE_SPACING) &
        (data["Connection"] == BASE_CONNECTION)
    ]

    if not exact.empty:
        ref_energy = exact["Energy_kWh"].values[0]

    else:
        # 2️⃣ Fallback: mean Flat Se-P for that month
        fallback = data[
            (data["Month"] == month) &
            (data["Shape"] == BASE_SHAPE) &
            (data["Connection"] == BASE_CONNECTION)
        ]

        if fallback.empty:
            raise ValueError(f"No baseline possible for month: {month}")

        ref_energy = fallback["Energy_kWh"].mean()

    baseline_rows.append({
        "Month": month,
        "Reference_Energy_kWh": ref_energy
    })

baseline_df = pd.DataFrame(baseline_rows)

# ------------------------------------------------------------
# MERGE & COMPUTE DEVIATION
# ------------------------------------------------------------
final = data.merge(baseline_df, on="Month", how="left")

final["Deviation_%"] = (
    (final["Energy_kWh"] - final["Reference_Energy_kWh"])
    / final["Reference_Energy_kWh"] * 100
)

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------
out_file = OUTPUT_DIR / "monthly_deviation_all_complete.csv"
final.to_csv(out_file, index=False)

print("✅ SUCCESS: No missing deviation values")
print(f"📁 Saved at: {out_file}")
