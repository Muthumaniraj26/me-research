import pandas as pd
from pathlib import Path

print("📉 Deviation analysis vs PVGIS started")

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = Path("PV_Research_split_equal")
MASTER_CSV = BASE_DIR / r"C:\Users\muthumaniraj\Documents\me research\PV_Research_chat_recheck\PV_ALL_DATA_MASTER.csv"
PVGIS_CSV  = Path(r"C:\Users\muthumaniraj\Documents\me research\pvgis_monthly_energy.csv")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv(MASTER_CSV)
pvgis = pd.read_csv(PVGIS_CSV)

# --------------------------------------------------
# ENERGY CALCULATION (15-min → kWh)
# --------------------------------------------------
TIME_STEP_HOURS = 15 / 60
df["Energy_kWh"] = df["Pm_W"] * TIME_STEP_HOURS / 1000

# --------------------------------------------------
# MONTHLY ENERGY (OUR MODEL)
# --------------------------------------------------
monthly_energy = (
    df.groupby(["Month", "Shape", "Spacing_cm"], as_index=False)
      .agg({"Energy_kWh": "sum"})
)

# --------------------------------------------------
# MERGE WITH PVGIS
# --------------------------------------------------
comparison = monthly_energy.merge(
    pvgis,
    on="Month",
    how="left"
)

# --------------------------------------------------
# DEVIATION CALCULATION
# --------------------------------------------------
comparison["Deviation_%"] = (
    (comparison["Energy_kWh"] - comparison["PVGIS_Energy_kWh"])
    / comparison["PVGIS_Energy_kWh"] * 100
)

# --------------------------------------------------
# SORT FOR READABILITY
# --------------------------------------------------
comparison = comparison.sort_values(
    by=["Month", "Shape", "Spacing_cm"]
)

# --------------------------------------------------
# DISPLAY IN TERMINAL (TABLE FORMAT)
# --------------------------------------------------
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", "{:.2f}".format)

print("\n📊 DEVIATION TABLE (OUR MODEL vs PVGIS)\n")
print(
    comparison[
        ["Month", "Shape", "Spacing_cm",
         "Energy_kWh", "PVGIS_Energy_kWh", "Deviation_%"]
    ].to_string(index=False)
)

print("\n✅ Deviation comparison completed")
