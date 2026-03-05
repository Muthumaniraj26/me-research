import pandas as pd

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
CSV_PATH = r"C:\Users\muthumaniraj\Documents\me research\PV_Research_chat_recheck\PV_ALL_DATA_MASTER.csv"
df = pd.read_csv(CSV_PATH)

print("✅ Data loaded successfully")

# ------------------------------------------------------------
# ENERGY CALCULATION
# ------------------------------------------------------------
# 15-minute timestep = 0.25 hours
df["Energy_kWh"] = df["Pm_W"] * 0.25 / 1000

# ------------------------------------------------------------
# GROUP BY SEASON, SHAPE, SPACING
# ------------------------------------------------------------
summary = (
    df.groupby(
        ["Season", "Shape", "Spacing_cm"],
        as_index=False
    )["Energy_kWh"]
    .sum()
)

print("\n--- ENERGY SUMMARY (kWh) ---")
print(summary.head())

# ------------------------------------------------------------
# FIND BEST CONFIGURATION PER SEASON
# ------------------------------------------------------------
best_per_season = (
    summary.sort_values("Energy_kWh", ascending=False)
    .groupby("Season", as_index=False)
    .first()
)

print("\n🏆 BEST CONFIGURATION PER SEASON")
print(best_per_season)

# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------
summary.to_csv("PV_Research/seasonal_energy_summary.csv", index=False)
best_per_season.to_csv("PV_Research/best_config_per_season.csv", index=False)

print("\n📁 Files saved:")
print(" - seasonal_energy_summary.csv")
print(" - best_config_per_season.csv")
