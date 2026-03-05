import pandas as pd
from pathlib import Path

print("🚀 WEEK-WISE PERFORMANCE ANALYSIS STARTED")

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
BASE_DIR = Path("PV_Research_chat_recheck")
CSV_FILE = BASE_DIR / "PV_ALL_DATA_MASTER.csv"
OUT_DIR = BASE_DIR / "weekly_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
df = pd.read_csv(CSV_FILE)

df["Date"] = pd.to_datetime(df["Date"])
df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

# 15-min → kWh/m²
df["Irr_kWh_m2"] = df["Irradiation_Wm2"] * (15 / 60) / 1000

# ------------------------------------------------------------
# WEEKLY ENERGY AGGREGATION
# ------------------------------------------------------------
weekly = (
    df.groupby(["Week", "Shape", "Spacing_cm"], as_index=False)
      .agg({"Irr_kWh_m2": "sum"})
)

# ------------------------------------------------------------
# FIND WEEKLY WINNERS (PER SPACING)
# ------------------------------------------------------------
winners = []

for (week, spacing), grp in weekly.groupby(["Week", "Spacing_cm"]):
    best = grp.loc[grp["Irr_kWh_m2"].idxmax()]
    winners.append({
        "Week": week,
        "Spacing_cm": spacing,
        "Best_Shape": best["Shape"],
        "Best_Irr_kWh_m2": best["Irr_kWh_m2"]
    })

winners_df = pd.DataFrame(winners)

# ------------------------------------------------------------
# DOMINANCE COUNT (HOW MANY WEEKS EACH SHAPE WINS)
# ------------------------------------------------------------
dominance = (
    winners_df
    .groupby(["Spacing_cm", "Best_Shape"])
    .size()
    .reset_index(name="Weeks_Won")
)

# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------
weekly.to_csv(OUT_DIR / "weekly_irradiation_all.csv", index=False)
winners_df.to_csv(OUT_DIR / "weekly_winners.csv", index=False)
dominance.to_csv(OUT_DIR / "weekly_dominance_summary.csv", index=False)

# ------------------------------------------------------------
# TERMINAL OUTPUT (IMPORTANT)
# ------------------------------------------------------------
print("\n🏆 WEEKLY DOMINANCE SUMMARY")
print(dominance.sort_values(["Spacing_cm", "Weeks_Won"], ascending=[True, False]))

print("\n📁 FILES GENERATED:")
print(" - weekly_irradiation_all.csv")
print(" - weekly_winners.csv")
print(" - weekly_dominance_summary.csv")

print("\n✅ WEEK-WISE ANALYSIS COMPLETE")
