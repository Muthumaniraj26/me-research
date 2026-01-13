# ============================================================
# PLOTTING: BEST SHAPE & SPACING PER SEASON
# Reads CSV results and generates paper-ready plots
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

print("📊 PLOTTING STARTED")

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
BASE_DIR = Path(r"C:\Users\muthumaniraj\Documents\me research\PV_Research")
SUMMARY_CSV = BASE_DIR / "seasonal_energy_summary.csv"
BEST_CSV = BASE_DIR / "best_config_per_season.csv"

PLOT_DIR = BASE_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
summary = pd.read_csv(SUMMARY_CSV)
best_per_season = pd.read_csv(BEST_CSV)

print("✅ CSV files loaded successfully")

# ------------------------------------------------------------
# 1️⃣ SEASON-WISE ENERGY COMPARISON (LINE PLOT)
# ------------------------------------------------------------
plt.figure(figsize=(10, 5))

for season in summary["Season"].unique():
    data = summary[summary["Season"] == season]
    plt.plot(
        range(len(data)),
        data["Energy_kWh"],
        marker="o",
        linewidth=2,
        label=season
    )

plt.title("Season-wise Energy Comparison (All Shapes & Spacings)")
plt.xlabel("Configuration Index")
plt.ylabel("Energy (kWh)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "Seasonwise_Energy_Comparison.png", dpi=600)
plt.close()

# ------------------------------------------------------------
# 2️⃣ SHAPE × SPACING COMPARISON (BAR PLOT)
# ------------------------------------------------------------
pivot_shape = summary.pivot_table(
    index=["Shape", "Spacing_cm"],
    columns="Season",
    values="Energy_kWh"
)

pivot_shape.plot(kind="bar", figsize=(12, 6))
plt.title("Energy Comparison by Shape and Spacing")
plt.ylabel("Energy (kWh)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "Shape_Spacing_Comparison.png", dpi=600)
plt.close()

# ------------------------------------------------------------
# 3️⃣ BEST CONFIGURATION PER SEASON
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))

bars = plt.bar(
    best_per_season["Season"],
    best_per_season["Energy_kWh"],
    color="tab:green"
)

for bar, (_, row) in zip(bars, best_per_season.iterrows()):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f'{row["Shape"]}\n{row["Spacing_cm"]} cm',
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.title("Best Performing Configuration per Season")
plt.ylabel("Energy (kWh)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "Best_Config_Per_Season.png", dpi=600)
plt.close()

# ------------------------------------------------------------
# DONE
# ------------------------------------------------------------
print("\n✅ ALL PLOTS GENERATED SUCCESSFULLY")
print(f"📁 Plots saved at: {PLOT_DIR.resolve()}")
