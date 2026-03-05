# ============================================================
# FULL SHAPE + SPACING COMPARISON WITH CROSSOVER DETECTION
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
CSV_PATH = r"PV_Research_split_equal/PV_ALL_DATA_MASTER.csv"
df = pd.read_csv(CSV_PATH)

df["Date"] = pd.to_datetime(df["Date"])
df["Energy_kWh"] = df["Pm_W"] * (15/60) / 1000

MONTH_ORDER = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]
df["Month"] = pd.Categorical(df["Month"], categories=MONTH_ORDER, ordered=True)

# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------
SPACINGS = [62, 77, 93]
SHAPES = ["Flat", "V-Shape", "Inverted-V"]
COLORS = {"Flat":"tab:blue", "V-Shape":"tab:green", "Inverted-V":"tab:red"}

BASE_PLOT_DIR = Path("plots_update_spacing")
BASE_PLOT_DIR.mkdir(exist_ok=True)

# ============================================================
# MAIN LOOP — PER SPACING
# ============================================================
for spacing in SPACINGS:

    print(f"\n📐 ANALYZING SPACING: {spacing} cm")

    out_dir = BASE_PLOT_DIR / f"spacing_{spacing}"
    out_dir.mkdir(exist_ok=True)

    # --------------------------------------------------------
    # MONTHLY ENERGY
    # --------------------------------------------------------
    monthly = (
        df[df["Spacing_cm"] == spacing]
        .groupby(["Month","Shape"], as_index=False)["Energy_kWh"]
        .sum()
    )

    pivot = monthly.pivot(index="Month", columns="Shape", values="Energy_kWh")

    # --------------------------------------------------------
    # CROSSOVER DETECTION
    # --------------------------------------------------------
    crossover_records = []

    for month in pivot.index:
        f = pivot.loc[month, "Flat"]
        v = pivot.loc[month, "V-Shape"]
        iv = pivot.loc[month, "Inverted-V"]

        if f > iv:
            crossover_records.append([month, spacing, "Flat > Inverted-V"])
        if v > iv:
            crossover_records.append([month, spacing, "V-Shape > Inverted-V"])
        if f > v:
            crossover_records.append([month, spacing, "Flat > V-Shape"])

    crossover_df = pd.DataFrame(
        crossover_records,
        columns=["Month", "Spacing_cm", "Observation"]
    )

    crossover_df.to_csv(out_dir / "crossover_table.csv", index=False)

    # --------------------------------------------------------
    # PLOT — MONTHWISE ENERGY
    # --------------------------------------------------------
    plt.figure(figsize=(11,5))

    for shape in SHAPES:
        plt.plot(
            pivot.index,
            pivot[shape],
            marker="o",
            linewidth=2.5,
            label=shape,
            color=COLORS[shape]
        )

    plt.title(f"Month-wise Energy Comparison @ {spacing} cm")
    plt.xlabel("Month")
    plt.ylabel("Energy (kWh)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "monthwise_energy.png", dpi=600)
    plt.close()

    print(f"✅ Plots & tables saved for {spacing} cm")

print("\n🏁 ALL SPACING ANALYSIS COMPLETED")
