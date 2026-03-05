import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

print("📊 Local Irradiance Advantage Analysis Started")

# --------------------------------------------------
# INPUT / OUTPUT PATHS
# --------------------------------------------------
DATA_CSV = Path(r"C:\Users\muthumaniraj\OneDrive\Documents\me research\PV_Research_chat_recheck\PV_ALL_DATA_MASTER.csv")
OUT_DIR = Path("plots_irradiance_local")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv(DATA_CSV)

# Ensure correct month order
MONTH_ORDER = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December"
]

df["Month"] = pd.Categorical(df["Month"], categories=MONTH_ORDER, ordered=True)

# --------------------------------------------------
# MONTHLY MEAN IRRADIANCE (shape-wise, spacing-wise)
# --------------------------------------------------
monthly_irr = (
    df.groupby(["Month", "Shape", "Spacing_cm"], as_index=False)
      ["Irradiation_Wm2"]
      .mean()
)

# --------------------------------------------------
# FUNCTION TO PLOT LOCAL ADVANTAGE
# --------------------------------------------------
def plot_local_advantage(spacing):
    subset = monthly_irr[monthly_irr["Spacing_cm"] == spacing]

    pivot = subset.pivot(
        index="Month",
        columns="Shape",
        values="Irradiation_Wm2"
    ).reindex(MONTH_ORDER)

    # Δ relative to Inverted-V
    pivot["Flat_minus_IV"] = pivot["Flat"] - pivot["Inverted-V"]
    pivot["V_minus_IV"]    = pivot["V-Shape"] - pivot["Inverted-V"]

    plt.figure(figsize=(11, 5))

    plt.plot(
        pivot.index, pivot["Flat_minus_IV"],
        marker="o", linewidth=2,
        label="Flat − Inverted-V"
    )

    plt.plot(
        pivot.index, pivot["V_minus_IV"],
        marker="s", linewidth=2,
        label="V-Shape − Inverted-V"
    )

    # Reference line
    plt.axhline(0, color="black", linestyle="--", linewidth=1)

    # Highlight zones where shapes are close to Inverted-V
    plt.fill_between(
        pivot.index,
        pivot["V_minus_IV"],
        0,
        where=(pivot["V_minus_IV"] > -12),
        alpha=0.15,
        label="V-Shape close-performance zone"
    )

    # Visual clarity
    plt.ylim(-45, -5)
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)

    plt.title(
        f"Local Irradiance Difference vs Inverted-V\nSpacing = {spacing} cm"
    )
    plt.ylabel("Δ Irradiance (W/m²)")
    plt.xlabel("Month")
    plt.legend()

    plt.tight_layout()
    out_file = OUT_DIR / f"Local_Irradiance_Advantage_{spacing}cm.png"
    plt.savefig(out_file, dpi=600)
    plt.close()

    print(f"✅ Saved plot: {out_file}")

# --------------------------------------------------
# RUN FOR ALL SPACINGS
# --------------------------------------------------
for cm in [62, 77, 93]:
    plot_local_advantage(cm)

print("\n🎯 Analysis Complete")
print("📁 All plots saved in:", OUT_DIR)
