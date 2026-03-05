import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

print("🔥 Minimal-Difference Irradiance Heatmap Analysis Started")

# --------------------------------------------------
# PATHS
# --------------------------------------------------
DATA_CSV = Path(r"C:\Users\muthumaniraj\OneDrive\Documents\me research\PV_Research_chat_recheck\PV_ALL_DATA_MASTER.csv")
OUT_DIR = Path(r"C:\Users\muthumaniraj\OneDrive\Documents\me research\plots_heatmap_min_diff")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv(DATA_CSV)

# Month order (critical for interpretation)
MONTH_ORDER = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December"
]

df["Month"] = pd.Categorical(df["Month"], categories=MONTH_ORDER, ordered=True)

# --------------------------------------------------
# MONTHLY MEAN IRRADIANCE
# --------------------------------------------------
monthly = (
    df.groupby(["Month", "Shape", "Spacing_cm"], as_index=False)
      ["Irradiation_Wm2"]
      .mean()
)

# --------------------------------------------------
# FUNCTION: HEATMAP FOR ONE SPACING
# --------------------------------------------------
def plot_heatmap(spacing):
    sub = monthly[monthly["Spacing_cm"] == spacing]

    pivot = sub.pivot(
        index="Month",
        columns="Shape",
        values="Irradiation_Wm2"
    ).reindex(MONTH_ORDER)

    # Compute minimal differences
    diff = pd.DataFrame(index=pivot.index)
    diff["Flat"] = pivot["Flat"] - pivot["Inverted-V"]
    diff["V-Shape"] = pivot["V-Shape"] - pivot["Inverted-V"]

    # --------------------------------------------------
    # HEATMAP
    # --------------------------------------------------
    plt.figure(figsize=(6, 6))

    sns.heatmap(
        diff,
        annot=True,
        fmt=".1f",
        cmap="coolwarm",
        center=0,
        vmin=-40, vmax=10,     # 🔑 MINIMAL DIFFERENCE SCALE
        linewidths=0.5,
        cbar_kws={"label": "Δ Irradiance vs Inverted-V (W/m²)"}
    )

    plt.title(
        f"Local Irradiance Difference (Minimal Scale)\nSpacing = {spacing} cm"
    )
    plt.ylabel("Month")
    plt.xlabel("Shape vs Inverted-V")

    plt.tight_layout()
    out = OUT_DIR / f"Heatmap_Irradiance_MinDiff_{spacing}cm.png"
    plt.savefig(out, dpi=600)
    plt.close()

    print(f"✅ Saved heatmap: {out}")

# --------------------------------------------------
# RUN FOR ALL SPACINGS
# --------------------------------------------------
for cm in [62, 77, 93]:
    plot_heatmap(cm)

print("\n🎯 Heatmap Analysis Complete")
print("📁 Output directory:", OUT_DIR)
