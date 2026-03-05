import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Using Seaborn for better styling
from pathlib import Path

print("📊 High-Visibility Irradiance Comparison Started")

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = Path(r"C:\Users\muthumaniraj\Documents\me research\PV_Research_chat_recheck")
CSV_FILE = BASE_DIR / "PV_ALL_DATA_MASTER.csv"
PLOT_DIR = BASE_DIR / "plots_high_visibility"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# LOAD & PREP DATA
# --------------------------------------------------
df = pd.read_csv(CSV_FILE)
df = df[df["Irradiation_Wm2"] > 100]

# Use larger bins (100 instead of 50) to reduce clutter
df["Irr_bin"] = (df["Irradiation_Wm2"] // 100) * 100

summary = (
    df.groupby(["Season", "Shape", "Irr_bin"], as_index=False)
      .agg({"Irradiation_Wm2": "mean"})
)

# --------------------------------------------------
# PLOT (GROUPED BAR STYLE)
# --------------------------------------------------
# Set a clean aesthetic
sns.set_theme(style="whitegrid")

for season in summary["Season"].unique():
    season_df = summary[summary["Season"] == season]
    
    # Increase figure size for readability
    plt.figure(figsize=(14, 8))
    
    # Use Seaborn's barplot for automatic grouping/dodging
    ax = sns.barplot(
        data=season_df,
        x="Irr_bin",
        y="Irradiation_Wm2",
        hue="Shape",
        palette="viridis", # High contrast color palette
        edgecolor="black"
    )

    # 🔍 ZOOM THE Y-AXIS to show micro-differences
    # We focus on the top 20% of the data where the differences happen
    data_min = season_df["Irradiation_Wm2"].min()
    data_max = season_df["Irradiation_Wm2"].max()
    plt.ylim(data_min - 10, data_max + 30) 

    # Add value labels on top of bars for precision
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3, rotation=90, fontsize=8)

    plt.title(f"Irradiance Variance by Shape: {season} (Zoomed View)", fontsize=16, fontweight='bold')
    plt.xlabel("Irradiance Bin (W/m²)", fontsize=12)
    plt.ylabel("Measured Mean Irradiance (W/m²)", fontsize=12)
    plt.legend(title="Panel Shape", loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()

    out = PLOT_DIR / f"high_vis_bar_{season}.png"
    plt.savefig(out, dpi=300)
    plt.close()

    print(f"✅ Saved: {out}")

print("\n🎯 Visualizations generated. Check the 'plots_high_visibility' folder.")