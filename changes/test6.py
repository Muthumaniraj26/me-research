import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("🚀 WEEKLY IRRADIATION – ALL SHAPES & SPACINGS")

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
BASE_DIR = Path("PV_Research_chat_recheck")
CSV_FILE = BASE_DIR / "PV_ALL_DATA_MASTER.csv"
OUT_DIR = BASE_DIR / "weekly_visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
df = pd.read_csv(CSV_FILE)

df["Date"] = pd.to_datetime(df["Date"])
df["Week"] = df["Date"].dt.isocalendar().week

# Convert W/m² → kWh/m² (15-min resolution)
df["Irr_kWh_m2"] = df["Irradiation_Wm2"] * (15 / 60) / 1000

# ------------------------------------------------------------
# WEEKLY AGGREGATION
# ------------------------------------------------------------
weekly = df.groupby(
    ["Week", "Shape", "Spacing_cm"],
    as_index=False
)["Irr_kWh_m2"].sum()

weekly["Label"] = weekly["Shape"] + " | " + weekly["Spacing_cm"].astype(str) + " cm"

# ------------------------------------------------------------
# 1️⃣ LINE PLOT — ALL SHAPES & SPACINGS
# ------------------------------------------------------------
plt.figure(figsize=(18, 7))

style_map = {62: ":", 77: "--", 93: "-"}
color_map = {
    "Flat": "#1f77b4",
    "V-Shape": "#ff7f0e",
    "Inverted-V": "#2ca02c"
}

for (shape, spacing), grp in weekly.groupby(["Shape", "Spacing_cm"]):
    plt.plot(
        grp["Week"],
        grp["Irr_kWh_m2"],
        linestyle=style_map[spacing],
        linewidth=2.0,
        color=color_map[shape],
        label=f"{shape} | {spacing} cm"
    )

plt.title("Weekly Irradiation Comparison – All Shapes & Spacings", fontsize=14)
plt.xlabel("ISO Week (1–52)")
plt.ylabel("Weekly Irradiation (kWh/m²)")
plt.grid(alpha=0.35)
plt.legend(ncol=3, fontsize=9)
plt.tight_layout()

line_file = OUT_DIR / "weekly_irradiation_all_shapes.png"
plt.savefig(line_file, dpi=300)
plt.show()

# ------------------------------------------------------------
# 2️⃣ HEATMAP — WEEK × SHAPE–SPACING
# ------------------------------------------------------------
heatmap_df = weekly.pivot_table(
    index="Label",
    columns="Week",
    values="Irr_kWh_m2"
)

plt.figure(figsize=(20, 6))
sns.heatmap(
    heatmap_df,
    cmap="viridis",
    linewidths=0.25,
    linecolor="white",
    cbar_kws={"label": "Weekly Irradiation (kWh/m²)"}
)

plt.title("Weekly Irradiation Heatmap (All Shapes & Spacings)", fontsize=14)
plt.xlabel("ISO Week")
plt.ylabel("Shape | Spacing")
plt.tight_layout()

heatmap_file = OUT_DIR / "weekly_irradiation_heatmap.png"
plt.savefig(heatmap_file, dpi=300)
plt.show()

print("✅ ALL VISUALIZATIONS CREATED")
print(f"📁 Saved in: {OUT_DIR}")
