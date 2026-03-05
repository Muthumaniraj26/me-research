import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

print("🚀 WEEKLY OUTPERFORMANCE ANALYSIS (vs Inverted-V)")

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
BASE_DIR = Path("PV_Research_chat_recheck")
CSV_FILE = BASE_DIR / "PV_ALL_DATA_MASTER.csv"
OUT_DIR = BASE_DIR / "weekly_outperformance_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
df = pd.read_csv(CSV_FILE)

df["Date"] = pd.to_datetime(df["Date"])
df["Week"] = df["Date"].dt.isocalendar().week

# Convert W/m² → kWh/m² (15-min)
df["Irr_kWh_m2"] = df["Irradiation_Wm2"] * (15 / 60) / 1000

# ------------------------------------------------------------
# WEEKLY SUM
# ------------------------------------------------------------
weekly = df.groupby(
    ["Week", "Shape", "Spacing_cm"],
    as_index=False
)["Irr_kWh_m2"].sum()

# ------------------------------------------------------------
# PLOT SETTINGS
# ------------------------------------------------------------
shape_compare = ["Flat", "V-Shape"]
spacing_styles = {62: ":", 77: "--", 93: "-"}
shape_colors = {"Flat": "#1f77b4", "V-Shape": "#ff7f0e"}

# ------------------------------------------------------------
# DIFFERENCE PLOT
# ------------------------------------------------------------
plt.figure(figsize=(16, 6))

for spacing in [62, 77, 93]:

    inv = weekly[
        (weekly["Shape"] == "Inverted-V") &
        (weekly["Spacing_cm"] == spacing)
    ].set_index("Week")

    for shape in shape_compare:
        cur = weekly[
            (weekly["Shape"] == shape) &
            (weekly["Spacing_cm"] == spacing)
        ].set_index("Week")

        diff = cur["Irr_kWh_m2"] - inv["Irr_kWh_m2"]

        plt.plot(
            diff.index,
            diff.values,
            linestyle=spacing_styles[spacing],
            linewidth=2.2,
            color=shape_colors[shape],
            label=f"{shape} − Inverted-V | {spacing} cm"
        )

# ------------------------------------------------------------
# ZERO LINE (DOMINANCE LINE)
# ------------------------------------------------------------
plt.axhline(0, color="black", linewidth=1, linestyle="--")

# ------------------------------------------------------------
# FORMATTING (EXPANDED SCALE)
# ------------------------------------------------------------
plt.title(
    "Weekly Irradiation Outperformance vs Inverted-V\n(Positive = Flat / V-Shape Wins)",
    fontsize=14
)
plt.xlabel("ISO Week (1–52)")
plt.ylabel("Δ Weekly Irradiation (kWh/m²)")
plt.grid(alpha=0.35)

plt.legend(ncol=3, fontsize=9)
plt.tight_layout()

out_file = OUT_DIR / "weekly_outperformance_vs_invertedV.png"
plt.savefig(out_file, dpi=300)
plt.show()

print(f"📊 Plot saved at: {out_file}")
print("✅ Local outperform regions are now clearly visible")
