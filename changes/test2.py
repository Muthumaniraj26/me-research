import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

print("🚀 WEEKLY IRRADIATION — ALL SHAPES IN ONE PLOT")

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
BASE_DIR = Path("PV_Research_chat_recheck")
CSV_FILE = BASE_DIR / "PV_ALL_DATA_MASTER.csv"
OUT_DIR = BASE_DIR / "weekly_all_shapes_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
df = pd.read_csv(CSV_FILE)

df["Date"] = pd.to_datetime(df["Date"])
df["Week"] = df["Date"].dt.isocalendar().week

# 15-min → kWh/m²
df["Irr_kWh_m2"] = df["Irradiation_Wm2"] * (15 / 60) / 1000

# ------------------------------------------------------------
# WEEKLY AGGREGATION
# ------------------------------------------------------------
weekly = df.groupby(
    ["Week", "Shape", "Spacing_cm"],
    as_index=False
)["Irr_kWh_m2"].sum()

# ------------------------------------------------------------
# PLOT: ALL SHAPES TOGETHER (PER SPACING)
# ------------------------------------------------------------
for spacing in sorted(weekly["Spacing_cm"].unique()):

    plt.figure(figsize=(14, 5))

    for shape, style in zip(
        ["Flat", "V-Shape", "Inverted-V"],
        ["--", "-.", "-"]
    ):
        sub = weekly[
            (weekly["Spacing_cm"] == spacing) &
            (weekly["Shape"] == shape)
        ]

        plt.plot(
            sub["Week"],
            sub["Irr_kWh_m2"],
            linestyle=style,
            linewidth=2.5,
            marker="o",
            label=shape
        )

    plt.title(
        f"Weekly Irradiation Comparison (All Shapes) — {spacing} cm spacing",
        fontsize=13
    )
    plt.xlabel("ISO Week")
    plt.ylabel("Weekly Irradiation (kWh/m²)")
    plt.grid(alpha=0.35)
    plt.legend(title="Panel Shape")
    plt.tight_layout()

    file = OUT_DIR / f"weekly_all_shapes_{spacing}cm.png"
    plt.savefig(file, dpi=300)
    plt.close()

    print(f"📊 Plot saved → {file}")

print("\n✅ DONE — You can now SEE local outperformance clearly")
