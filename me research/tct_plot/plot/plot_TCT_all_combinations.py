# ============================================================
# TCT ALL-COMBINATION VISUALIZATION
# Shapes × Spacings | Summer | 15-min data
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------
# LOAD ALL TCT CSV FILES
# ------------------------------------------------------------
data_dir = Path(r"me research\tct\results_tct")
csv_files = list(data_dir.glob("*_TCT_Summer_15min.csv"))

df_list = []
for f in csv_files:
    df = pd.read_csv(f)
    df_list.append(df)

df = pd.concat(df_list, ignore_index=True)

# ------------------------------------------------------------
# CLEAN TIME (USE ONLY MAJOR TIMES)
# ------------------------------------------------------------
df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"])

# Select only major times → clear plots
major_times = ["08:00", "10:00", "12:00", "14:00", "16:00", "18:00"]
df = df[df["Time"].isin(major_times)]

# ------------------------------------------------------------
# OUTPUT DIRECTORY
# ------------------------------------------------------------
out_dir = Path("plots_TCT")
out_dir.mkdir(exist_ok=True)

# ------------------------------------------------------------
# STYLES
# ------------------------------------------------------------
shape_colors = {
    "Flat": "tab:blue",
    "V-Shape": "tab:green",
    "Inverted-V": "tab:red"
}

spacing_styles = {
    62: "--",
    77: "-.",
    93: "-"
}

# ============================================================
# 1️⃣ COMMON SPACING → ALL SHAPES (LINE PLOTS)
# ============================================================
for cm in [62, 77, 93]:
    plt.figure(figsize=(9, 5))
    subset = df[df["Spacing_cm"] == cm]

    for shape in ["Flat", "V-Shape", "Inverted-V"]:
        d = subset[subset["Shape"] == shape]
        plt.plot(
            d["DateTime"],
            d["Pm_W"],
            label=shape,
            color=shape_colors[shape],
            linewidth=2,
            marker="o"
        )

    plt.title(f"TCT Power Comparison — All Shapes @ {cm} cm")
    plt.xlabel("Time")
    plt.ylabel("Power (W)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"TCT_Line_Shapes_{cm}cm.png", dpi=600)
    plt.close()

# ============================================================
# 2️⃣ COMMON SHAPE → ALL SPACINGS (LINE PLOTS)
# ============================================================
for shape in ["Flat", "V-Shape", "Inverted-V"]:
    plt.figure(figsize=(9, 5))
    subset = df[df["Shape"] == shape]

    for cm in [62, 77, 93]:
        d = subset[subset["Spacing_cm"] == cm]
        plt.plot(
            d["DateTime"],
            d["Pm_W"],
            label=f"{cm} cm",
            linestyle=spacing_styles[cm],
            linewidth=2,
            marker="s"
        )

    plt.title(f"TCT Power Comparison — {shape} (All Spacings)")
    plt.xlabel("Time")
    plt.ylabel("Power (W)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"TCT_Line_Spacings_{shape}.png", dpi=600)
    plt.close()

# ============================================================
# 3️⃣ ELECTRICAL PARAMETER BAR COMPARISON
# ============================================================
params = ["Voc_V", "Isc_A", "Vm_V", "Im_A", "Pm_W"]

avg_df = (
    df.groupby(["Shape", "Spacing_cm"])[params]
    .mean()
    .reset_index()
)

for param in params:
    plt.figure(figsize=(8, 4))
    for shape in ["Flat", "V-Shape", "Inverted-V"]:
        d = avg_df[avg_df["Shape"] == shape]
        plt.plot(
            d["Spacing_cm"],
            d[param],
            marker="o",
            linewidth=2,
            label=shape
        )

    plt.title(f"TCT Average {param} Comparison")
    plt.xlabel("Spacing (cm)")
    plt.ylabel(param)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"TCT_{param}_Comparison.png", dpi=600)
    plt.close()

print("✅ ALL TCT COMBINATION PLOTS GENERATED")
print(f"📁 Saved in folder: {out_dir}")
