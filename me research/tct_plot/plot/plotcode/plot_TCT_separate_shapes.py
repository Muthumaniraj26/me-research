# ============================================================
# OVERALL COMBINATION VISUALIZATION
# Shapes × Spacings × ALL PARAMETERS
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import NullLocator
from pathlib import Path

# ------------------------------------------------------------
# LOAD DATA (ALL CSVs FROM FOLDER)
# ------------------------------------------------------------
data_dir = Path(r"C:\Users\muthumaniraj\Documents\me research\me research\tct\results_tct")
csv_files = list(data_dir.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("❌ No CSV files found in results_tct")

df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

# ------------------------------------------------------------
# PREPROCESS
# ------------------------------------------------------------
df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"])

# Mean diurnal values
df_mean = df.groupby(
    ["Time", "Shape", "Spacing_cm"],
    as_index=False
).mean(numeric_only=True)

# ------------------------------------------------------------
# PARAMETERS TO PLOT
# ------------------------------------------------------------
parameters = {
    "CellTemp_C": "Cell Temperature (°C)",
    "Voc_V": "Open Circuit Voltage (V)",
    "Isc_A": "Short Circuit Current (A)",
    "Vm_V": "Voltage at Max Power (V)",
    "Im_A": "Current at Max Power (A)",
    "Pm_W": "Power Output (W)"
}

shapes = ["Flat", "V-Shape", "Inverted-V"]
spacings = [62, 77, 93]

shape_colors = {
    "Flat": "tab:blue",
    "V-Shape": "tab:green",
    "Inverted-V": "tab:orange"
}

spacing_styles = {62: "--", 77: ":", 93: "-"}

# ------------------------------------------------------------
# OUTPUT DIRECTORY
# ------------------------------------------------------------
out_root = Path(r"C:\Users\muthumaniraj\Documents\me research\me research\overall_plots")
out_root.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOOP OVER EACH PARAMETER
# ============================================================
for param, ylabel in parameters.items():

    param_dir = out_root / param
    param_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # 1️⃣ OVERALL: ALL SHAPES × ALL SPACINGS
    # ========================================================
    plt.figure(figsize=(12,5))

    for shape in shapes:
        for cm in spacings:
            subset = df_mean[
                (df_mean["Shape"] == shape) &
                (df_mean["Spacing_cm"] == cm)
            ]

            plt.plot(
                pd.to_datetime(subset["Time"], format="%H:%M"),
                subset[param],
                linewidth=1.8,
                linestyle=spacing_styles[cm],
                label=f"{shape} | {cm} cm"
            )

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_minor_locator(NullLocator())

    plt.title(f"Overall Comparison — {param}")
    plt.xlabel("Time of Day")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(param_dir / f"Overall_{param}.png", dpi=600)
    plt.close()

    # ========================================================
    # 2️⃣ COMMON SPACING → ALL SHAPES
    # ========================================================
    for cm in spacings:
        plt.figure(figsize=(10,5))

        for shape in shapes:
            subset = df_mean[
                (df_mean["Shape"] == shape) &
                (df_mean["Spacing_cm"] == cm)
            ]

            plt.plot(
                pd.to_datetime(subset["Time"], format="%H:%M"),
                subset[param],
                linewidth=2.2,
                color=shape_colors[shape],
                label=shape
            )

        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_minor_locator(NullLocator())

        plt.title(f"{ylabel} — Spacing {cm} cm")
        plt.xlabel("Time of Day")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(param_dir / f"{param}_Shapes_{cm}cm.png", dpi=600)
        plt.close()

    # ========================================================
    # 3️⃣ COMMON SHAPE → ALL SPACINGS
    # ========================================================
    for shape in shapes:
        plt.figure(figsize=(10,5))

        for cm in spacings:
            subset = df_mean[
                (df_mean["Shape"] == shape) &
                (df_mean["Spacing_cm"] == cm)
            ]

            plt.plot(
                pd.to_datetime(subset["Time"], format="%H:%M"),
                subset[param],
                linewidth=2.2,
                linestyle=spacing_styles[cm],
                label=f"{cm} cm"
            )

        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_minor_locator(NullLocator())

        plt.title(f"{ylabel} — {shape}")
        plt.xlabel("Time of Day")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(param_dir / f"{param}_Spacings_{shape}.png", dpi=600)
        plt.close()

print("✅ ALL PARAMETERS × ALL COMBINATIONS VISUALIZED SUCCESSFULLY")
print(f"📁 Output directory: {out_root}")
