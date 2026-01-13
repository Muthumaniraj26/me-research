import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ============================================================
# 1. LOAD DATA
# ============================================================
data_dir = Path("results")                 # folder with *_Summer_15min.csv
out_dir = Path("plots_NO_BASELINE_FINAL")  # output plots
out_dir.mkdir(parents=True, exist_ok=True)

csv_files = list(data_dir.glob("*_Summer_15min.csv"))
if not csv_files:
    raise FileNotFoundError("❌ No CSV files found in 'results' directory")

df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

# ------------------------------------------------------------
# 2. CLEAN & NORMALIZE (CRITICAL FIX)
# ------------------------------------------------------------
df.columns = df.columns.str.strip()

df["Connection"] = (
    df["Connection"]
    .astype(str)
    .str.strip()
    .replace({
        "Se-P": "SeP",
        "Series-Parallel": "SeP",
        "Series Parallel": "SeP",
        "SP": "SeP"
    })
)

# Ensure correct time parsing
df["Time"] = df["Time"].astype(str)
df["Time_dt"] = pd.to_datetime(df["Time"], format="%H:%M")

# ------------------------------------------------------------
# 3. AVERAGE DAILY PROFILE (NO FAKE DATA)
# ------------------------------------------------------------
avg_df = (
    df.groupby(["Connection", "Shape", "Spacing_cm", "Time"], as_index=False)
      .mean(numeric_only=True)
)

# Daily energy (Wh) — true integration of 15-min samples
energy_df = (
    df.groupby(["Connection", "Shape", "Spacing_cm"])["Pm_W"]
      .sum() * 0.25
).reset_index()

energy_df.rename(columns={"Pm_W": "Energy_Wh"}, inplace=True)

# ------------------------------------------------------------
# 4. VISUAL STYLES (ALL EQUAL — NO BASELINE)
# ------------------------------------------------------------
connections = ["SeP", "TCT", "TSSC"]

color_map = {
    "SeP": "#1f77b4",   # Blue
    "TCT": "#2ca02c",   # Green
    "TSSC": "#d62728"   # Red
}

marker_map = {
    "SeP": "s",
    "TCT": "o",
    "TSSC": "^"
}

# ============================================================
# 5. GENERATE PLOTS
# ============================================================
for shape in avg_df["Shape"].unique():
    for cm in sorted(avg_df["Spacing_cm"].unique()):

        fig = plt.figure(figsize=(14, 12))
        gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.2])

        ax_line = fig.add_subplot(gs[0, :])
        ax_bar  = fig.add_subplot(gs[1, 0])
        ax_tbl  = fig.add_subplot(gs[1, 1])

        subset_line = avg_df[
            (avg_df["Shape"] == shape) &
            (avg_df["Spacing_cm"] == cm)
        ]

        subset_energy = energy_df[
            (energy_df["Shape"] == shape) &
            (energy_df["Spacing_cm"] == cm)
        ]

        # ----------------------------------------------------
        # A. ABSOLUTE POWER LINE PLOT (NO BASELINE)
        # ----------------------------------------------------
        for conn in connections:
            d = subset_line[subset_line["Connection"] == conn]
            if d.empty:
                continue

            ax_line.plot(
                pd.to_datetime(d["Time"], format="%H:%M"),
                d["Pm_W"],
                label=conn,
                color=color_map[conn],
                marker=marker_map[conn],
                linewidth=3,
                markersize=8,
                alpha=0.95
            )

        ax_line.set_title(
            f"Absolute Power Output — {shape} | {cm} cm (Summer)",
            fontsize=16, fontweight="bold"
        )
        ax_line.set_ylabel("Power (W)", fontsize=12)
        ax_line.set_xlabel("Time of Day", fontsize=12)
        ax_line.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax_line.grid(True, linestyle="--", alpha=0.4)
        ax_line.legend(frameon=True, shadow=True)

        # ----------------------------------------------------
        # B. DAILY ENERGY BAR CHART (Wh)
        # ----------------------------------------------------
        bars = ax_bar.bar(
            subset_energy["Connection"],
            subset_energy["Energy_Wh"],
            color=[color_map[c] for c in subset_energy["Connection"]],
            edgecolor="black"
        )

        ax_bar.set_title("Total Daily Energy Yield", fontsize=13, fontweight="bold")
        ax_bar.set_ylabel("Energy (Wh)")

        for bar in bars:
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():.1f}",
                ha="center",
                va="bottom",
                fontweight="bold"
            )

        # ----------------------------------------------------
        # C. SUMMARY TABLE
        # ----------------------------------------------------
        table_data = [["Connection", "Peak Power (W)", "Energy (Wh)"]]
        for conn in connections:
            peak = subset_line[subset_line["Connection"] == conn]["Pm_W"].max()
            energy = subset_energy[subset_energy["Connection"] == conn]["Energy_Wh"].values
            if len(energy) == 0:
                continue
            table_data.append([conn, f"{peak:.2f}", f"{energy[0]:.2f}"])

        ax_tbl.axis("off")
        tbl = ax_tbl.table(cellText=table_data, loc="center", cellLoc="center")
        tbl.scale(1.2, 2)
        ax_tbl.set_title("Performance Summary", fontweight="bold")

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------
        plt.tight_layout()
        plt.savefig(
            out_dir / f"ABSOLUTE_COMPARE_{shape}_{cm}cm.png",
            dpi=300
        )
        plt.show()

print("\n✅ SUCCESS")
print(f"📁 Plots saved in: {out_dir}")
