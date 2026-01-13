import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 1. LOAD AND PREPROCESS DATA
data_dir = Path(r"me research\tssc\results_tssc")
out_dir = Path(r"me research\tssc\plot\plots_TSSC")
out_dir.mkdir(parents=True, exist_ok=True)

csv_files = list(data_dir.glob("*_TSSC_Summer_15min.csv"))
if not csv_files:
    print("❌ No CSV files found! Check your path.")
    exit()

df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

# Convert Time to actual Time objects for proper sorting on X-axis
df["Time"] = pd.to_datetime(df["Time"], format='%H:%M').dt.time

# Calculate Average Daily Profile (to avoid messy overlapping lines from multiple days)
avg_profile = df.groupby(["Shape", "Spacing_cm", "Time"]).mean(numeric_only=True).reset_index()

# 2. DEFINE STYLES
shape_colors = {"Flat": "#1f77b4", "V-Shape": "#2ca02c", "Inverted-V": "#d62728"}
shape_markers = {"Flat": "o", "V-Shape": "s", "Inverted-V": "^"}
spacing_styles = {62: ":", 77: "--", 93: "-"}

# ============================================================
# PLOT 1: ALL SHAPES PER SPACING (Side-by-Side Comparison)
# ============================================================
for cm in [62, 77, 93]:
    plt.figure(figsize=(10, 6))
    subset = avg_profile[avg_profile["Spacing_cm"] == cm]
    
    for shape in ["Flat", "V-Shape", "Inverted-V"]:
        data = subset[subset["Shape"] == shape]
        plt.plot(
            data["Time"].astype(str), data["Pm_W"], 
            label=f"{shape}", 
            color=shape_colors[shape], 
            marker=shape_markers[shape],
            linewidth=2, markersize=6
        )
    
    plt.title(f"Solar Power Profile: Shape Comparison at {cm}cm Spacing", fontsize=14, fontweight='bold')
    plt.xlabel("Time of Day", fontsize=12)
    plt.ylabel("Power Output (Pm_W)", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title="Panel Shape", frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / f"Compare_Shapes_{cm}cm.png", dpi=300)
    plt.show()

# ============================================================
# PLOT 2: SPACING IMPACT PER SHAPE
# ============================================================
for shape in ["Flat", "V-Shape", "Inverted-V"]:
    plt.figure(figsize=(10, 6))
    subset = avg_profile[avg_profile["Shape"] == shape]
    
    for cm in [62, 77, 93]:
        data = subset[subset["Spacing_cm"] == cm]
        plt.plot(
            data["Time"].astype(str), data["Pm_W"], 
            label=f"Spacing: {cm}cm", 
            linestyle=spacing_styles[cm],
            color=shape_colors[shape], # Keep same base color for the shape
            marker="o", markersize=4, alpha=0.8
        )
    
    plt.title(f"Impact of Spacing on {shape} Configuration", fontsize=14, fontweight='bold')
    plt.xlabel("Time of Day", fontsize=12)
    plt.ylabel("Power Output (Pm_W)", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title="Spacing Depth", frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / f"Compare_Spacings_{shape}.png", dpi=300)
    plt.show()

# ============================================================
# PLOT 3: TOTAL EFFICIENCY (BAR CHART)
# ============================================================
# This shows the "Bottom Line" — which combination wins overall?
summary_df = avg_profile.groupby(["Shape", "Spacing_cm"])["Pm_W"].mean().unstack()

summary_df.plot(kind='bar', figsize=(10, 6), color=['#a1c9f4', '#8de5a1', '#ff9f9b'])
plt.title("Average Daily Power Generation Across All Combinations", fontsize=14, fontweight='bold')
plt.ylabel("Mean Power (W)")
plt.xlabel("Array Shape")
plt.xticks(rotation=0)
plt.legend(title="Spacing (cm)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(out_dir / "Total_Efficiency_Comparison.png", dpi=300)
plt.show()

print(f"✅ All enhanced plots generated and saved to: {out_dir}")