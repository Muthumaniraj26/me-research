import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
CSV_PATH = r"C:\Users\muthumaniraj\OneDrive\Documents\me research\PV_DAILY_NASA_DIRECT\PV_DAILY_ALL_SHAPES_SPACINGS.csv"
df = pd.read_csv(CSV_PATH)

df["Date"] = pd.to_datetime(df["Date"])
df["Day"] = df["Date"].dt.dayofyear

# --------------------------------------------------
# AGGREGATE: DAILY MEAN IRRADIANCE
# --------------------------------------------------
daily = (
    df.groupby(["Day", "Shape", "Spacing_cm"], as_index=False)
      .agg({"Irradiance_Wm2": "mean"})
)

# --------------------------------------------------
# PLOT 1: DAILY IRRADIANCE (ALL SHAPES, ALL SPACINGS)
# --------------------------------------------------
plt.figure(figsize=(16, 6))

for shape in ["Flat", "V-Shape", "Inverted-V"]:
    subset = daily[daily["Shape"] == shape]
    plt.plot(
        subset["Day"],
        subset["Irradiance_Wm2"],
        label=shape,
        linewidth=1.5
    )

plt.title("Daily Irradiance Comparison (All Shapes, All Spacings)")
plt.xlabel("Day of Year")
plt.ylabel("Irradiance (W/m²)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# PLOT 2: LOCAL OUTPERFORMANCE vs INVERTED-V
# --------------------------------------------------
pivot = daily.pivot_table(
    index="Day",
    columns="Shape",
    values="Irradiance_Wm2",
    aggfunc="mean"
).reset_index()

pivot["Flat_minus_IV"] = pivot["Flat"] - pivot["Inverted-V"]
pivot["V_minus_IV"]    = pivot["V-Shape"] - pivot["Inverted-V"]

plt.figure(figsize=(16, 6))
plt.plot(pivot["Day"], pivot["Flat_minus_IV"], label="Flat − Inverted-V")
plt.plot(pivot["Day"], pivot["V_minus_IV"], label="V-Shape − Inverted-V")
plt.axhline(0, color="black", linestyle="--", linewidth=1)

plt.title("Local Irradiance Advantage over Inverted-V (Daily)")
plt.xlabel("Day of Year")
plt.ylabel("Δ Irradiance (W/m²)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# PLOT 3: HEATMAP (LOCAL WINS)
# --------------------------------------------------
heatmap_df = pivot[["Day", "Flat_minus_IV", "V_minus_IV"]].set_index("Day")

plt.figure(figsize=(14, 5))
sns.heatmap(
    heatmap_df.T,
    cmap="coolwarm",
    center=0,
    cbar_kws={"label": "Δ Irradiance (W/m²)"}
)

plt.title("Heatmap: Local Irradiance Gain over Inverted-V")
plt.xlabel("Day of Year")
plt.ylabel("Shape Comparison")
plt.tight_layout()
plt.show()
