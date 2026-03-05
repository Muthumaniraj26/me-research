import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
CSV_PATH = r"C:\Users\muthumaniraj\OneDrive\Documents\me research\PV_Research_chat_recheck\PV_ALL_DATA_MASTER.csv"   # adjust path if needed
df = pd.read_csv(CSV_PATH)

# --------------------------------------------------
# 1️⃣ ANNUAL IRRADIANCE — GLOBAL GEOMETRY EFFECT
# --------------------------------------------------
annual_irr = (
    df.groupby(["Shape", "Spacing_cm"], as_index=False)
      ["Irradiation_Wm2"].mean()
)

plt.figure(figsize=(10,5))
sns.barplot(
    data=annual_irr,
    x="Spacing_cm",
    y="Irradiation_Wm2",
    hue="Shape"
)
plt.title("Annual Average Effective Irradiance (Geometry Comparison)")
plt.ylabel("Average Irradiance (W/m²)")
plt.xlabel("Panel Spacing (cm)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 2️⃣ MONTHLY IRRADIANCE HEATMAP — LOCAL ADVANTAGE
# --------------------------------------------------
monthly_irr = (
    df.groupby(["Month", "Shape"], as_index=False)
      ["Irradiation_Wm2"].mean()
)

pivot_month = monthly_irr.pivot(
    index="Month",
    columns="Shape",
    values="Irradiation_Wm2"
)

plt.figure(figsize=(8,6))
sns.heatmap(
    pivot_month,
    annot=True,
    fmt=".0f",
    cmap="YlOrRd",
    linewidths=0.5
)
plt.title("Monthly Average Irradiance (Shape-wise)")
plt.ylabel("Month")
plt.xlabel("Shape")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 3️⃣ SEASONAL IRRADIANCE CONTRIBUTION
# --------------------------------------------------
seasonal_irr = (
    df.groupby(["Season", "Shape"], as_index=False)
      ["Irradiation_Wm2"].mean()
)

plt.figure(figsize=(9,5))
sns.barplot(
    data=seasonal_irr,
    x="Season",
    y="Irradiation_Wm2",
    hue="Shape"
)
plt.title("Seasonal Average Irradiance Capture by Geometry")
plt.ylabel("Average Irradiance (W/m²)")
plt.xlabel("Season")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 4️⃣ MONTHLY RANK SWITCHING — IRRADIANCE ONLY
# --------------------------------------------------
rank_df = monthly_irr.copy()
rank_df["Rank"] = (
    rank_df.groupby("Month")["Irradiation_Wm2"]
    .rank(ascending=False)
)

plt.figure(figsize=(9,5))
sns.lineplot(
    data=rank_df,
    x="Month",
    y="Rank",
    hue="Shape",
    marker="o"
)
plt.gca().invert_yaxis()
plt.title("Monthly Geometry Rank Based on Irradiance (1 = Best)")
plt.ylabel("Rank")
plt.xlabel("Month")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
