import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
CSV_PATH = r"C:\Users\muthumaniraj\OneDrive\Documents\me research\PV_Research_chat_recheck\PV_ALL_DATA_MASTER.csv"   # adjust if needed
df = pd.read_csv(CSV_PATH)

# 15-min energy
df["Energy_kWh"] = df["Pm_W"] * 0.25 / 1000

# --------------------------------------------------
# 1️⃣ ANNUAL ENERGY — GLOBAL WINNER
# --------------------------------------------------
annual = (
    df.groupby(["Shape", "Spacing_cm"], as_index=False)
      ["Energy_kWh"].sum()
)

plt.figure(figsize=(10,5))
sns.barplot(
    data=annual,
    x="Spacing_cm",
    y="Energy_kWh",
    hue="Shape"
)
plt.title("Annual Energy Comparison (Global Performance)")
plt.ylabel("Annual Energy (kWh)")
plt.xlabel("Panel Spacing (cm)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 2️⃣ MONTHLY ENERGY HEATMAP — LOCAL WINNERS
# --------------------------------------------------
monthly = (
    df.groupby(["Month", "Shape"], as_index=False)
      ["Energy_kWh"].sum()
)

pivot = monthly.pivot(
    index="Month",
    columns="Shape",
    values="Energy_kWh"
)

plt.figure(figsize=(8,6))
sns.heatmap(
    pivot,
    annot=True,
    fmt=".1f",
    cmap="YlGnBu",
    linewidths=0.5
)
plt.title("Monthly Energy Heatmap (Local Shape Dominance)")
plt.ylabel("Month")
plt.xlabel("Shape")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 3️⃣ MONTHLY RANK SWITCHING (WHO WINS EACH MONTH)
# --------------------------------------------------
monthly_rank = monthly.copy()
monthly_rank["Rank"] = (
    monthly_rank.groupby("Month")["Energy_kWh"]
    .rank(ascending=False)
)

plt.figure(figsize=(9,5))
sns.lineplot(
    data=monthly_rank,
    x="Month",
    y="Rank",
    hue="Shape",
    marker="o"
)
plt.gca().invert_yaxis()
plt.title("Monthly Rank Switching (1 = Best)")
plt.ylabel("Rank")
plt.xlabel("Month")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 4️⃣ SEASONAL CONTRIBUTION — WHY INVERTED-V WINS ANNUALLY
# --------------------------------------------------
seasonal = (
    df.groupby(["Season", "Shape"], as_index=False)
      ["Energy_kWh"].sum()
)

plt.figure(figsize=(9,5))
sns.barplot(
    data=seasonal,
    x="Season",
    y="Energy_kWh",
    hue="Shape"
)
plt.title("Seasonal Energy Contribution (Explains Annual Dominance)")
plt.ylabel("Energy (kWh)")
plt.xlabel("Season")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
d