import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("📊 Local Month-wise Shape Winner Analysis Started")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
CSV_PATH = "PV_Research_split_equal/PV_ALL_DATA_MASTER.csv"
df = pd.read_csv(CSV_PATH)

# --------------------------------------------------
# PREPARE DATA
# --------------------------------------------------
# Monthly total irradiance per shape & spacing
monthly_irr = (
    df.groupby(["Month", "Shape", "Spacing_cm"], as_index=False)
      ["Irradiation_Wm2"]
      .sum()
)

# Identify winner per month & spacing
def get_winner(sub):
    return sub.loc[sub["Irradiation_Wm2"].idxmax(), "Shape"]

winners = (
    monthly_irr
    .groupby(["Month", "Spacing_cm"])
    .apply(get_winner)
    .reset_index(name="Winning_Shape")
)

# Encode shapes numerically for heatmap
shape_code = {"Flat": 0, "V-Shape": 1, "Inverted-V": 2}
winners["Code"] = winners["Winning_Shape"].map(shape_code)

# Month order
month_order = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]
winners["Month"] = pd.Categorical(winners["Month"], month_order, ordered=True)

# --------------------------------------------------
# PLOTTING
# --------------------------------------------------
for spacing in sorted(winners["Spacing_cm"].unique()):

    data = winners[winners["Spacing_cm"] == spacing]
    pivot = data.pivot(index="Month", columns="Spacing_cm", values="Code")

    plt.figure(figsize=(4, 8))
    sns.heatmap(
        pivot,
        annot=data["Winning_Shape"].values.reshape(-1,1),
        fmt="",
        cmap=["#4C72B0", "#DD8452", "#55A868"],
        cbar=False,
        linewidths=0.5,
        linecolor="gray"
    )

    plt.title(f"Local Monthly Shape Winner (Irradiance)\nSpacing = {spacing} cm")
    plt.ylabel("Month")
    plt.xlabel("")

    plt.tight_layout()
    plt.savefig(f"local_monthly_winner_{spacing}cm.png", dpi=300)
    plt.show()

print("✅ Heatmaps generated successfully")
