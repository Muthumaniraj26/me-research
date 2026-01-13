# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path

# print("📊 Relative irradiance visualization started")

# # --------------------------------------------------
# # PATHS
# # --------------------------------------------------
# BASE_DIR = Path(r"C:\Users\muthumaniraj\Documents\me research\PV_Research_chat_recheck")
# CSV_FILE = BASE_DIR / r"C:\Users\muthumaniraj\Documents\me research\PV_Research_chat_recheck\PV_ALL_DATA_MASTER.csv"
# PLOT_DIR = BASE_DIR / "plots_irradiance_relative"
# PLOT_DIR.mkdir(parents=True, exist_ok=True)

# # --------------------------------------------------
# # LOAD DATA
# # --------------------------------------------------
# df = pd.read_csv(CSV_FILE)

# # Use daytime only
# df = df[df["Irradiation_Wm2"] > 100]

# # Bin irradiance
# df["Irr_bin"] = (df["Irradiation_Wm2"] // 50) * 50

# # --------------------------------------------------
# # MEAN IRRADIANCE
# # --------------------------------------------------
# mean_irr = (
#     df.groupby(["Season", "Shape", "Irr_bin"], as_index=False)
#       .agg({"Irradiation_Wm2": "mean"})
# )

# # --------------------------------------------------
# # RELATIVE TO FLAT
# # --------------------------------------------------
# flat_ref = mean_irr[mean_irr["Shape"] == "Flat"][
#     ["Season", "Irr_bin", "Irradiation_Wm2"]
# ].rename(columns={"Irradiation_Wm2": "Flat_Irr"})

# merged = mean_irr.merge(flat_ref, on=["Season", "Irr_bin"])

# merged["Relative_Irr_%"] = (
#     merged["Irradiation_Wm2"] / merged["Flat_Irr"] * 100
# )

# # --------------------------------------------------
# # PLOT
# # --------------------------------------------------
# for season in merged["Season"].unique():

#     plt.figure(figsize=(9, 6))
#     season_df = merged[merged["Season"] == season]

#     for shape, sub in season_df.groupby("Shape"):
#         plt.plot(
#             sub["Irr_bin"],
#             sub["Relative_Irr_%"],
#             marker="o",
#             linewidth=2.5,
#             label=shape
#         )

#     plt.axhline(100, color="black", linestyle="--", linewidth=1)

#     plt.xlabel("Solar Irradiance Bin (W/m²)")
#     plt.ylabel("Relative Irradiance (%)")
#     plt.title(f"Relative Irradiance Gain by Shape – {season}")
#     plt.ylim(98, 112)   # 🔥 THIS MAKES DIFFERENCE CLEAR
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()

#     out = PLOT_DIR / f"relative_irradiance_{season}.png"
#     plt.savefig(out, dpi=300)
#     plt.close()

#     print(f"✅ Saved: {out}")

# print("\n🎯 Relative irradiance plots generated")
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

print("📊 Absolute irradiance comparison (zoomed scale) started")

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = Path(r"C:\Users\muthumaniraj\Documents\me research\PV_Research_chat_recheck")
CSV_FILE = BASE_DIR / r"C:\Users\muthumaniraj\Documents\me research\PV_Research_chat_recheck\PV_ALL_DATA_MASTER.csv"
PLOT_DIR = BASE_DIR / "plots_irradiance_absolute_1"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv(CSV_FILE)

# Daytime filter
df = df[df["Irradiation_Wm2"] > 100]

# Bin irradiance for smooth curves
df["Irr_bin"] = (df["Irradiation_Wm2"] // 50) * 50

# Mean irradiance per shape
summary = (
    df.groupby(["Season", "Shape", "Irr_bin"], as_index=False)
      .agg({"Irradiation_Wm2": "mean"})
)

# --------------------------------------------------
# PLOT
# --------------------------------------------------
for season in summary["Season"].unique():

    plt.figure(figsize=(9, 6))
    season_df = summary[summary["Season"] == season]

    for shape, sub in season_df.groupby("Shape"):
        plt.plot(
            sub["Irr_bin"],
            sub["Irradiation_Wm2"],
            marker="o",
            linewidth=2.5,
            label=shape
        )

    # 🔍 Zoomed Y-axis (KEY LINE)
    ymin = season_df["Irradiation_Wm2"].min() - 20
    ymax = season_df["Irradiation_Wm2"].max() + 20
    plt.ylim(ymin, ymax)

    plt.xlabel("Solar Irradiance Bin (W/m²)")
    plt.ylabel("Mean Irradiance (W/m²)")
    plt.title(f"Absolute Irradiance Comparison by Shape – {season}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out = PLOT_DIR / f"absolute_irradiance_{season}.png"
    plt.savefig(out, dpi=300)
    plt.close()

    print(f"✅ Saved: {out}")

print("\n🎯 Absolute irradiance plots generated successfully")
