import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv(r"C:\Users\muthumaniraj\OneDrive\Documents\me research\PV_Research_chat_recheck\PV_ALL_DATA_MASTER.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df['ISO_Week'] = df['Date'].dt.isocalendar().week

# Calculate Energy in kWh/m2 (Power * 15min / 60min / 1000)
df['Irr_kWh_m2'] = df['Irradiation_Wm2'] * (15 / 60) / 1000

# Aggregate by Week
weekly = df.groupby(['ISO_Week', 'Shape', 'Spacing_cm'])['Irr_kWh_m2'].sum().reset_index()

# Setup Styling
sns.set_style("whitegrid")
plt.figure(figsize=(16, 9))

colors = {"Flat": "#D32F2F", "V-Shape": "#1976D2", "Inverted-V": "#388E3C"}
styles = {62: (0, (1, 1)), 77: (0, (5, 5)), 93: 'solid'}

# Plot each configuration
for shape in ["Flat", "V-Shape", "Inverted-V"]:
    for spacing in [62, 77, 93]:
        subset = weekly[(weekly['Shape'] == shape) & (weekly['Spacing_cm'] == spacing)]
        
        # Make the 'Best' configuration stand out
        lw = 2.5 if (shape == "Inverted-V" and spacing == 93) else 1.5
        alpha = 1.0 if (shape == "Inverted-V" and spacing == 93) else 0.7
        
        plt.plot(subset['ISO_Week'], subset['Irr_kWh_m2'],
                 label=f"{shape} ({spacing}cm)", color=colors[shape],
                 linestyle=styles[spacing], linewidth=lw, alpha=alpha,
                 marker='o' if spacing == 93 else None, markersize=4)

# Formatting
plt.title("Weekly Solar Irradiation Yield: Comparative Analysis of Geometries", fontsize=18, fontweight='bold', pad=20)
plt.xlabel("ISO Week Number", fontsize=14)
plt.ylabel("Cumulative Weekly Irradiation ($kWh/m^2$)", fontsize=14)
plt.legend(title="Configuration", bbox_to_anchor=(1.02, 1), loc='upper left', shadow=True)

plt.tight_layout()
plt.savefig("enhanced_weekly_irradiation_comparison.png", dpi=300)