import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv(r"C:\Users\muthumaniraj\OneDrive\Documents\me research\PV_Research_Weekwise_All\PV_ALL_DATA_MASTER_WEEKWISE.csv")

# 2. Daily Irradiation Profile Plot
# Aggregating by Time and Shape to get the mean profile
daily_profile = df.groupby(['Time', 'Shape'])['Irradiation_Wm2'].mean().reset_index()
daily_profile['Time_dt'] = pd.to_datetime(daily_profile['Time'], format='%H:%M')
daily_profile = daily_profile.sort_values('Time_dt')

plt.figure(figsize=(12, 6))
sns.lineplot(data=daily_profile, x='Time', y='Irradiation_Wm2', hue='Shape', palette='viridis')
plt.title('Average Daily Irradiation Profile by Panel Shape')
plt.xlabel('Time of Day')
plt.ylabel('Irradiation (W/m²)')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('daily_irradiation_profile.png')

# 3. Shape Comparison Bar Chart
# Proving the "Best to Low" ranking
shape_comparison = df.groupby('Shape')['Irradiation_Wm2'].mean().reset_index()
shape_comparison = shape_comparison.sort_values('Irradiation_Wm2', ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(data=shape_comparison, x='Shape', y='Irradiation_Wm2', palette='magma')
plt.title('Comparison of Average Irradiation Received by Shape')
plt.ylabel('Average Irradiation (W/m²)')
plt.xlabel('Panel Configuration')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('shape_irradiation_comparison.png')