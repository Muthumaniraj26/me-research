import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the dataset
# Ensure the file 'PV_ALL_DATA_MASTER_WEEKWISE.csv' is in your working directory
df = pd.read_csv(r'C:\Users\muthumaniraj\OneDrive\Documents\me research\PV_Research_chat_recheck\PV_ALL_DATA_MASTER.csv')

# Ensure Date is parsed correctly
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
# Create ISO_Week column
df['ISO_Week'] = df['Date'].dt.isocalendar().week

# 2. Group by ISO_Week, Time, and Shape
# This calculates the mean irradiation for each specific time slot across different weeks
pivot_df = df.groupby(['ISO_Week', 'Time', 'Shape'])['Irradiation_Wm2'].mean().unstack()

# 3. Calculate Performance Deviation (Delta)
# Delta = (Inverted-V Performance) - (Flat Baseline)
# A positive value indicates how much extra energy the Inverted-V captures
pivot_df['Delta'] = pivot_df['Inverted-V'] - pivot_df['Flat']

# 4. Reshape for Heatmap Visualization
# Rows = Time of Day (Daily Scaling)
# Columns = ISO Week (Weekly/Seasonal Scaling)
heatmap_data = pivot_df['Delta'].reset_index().pivot(index='Time', columns='ISO_Week', values='Delta')

# 5. Correct Time Sorting
# Convert index to datetime objects to ensure 09:00 comes before 10:00 correctly
heatmap_data.index = pd.to_datetime(heatmap_data.index, format='%H:%M').time
heatmap_data = heatmap_data.sort_index()

# 6. Create the Visualization
plt.figure(figsize=(16, 10))
sns.heatmap(
    heatmap_data, 
    cmap='magma', 
    cbar_kws={'label': 'Irradiation Deviation (W/m²)'},
    linewidths=0.01,
    linecolor='white'
)

# 7. Add Professional Formatting
plt.title('Performance Deviation Spot Analysis: Inverted-V vs. Flat Panel', 
          fontsize=18, fontweight='bold', pad=20)
plt.xlabel('ISO Week Number (Seasonal Scaling)', fontsize=14)
plt.ylabel('Time of Day (Daily Scaling)', fontsize=14)

# Improve readability
plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.tight_layout()

# 8. Save and Display
# In a local environment, you can use plt.show()
plt.savefig('performance_deviation_heatmap.png', dpi=300)
print("Analysis Complete. Plot saved as 'performance_deviation_heatmap.png'")