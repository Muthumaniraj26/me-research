import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the metrics data
df = pd.read_csv(r'C:\Users\muthumaniraj\OneDrive\Documents\me research\high_precision_solar_final.csv')
metrics = ["Train_Acc(R2)", "Test_Acc(R2)", "RMSE", "MAE", "MSE", "Precision*"]
algorithms = df['Model']

# =====================================================
# PART 1: COMBINED COMPARISON PLOT (All in One)
# =====================================================
fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 26))
plt.subplots_adjust(hspace=0.5)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for i, metric in enumerate(metrics):
    axes[i].plot(algorithms, df[metric], marker='s', markersize=8, 
                 linestyle='-', linewidth=2, color=colors[i], label=f"Metric: {metric}")
    axes[i].set_title(f'Comparative Analysis: {metric}', fontsize=14, fontweight='bold')
    axes[i].set_ylabel(metric, fontsize=12)
    axes[i].grid(True, which='both', linestyle='--', alpha=0.5)
    axes[i].tick_params(axis='x', rotation=30)
    axes[i].legend(loc='best')

plt.tight_layout()
plt.savefig('combined_forecasting_metrics.png')
print("✅ Combined plot saved as 'combined_forecasting_metrics.png'")

# =====================================================
# PART 2: SEPARATE PLOTS (One Plot per Metric)
# =====================================================
for i, metric in enumerate(metrics):
    plt.figure(figsize=(10, 6))
    plt.plot(algorithms, df[metric], marker='o', markersize=10, 
             linestyle='-', linewidth=2.5, color=colors[i])
    
    plt.title(f'Algorithm Performance - {metric}', fontsize=16, fontweight='bold')
    plt.xlabel('Forecasting Algorithms', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Format filename for saving
    clean_name = metric.replace('*', '').replace('(', '').replace(')', '').replace(' ', '_')
    plt.tight_layout()
    plt.savefig(f'Separate_Plot_{clean_name}.png')

    import os
    os.makedirs('plots_final', exist_ok=True)
    os.replace(f'Separate_Plot_{clean_name}.png', os.path.join('plots_final', f'Separate_Plot_{clean_name}.png'))
    plt.close() # Close to save memory

    print(f"✅ Individual plot saved as 'Separate_Plot_{clean_name}.png'")

print("\n🎯 Visualization Suite Complete.")