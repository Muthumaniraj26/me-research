import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# =====================================================
# CONFIGURATION
# =====================================================
PRED_FILE = "model_predictions.csv"
PLOTS_DIR = "plots"

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

# Try to load predictions
if not os.path.exists(PRED_FILE):
    print(f"⚠️ Prediction file '{PRED_FILE}' not found. Please run 'check2_novel.py' first.")
    # Create dummy data for demonstration
    print("Creating dummy data for demonstration...")
    timesteps = 500
    t = np.linspace(0, 4*np.pi, timesteps)
    actual = np.sin(t) * 100 + 500 + np.random.normal(0, 10, timesteps)
    
    data = {"Actual": actual}
    models = ["ANN", "LSTM", "BiLSTM", "GRU", "CNN-LSTM", "iTransformer", "Liquid-NN", "N-BEATS"]
    
    for m in models:
        noise = np.random.normal(0, 15, timesteps)
        data[m] = actual + noise * 0.5
    
    df = pd.DataFrame(data)
else:
    print(f"📂 Loading predictions from: {PRED_FILE}")
    df = pd.read_csv(PRED_FILE)

# Set style
plt.rcParams.update({'font.size': 12})
plt.style.use('ggplot') 
sns.set_palette("tab10")

# =====================================================
# 1. COMBINED LINE PLOT (ALL MODELS)
# =====================================================
def plot_combined(df, start=0, end=300):
    plt.figure(figsize=(16, 8))
    
    # Plot Actual
    plt.plot(df["Actual"][start:end], label="Actual Irradiance", color="black", linewidth=2.5, linestyle="-")
    
    # Plot Models
    models = [c for c in df.columns if c != "Actual"]
    for model in models:
        plt.plot(df[model][start:end], label=model, alpha=0.7, linewidth=1.5)
    
    plt.title(f"Model Predictions vs Actual (First {end-start} Timesteps)", fontsize=18, fontweight='bold')
    plt.ylabel("Irradiance (Wm²)", fontsize=14)
    plt.xlabel("Time Steps (15-min intervals)", fontsize=14)
    plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "all_models_comparison.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved {save_path}")
    plt.close()

plot_combined(df, start=0, end=300)

# =====================================================
# 2. INDIVIDUAL LINE PLOTS (Actual vs Predicted)
# =====================================================
models = [c for c in df.columns if c != "Actual"]
sub_start, sub_end = 0, 200 # Zoomed in for individual plots

for model in models:
    plt.figure(figsize=(14, 6))
    
    # Plot Actual vs Model
    plt.plot(df["Actual"][sub_start:sub_end], label="Actual", color="black", linewidth=2)
    plt.plot(df[model][sub_start:sub_end], label=f"{model} Prediction", color="dodgerblue", linewidth=2, linestyle="--")
    
    # Fill difference
    plt.fill_between(df.index[sub_start:sub_end], 
                     df["Actual"][sub_start:sub_end], 
                     df[model][sub_start:sub_end], 
                     color="red", alpha=0.1, label="Error")
    
    plt.title(f"{model}: Prediction Performance", fontsize=16, fontweight='bold')
    plt.ylabel("Irradiance (Wm²)")
    plt.xlabel("Time Steps")
    plt.legend()
    
    save_path = os.path.join(PLOTS_DIR, f"model_prediction_{model}.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved {save_path}")
    plt.close()

print("\n🚀 All visualizations generated in 'plots/' directory.")
