import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# =====================================================
# 0. REPRODUCIBILITY BLOCK (CRITICAL)
# =====================================================
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

set_seeds(42)
print("🚀 Seeds Set (42). Training is now Deterministic.")

# =====================================================
# 1. CONFIGURATION
# =====================================================
DATA_FILE = r"C:\Users\muthumaniraj\OneDrive\Documents\me research\Final _PV_FULL_DATASET\PV_MASTER_15MIN_2024_2025.csv"
TARGET = "Irradiance_Wm2"
EXOGENOUS = ["AmbientTemp_C", "CellTemp_C"] 
LOOKBACK = 24  
EPOCHS = 60  # Professional standard for your paper
BATCH = 64

# Load and Preprocess
df = pd.read_csv(DATA_FILE)
df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
df = df.sort_values("Datetime")

features = [TARGET] + EXOGENOUS
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[features].values)

def make_sequences(data, step):
    X, y = [], []
    for i in range(len(data) - step):
        X.append(data[i:i+step, :])
        y.append(data[i+step, 0])
    return np.array(X), np.array(y)

X, y = make_sequences(data_scaled, LOOKBACK)
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# =====================================================
# 2. MODEL DEFINITIONS (Base vs. Attention)
# =====================================================

def build_GR_RNN(lb, nv, use_attention=False):
    inputs = tf.keras.layers.Input(shape=(lb, nv))
    # Recurrent Core
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True))(inputs)
    # Residual Path
    res = tf.keras.layers.Conv1D(128, 1)(inputs) 
    x = tf.keras.layers.Add()([x, res])
    x = tf.keras.layers.LayerNormalization()(x)
    
    if use_attention:
        # Multi-Head Attention Block
        attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
        x = tf.keras.layers.Add()([x, attn])
        x = tf.keras.layers.LayerNormalization()(x)

    # Gating Logic
    gate = tf.keras.layers.Dense(128, activation='sigmoid')(x)
    x = tf.keras.layers.Multiply()([x, gate])
    
    # Output Projection
    x = tf.keras.layers.GRU(64, return_sequences=False)(x)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.Activation('gelu')(x)
    outputs = tf.keras.layers.Dense(1)(x)
    
    name = "GR_RNN_Attention" if use_attention else "GR_RNN_Base"
    model = tf.keras.models.Model(inputs, outputs, name=name)
    model.compile(loss="huber", optimizer=tf.keras.optimizers.AdamW(1e-3))
    return model

# =====================================================
# 3. TRAINING & EVALUATION
# =====================================================
n_vars = X.shape[2]
model_base = build_GR_RNN(LOOKBACK, n_vars, use_attention=False)
model_attn = build_GR_RNN(LOOKBACK, n_vars, use_attention=True)

eval_metrics = []
predictions = {}

for model in [model_base, model_attn]:
    print(f"\n📘 Training Architecture: {model.name}...")
    es = tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH, verbose=1, 
              validation_split=0.1, callbacks=[es])
    
    preds = model.predict(X_test, verbose=0).flatten()
    predictions[model.name] = preds
    
    # Calculation
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    
    eval_metrics.append({"Model": model.name, "R2": r2, "RMSE": rmse, "MAE": mae})
    print(f"✅ {model.name} Done. R2: {r2:.8f}")

eval_df = pd.DataFrame(eval_metrics)

# =====================================================
# 4. FULL RESEARCH VISUALIZATION
# =====================================================
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2)

# Subplot 1: Prediction Tracking (Last 100 points)
ax1 = fig.add_subplot(gs[0, :])
p_range = 100
ax1.plot(y_test[-p_range:], label="Actual", color='black', linewidth=2)
ax1.plot(predictions["GR_RNN_Base"][-p_range:], label="Base GR-RNN", linestyle='--', color='blue')
ax1.plot(predictions["GR_RNN_Attention"][-p_range:], label="Attention GR-RNN", linestyle=':', color='green')
ax1.set_title("Temporal Prediction Tracking (15-min Intervals)")
ax1.legend()

# Subplot 2: RMSE Comparison
ax2 = fig.add_subplot(gs[1, 0])
sns.barplot(x="Model", y="RMSE", data=eval_df, ax=ax2, palette="Blues_d")
ax2.set_title("Root Mean Square Error (RMSE) - Lower is Better")

# Subplot 3: Residual Error (Actual - Predicted)
ax3 = fig.add_subplot(gs[1, 1])
res_base = y_test - predictions["GR_RNN_Base"]
res_attn = y_test - predictions["GR_RNN_Attention"]
ax3.plot(res_base[-p_range:], label="Base Error", color='red', alpha=0.5)
ax3.plot(res_attn[-p_range:], label="Attention Error", color='green', alpha=0.8)
ax3.axhline(0, color='black', linewidth=1)
ax3.set_title("Residual Error Analysis")
ax3.legend()

# Subplot 4: Error Distribution
ax4 = fig.add_subplot(gs[2, 0])
sns.kdeplot(res_base, fill=True, label="Base", ax=ax4)
sns.kdeplot(res_attn, fill=True, label="Attention", ax=ax4)
ax4.set_title("Error Density Distribution")
ax4.legend()

# Subplot 5: Numerical Summary Table
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('off')
table = ax5.table(cellText=eval_df.round(8).values, colLabels=eval_df.columns, loc='center', cellLoc='center')
table.scale(1, 2)
ax5.set_title("Final Performance Matrix")

plt.tight_layout()
plt.show()