import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, GRU, Conv1D, Bidirectional, 
                                     Multiply, Add, LayerNormalization, Activation, 
                                     MultiHeadAttention)

# =====================================================
# 0. REPRODUCIBILITY & CONFIG
# =====================================================
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds(42)

DATA_FILE = r"C:\Users\muthumaniraj\OneDrive\Documents\me research\Final _PV_FULL_DATASET\PV_MASTER_15MIN_2024_2025.csv"
LOOKBACK = 24
EPOCHS = 50 # Adjusted for quick testing; use 120 for final paper
BATCH = 64

# =====================================================
# 1. DATA LOADING
# =====================================================
df = pd.read_csv(DATA_FILE)
features = ["Irradiance_Wm2", "AmbientTemp_C", "CellTemp_C"]
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[features].values)

X, y = [], []
for i in range(len(data_scaled) - LOOKBACK):
    X.append(data_scaled[i:i+LOOKBACK])
    y.append(data_scaled[i+LOOKBACK, 0])
X, y = np.array(X), np.array(y)

split = int(0.8 * len(X))
X_train, y_train, X_test, y_test = X[:split], y[:split], X[split:], y[split:]

# =====================================================
# 2. MODEL DEFINITIONS
# =====================================================

def build_GR_RNN_Base(lb, nv):
    inputs = Input(shape=(lb, nv))
    x = Bidirectional(GRU(64, return_sequences=True))(inputs)
    res = Conv1D(128, 1)(inputs) 
    x = Add()([x, res])
    x = LayerNormalization()(x)
    gate = Dense(128, activation='sigmoid')(x)
    x = Multiply()([x, gate])
    x = GRU(64, return_sequences=False)(x)
    outputs = Dense(1)(Activation('gelu')(Dense(64)(x)))
    model = Model(inputs, outputs, name="GR_RNN_Base")
    model.compile(loss="huber", optimizer="adam")
    return model

def build_GR_RNN_Attention(lb, nv):
    inputs = Input(shape=(lb, nv))
    x = Bidirectional(GRU(64, return_sequences=True))(inputs)
    res = Conv1D(128, 1)(inputs) 
    x = Add()([x, res])
    x = LayerNormalization()(x)
    
    # Attention Layer
    attn_out = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = Add()([x, attn_out])
    x = LayerNormalization()(x)

    gate = Dense(128, activation='sigmoid')(x)
    x = Multiply()([x, gate])
    x = GRU(64, return_sequences=False)(x)
    outputs = Dense(1)(Activation('gelu')(Dense(64)(x)))
    model = Model(inputs, outputs, name="GR_RNN_Attention")
    model.compile(loss="huber", optimizer="adam")
    return model

# =====================================================
# 3. TRAINING & EVALUATION
# =====================================================
n_vars = X.shape[2]
models = [build_GR_RNN_Base(LOOKBACK, n_vars), build_GR_RNN_Attention(LOOKBACK, n_vars)]
results = {}

for model in models:
    print(f"\n🔥 Training {model.name}...")
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH, verbose=0)
    preds = model.predict(X_test).flatten()
    results[model.name] = preds
    
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"✅ {model.name} Metrics -> R2: {r2:.6f}, RMSE: {rmse:.6f}")

# =====================================================
# 4. VISUALIZATION
# =====================================================
# We will plot the last 100 points for clarity
plot_range = 100 
plt.figure(figsize=(15, 10))

# Subplot 1: Base Model Visualization
plt.subplot(3, 1, 1)
plt.plot(y_test[-plot_range:], label="Actual Irradiance", color="black", linewidth=2)
plt.plot(results["GR_RNN_Base"][-plot_range:], label="Base Prediction", color="blue", linestyle="--")
plt.title("Gated-Residual RNN (Base) Performance")
plt.legend()

# Subplot 2: Attention Model Visualization
plt.subplot(3, 1, 2)
plt.plot(y_test[-plot_range:], label="Actual Irradiance", color="black", linewidth=2)
plt.plot(results["GR_RNN_Attention"][-plot_range:], label="Attention Prediction", color="green", linestyle="--")
plt.title("Gated-Residual RNN (With Attention) Performance")
plt.legend()

# Subplot 3: Comparative Visualization
plt.subplot(3, 1, 3)
plt.plot(y_test[-plot_range:], label="Actual", color="black", alpha=0.5)
plt.plot(results["GR_RNN_Base"][-plot_range:], label="Base", color="blue", alpha=0.7)
plt.plot(results["GR_RNN_Attention"][-plot_range:], label="Attention", color="green", alpha=0.7)
plt.title("Direct Comparison: Base vs. Attention")
plt.xlabel("Time Steps (15-min intervals)")
plt.legend()

plt.tight_layout()
plt.show()