import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, Dense, LSTM, GRU, Conv1D, Flatten, Reshape, 
                                     Add, Subtract, Multiply, RNN, Bidirectional, 
                                     Dropout, MultiHeadAttention, LayerNormalization, 
                                     Permute, GlobalAveragePooling1D, MaxPooling1D, SimpleRNN)

# =====================================================
# 0. GPU INITIALIZATION (TF 2.11+ COMPATIBLE)
# =====================================================
print(f"🔍 TensorFlow Version: {tf.__version__}")
print("🔍 Checking System for GPU...")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Detected: {len(gpus)} device(s) active. Running on GPU Mode.")
    except RuntimeError as e:
        print(f"⚠️ GPU Initialization Error: {e}")
else:
    print("ℹ️ No GPU detected by TensorFlow. defaulting to CPU.")

# =====================================================
# 1. CONFIGURATION & DATA PREPROCESSING
# =====================================================
DATA_FILE = r"C:\Users\muthumaniraj\OneDrive\Documents\me research\Final _PV_FULL_DATASET\PV_MASTER_15MIN_2024_2025.csv"
TARGET = "Irradiance_Wm2"
EXOGENOUS = ["AmbientTemp_C", "CellTemp_C"] 
LOOKBACK = 24  
EPOCHS = 25
BATCH = 128 

# Load and Sort
df = pd.read_csv(DATA_FILE)
df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
df = df.sort_values("Datetime")

# Feature Scaling
features = [TARGET] + EXOGENOUS
data = df[features].values
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Sequence Generation
def make_sequences(data, step):
    X, y = [], []
    for i in range(len(data) - step):
        X.append(data[i:i+step, :])
        y.append(data[i+step, 0])
    return np.array(X), np.array(y)

X, y = make_sequences(data_scaled, LOOKBACK)
train_split = int(0.7 * len(X))
val_split = int(0.8 * len(X))

X_train, y_train = X[:train_split], y[:train_split]
X_val, y_val = X[train_split:val_split], y[train_split:val_split]
X_test, y_test = X[val_split:], y[val_split:]

# =====================================================
# 2. CUSTOM ARCHITECTURE MODULES
# =====================================================

class LiquidRNNCell(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units, self.state_size = units, units
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units), initializer='glorot_uniform')
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units), initializer='orthogonal')
        self.tau = self.add_weight(shape=(self.units,), initializer='ones')
    def call(self, inputs, states):
        prev_h = states[0]
        gate = tf.nn.tanh(tf.matmul(inputs, self.kernel) + tf.matmul(prev_h, self.recurrent_kernel))
        h = prev_h + (1.0 / (tf.exp(self.tau) + 1.0)) * (-prev_h + gate)
        return h, [h]

def nbeats_block(input_layer, lookback, n_vars=1, units=256):
    x = Flatten()(input_layer)
    x = Dense(units, activation='relu')(x)
    x = Dense(units, activation='relu')(x)
    backcast = Dense(lookback * n_vars)(x)
    backcast = Reshape((lookback, n_vars))(backcast)
    forecast = Dense(1)(x)
    return backcast, forecast

# =====================================================
# 3. ADVANCED MODEL FACTORY (16 ALGORITHMS)
# =====================================================
def get_model(name, lookback, n_vars):
    inputs = Input(shape=(lookback, n_vars))
    
    if name == "ANN":
        x = Flatten()(inputs)
        x = Dense(128, activation="relu")(x)
        x = Dense(64, activation="relu")(x)
        m = Model(inputs, Dense(1)(x))
    elif name == "SimpleRNN":
        x = SimpleRNN(64)(inputs)
        m = Model(inputs, Dense(1)(x))
    elif name == "LSTM":
        x = LSTM(128)(inputs)
        m = Model(inputs, Dense(1)(x))
    elif name == "BiLSTM":
        x = Bidirectional(LSTM(64))(inputs)
        m = Model(inputs, Dense(1)(x))
    elif name == "GRU":
        x = GRU(128)(inputs)
        m = Model(inputs, Dense(1)(x))
    elif name == "CNN-LSTM":
        x = Conv1D(64, 3, activation="relu")(inputs)
        x = MaxPooling1D(2)(x)
        x = LSTM(64)(x)
        m = Model(inputs, Dense(1)(x))
    elif name == "TCN":
        x = Conv1D(64, 3, padding='causal', dilation_rate=1, activation='relu')(inputs)
        x = Conv1D(64, 3, padding='causal', dilation_rate=2, activation='relu')(x)
        x = GlobalAveragePooling1D()(x)
        m = Model(inputs, Dense(1)(x))
    elif name == "iTransformer":
        x = Permute((2, 1))(inputs); x = Dense(128)(x)
        attn = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        x = Add()([x, attn]); x = LayerNormalization()(x); x = Flatten()(x)
        m = Model(inputs, Dense(1)(x))
    elif name == "Liquid-NN":
        x = RNN(LiquidRNNCell(64))(inputs)
        m = Model(inputs, Dense(1)(x))
    elif name == "N-BEATS":
        inp_uni = Input(shape=(lookback, 1)) # Univariate
        b1, f1 = nbeats_block(inp_uni, lookback, 1)
        res1 = Subtract()([inp_uni, b1]); b2, f2 = nbeats_block(res1, lookback, 1)
        m = Model(inp_uni, Add()([f1, f2]))
    elif name == "N-BEATSx":
        b1, f1 = nbeats_block(inputs, lookback, n_vars); res1 = Subtract()([inputs, b1])
        b2, f2 = nbeats_block(res1, lookback, n_vars)
        m = Model(inputs, Add()([f1, f2]))
    elif name == "DLinear":
        trend = tf.keras.layers.AveragePooling1D(5, 1, 'same')(inputs)
        seasonal = Subtract()([inputs, trend])
        t_out = Dense(1)(Flatten()(trend)); s_out = Dense(1)(Flatten()(seasonal))
        m = Model(inputs, Add()([t_out, s_out]))
    elif name == "DeepRes-GRU":
        x = GRU(64, return_sequences=True)(inputs); res = x
        x = GRU(64, return_sequences=True)(x); x = Add()([x, res]); x = GRU(32)(x)
        m = Model(inputs, Dense(1)(x))
    elif name == "Autoformer-MLP":
        trend = tf.keras.layers.AveragePooling1D(5, 1, 'same')(inputs)
        s = Dense(64, activation='relu')(Flatten()(Subtract()([inputs, trend])))
        t = Dense(64, activation='relu')(Flatten()(trend))
        m = Model(inputs, Dense(1)(Add()([s, t])))
    elif name == "LSTNet-Simple":
        x = Conv1D(64, 3, activation='relu')(inputs); x = LSTM(64)(x)
        ar = Flatten()(inputs); ar = Dense(1)(ar)
        m = Model(inputs, Add()([x, ar]))
    elif name == "PatchTST-Lite":
        x = Reshape((lookback // 4, 4 * n_vars))(inputs); x = LSTM(64)(x)
        m = Model(inputs, Dense(1)(x))

    m.compile(loss="mse", optimizer="adam")
    return m

# =====================================================
# 4. EXECUTION & HIGH-PRECISION EVALUATION
# =====================================================
model_names = ["ANN", "SimpleRNN", "LSTM", "BiLSTM", "GRU", "CNN-LSTM", "TCN", "iTransformer", 
               "Liquid-NN", "N-BEATS", "N-BEATSx", "DLinear", "DeepRes-GRU", "Autoformer-MLP", "LSTNet-Simple", "PatchTST-Lite"]

results = []
all_te_preds = {}

for name in model_names:
    print(f"\n📘 Training {name} on GPU Mode...")
    curr_X_train = X_train[:, :, :1] if name == "N-BEATS" else X_train
    curr_X_val = X_val[:, :, :1] if name == "N-BEATS" else X_val
    curr_X_test = X_test[:, :, :1] if name == "N-BEATS" else X_test
    
    model = get_model(name, LOOKBACK, len(features))
    model.fit(curr_X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH, verbose=0)
    
    # Precise Predictions
    tr_preds = model.predict(curr_X_train, verbose=0).flatten()
    te_preds = model.predict(curr_X_test, verbose=0).flatten()
    all_te_preds[name] = te_preds
    
    # Metric Calculation
    r2_tr = r2_score(y_train, tr_preds)
    r2_te = r2_score(y_test, te_preds)
    mse = mean_squared_error(y_test, te_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, te_preds)
    
    # Precision Hit-Rate (10% Threshold)
    hits = np.abs((y_test - te_preds) / (y_test + 1e-7)) < 0.10
    prec = np.sum(hits) / len(y_test)

    results.append({
        "Model": name, "Train_Acc(R2)": r2_tr, "Test_Acc(R2)": r2_te,
        "RMSE": rmse, "MAE": mae, "MSE": mse, "Precision*": prec
    })
    print(f"✅ {name} Evaluation Complete.")

# =====================================================
# 5. FINAL REPORT & DATA EXPORT
# =====================================================
report_df = pd.DataFrame(results)

# 8-Digit Precision Formatting for Display
formatted_df = report_df.copy()
for col in formatted_df.columns[1:]: formatted_df[col] = formatted_df[col].map('{:.8f}'.format)

print("\n" + "="*145)
print("🏆 ULTIMATE RESEARCH PERFORMANCE MATRIX (16 ALGORITHMS)")
print("="*145)
print(formatted_df.sort_values(by="Test_Acc(R2)", ascending=False).to_string(index=False))
print("="*145)

report_df.to_csv("ultimate_high_precision_solar_results.csv", index=False)

# =====================================================
# 6. VISUALIZATION SUITE (COMBINED & SEPARATE)
# =====================================================
metrics = ["Train_Acc(R2)", "Test_Acc(R2)", "RMSE", "MAE", "MSE", "Precision*"]
colors = plt.cm.tab20(np.linspace(0, 1, len(model_names)))

# --- 6.1 Combined Comparison Chart ---
fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 32))
for i, metric in enumerate(metrics):
    axes[i].plot(report_df['Model'], report_df[metric], marker='s', markersize=8, 
                 linewidth=2.5, color='darkblue', label=metric)
    axes[i].set_title(f'Comparative Analysis: {metric}', fontsize=15, fontweight='bold')
    axes[i].set_ylabel(metric, fontsize=12)
    axes[i].grid(True, linestyle='--', alpha=0.6)
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].legend(loc='best')

plt.tight_layout()
plt.savefig('combined_forecasting_analysis.png')
print("📁 Generated: combined_forecasting_analysis.png")

# --- 6.2 Separate Detailed Plots ---
for metric in metrics:
    plt.figure(figsize=(11, 6))
    plt.plot(report_df['Model'], report_df[metric], marker='o', markersize=10, 
             linewidth=3, color='crimson', label=metric)
    plt.title(f'Performance Distribution - {metric}', fontsize=16, fontweight='bold')
    plt.xlabel('Deep Learning Algorithms', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Safe Filename
    clean_name = metric.replace("*", "").replace("(", "").replace(")", "").replace(" ", "_")
    plt.savefig(f'Separate_Plot_{clean_name}.png')
    plt.close()

print("\n🎯 Complete Framework Execution Success. All Reports and 7 Charts Saved.")