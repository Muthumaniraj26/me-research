import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, Dense, LSTM, GRU, Conv1D, Flatten, Reshape, 
                                     Add, Subtract, Multiply, RNN, Bidirectional, 
                                     Dropout, MultiHeadAttention, LayerNormalization, 
                                     Permute, GlobalAveragePooling1D, MaxPooling1D)

print("🚀 2026 Solar Research Framework: Standard N-BEATS & High-Precision Metrics")

# =====================================================
# 1. CONFIGURATION & DATA LOADING
# =====================================================
DATA_FILE = r"C:\Users\muthumaniraj\OneDrive\Documents\me research\Final _PV_FULL_DATASET\PV_MASTER_15MIN_2024_2025.csv"
TARGET = "Irradiance_Wm2"
EXOGENOUS = ["AmbientTemp_C", "CellTemp_C"] 
LOOKBACK = 24  
EPOCHS = 25
BATCH = 64

df = pd.read_csv(DATA_FILE)
df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
df = df.sort_values("Datetime")

features = [TARGET] + EXOGENOUS
data = df[features].values
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

def make_sequences(data, step):
    X, y = [], []
    for i in range(len(data) - step):
        X.append(data[i:i+step, :])
        y.append(data[i+step, 0])
    return np.array(X), np.array(y)

X, y = make_sequences(data_scaled, LOOKBACK)
train_split, val_split = int(0.7 * len(X)), int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_val, y_val = X[train_split:val_split], y[train_split:val_split]
X_test, y_test = X[val_split:], y[val_split:]

# =====================================================
# 2. ARCHITECTURE DEFINITIONS
# =====================================================

# --- Standard N-BEATS (Univariate Focus) ---
def nbeats_block(input_layer, lookback, units=256):
    x = Flatten()(input_layer)
    x = Dense(units, activation='relu')(x)
    x = Dense(units, activation='relu')(x)
    # Backcast (Past reconstruction)
    backcast = Dense(lookback)(x)
    backcast = Reshape((lookback, 1))(backcast)
    # Forecast (Future prediction)
    forecast = Dense(1)(x)
    return backcast, forecast

def NBEATS_Standard(lookback):
    inputs = Input(shape=(lookback, 1)) # Only takes the Target variable
    # Block 1
    b1, f1 = nbeats_block(inputs, lookback)
    res1 = Subtract()([inputs, b1])
    # Block 2
    b2, f2 = nbeats_block(res1, lookback)
    output = Add()([f1, f2])
    model = Model(inputs, output)
    model.compile(loss="mse", optimizer="adam")
    return model

# --- Liquid Neural Network Cell ---
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

# =====================================================
# 3. MODEL FACTORY
# =====================================================

def get_model(name, lookback, n_vars):
    if name == "ANN":
        m = Sequential([Dense(64, activation="relu", input_shape=(lookback, n_vars)), Flatten(), Dense(32, activation="relu"), Dense(1)])
    elif name == "LSTM":
        m = Sequential([LSTM(64, input_shape=(lookback, n_vars)), Dense(1)])
    elif name == "BiLSTM":
        m = Sequential([Bidirectional(LSTM(64), input_shape=(lookback, n_vars)), Dense(1)])
    elif name == "GRU":
        m = Sequential([GRU(64, input_shape=(lookback, n_vars)), Dense(1)])
    elif name == "CNN-LSTM":
        m = Sequential([Conv1D(64, 3, activation="relu", input_shape=(lookback, n_vars)), MaxPooling1D(2), LSTM(50), Dense(1)])
    elif name == "iTransformer":
        inputs = Input(shape=(lookback, n_vars))
        x = Permute((2, 1))(inputs); x = Dense(128)(x)
        attn = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        x = Add()([x, attn]); x = LayerNormalization()(x); x = Flatten()(x)
        m = Model(inputs, Dense(1)(x))
    elif name == "Liquid-NN":
        inputs = Input(shape=(lookback, n_vars))
        x = RNN(LiquidRNNCell(64))(inputs)
        m = Model(inputs, Dense(1)(x))
    elif name == "N-BEATS":
        return NBEATS_Standard(lookback) # Univariate
    
    m.compile(loss="mse", optimizer="adam")
    return m
    # --- N-BEATSx Block (Found in Section 3 of the code) ---
def nbeatsx_block(input_layer, lookback, n_vars):
    x = Flatten()(input_layer)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    
    # Backcast: What the model thinks happened in the LOOKBACK period
    backcast = Dense(lookback * n_vars)(x)
    backcast = Reshape((lookback, n_vars))(backcast)
    
    # Forecast: The actual prediction for the next step
    forecast = Dense(1)(x)
    return backcast, forecast
# =====================================================
# 4. EXECUTION & EVALUATION
# =====================================================

model_names = ["ANN", "LSTM", "BiLSTM", "GRU", "CNN-LSTM", "iTransformer", "Liquid-NN", "N-BEATS"]
results = []
prediction_data = {}

for name in model_names:
    print(f"\n📘 Training {name}...")
    
    # N-BEATS only uses the target column [:, :, 0]
    curr_X_train = X_train[:, :, :1] if name == "N-BEATS" else X_train
    curr_X_val = X_val[:, :, :1] if name == "N-BEATS" else X_val
    curr_X_test = X_test[:, :, :1] if name == "N-BEATS" else X_test
    
    model = get_model(name, LOOKBACK, len(features))
    model.fit(curr_X_train, y_train, validation_data=(curr_X_val, y_val), epochs=EPOCHS, batch_size=BATCH, verbose=0)
    
    # 8-Digit Precision Metrics
    tr_preds = model.predict(curr_X_train, verbose=0)
    te_preds = model.predict(curr_X_test, verbose=0)
    
    r2_tr = r2_score(y_train, tr_preds)
    r2_te = r2_score(y_test, te_preds)
    mse = mean_squared_error(y_test, te_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, te_preds)
    
    # Hit Rate (10% Threshold)
    hits = np.abs((y_test - te_preds.flatten()) / (y_test + 1e-7)) < 0.10
    prec = np.sum(hits) / len(hits)

    results.append({
        "Model": name,
        "Train_Acc(R2)": f"{r2_tr:.8f}",
        "Test_Acc(R2)": f"{r2_te:.8f}",
        "RMSE": f"{rmse:.8f}",
        "MAE": f"{mae:.8f}",
        "MSE": f"{mse:.8f}",
        "Precision*": f"{prec:.8f}"
    })
    
    # Store predictions for visualization
    if "Actual" not in prediction_data:
        prediction_data["Actual"] = y_test
    
    prediction_data[name] = te_preds.flatten()
    
    print(f"✅ {name} Done")

# =====================================================
# 5. FINAL REPORT & SAVING
# =====================================================
# Save Predictions
pred_df = pd.DataFrame(prediction_data)
pred_df.to_csv("model_predictions.csv", index=False)
print("📁 Saved: model_predictions.csv")

# =====================================================
# 5. FINAL REPORT
# =====================================================
report_df = pd.DataFrame(results)
print("\n" + "="*145)
print("🏆 ULTIMATE RESEARCH PERFORMANCE MATRIX (8-DIGIT PRECISION)")
print("="*145)
print(report_df.sort_values(by="Test_Acc(R2)", ascending=False).to_string(index=False))
print("="*145)

report_df.to_csv("high_precision_solar_final.csv", index=False)