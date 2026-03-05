import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, Dense, LSTM, GRU, Conv1D, Flatten, 
                                     Bidirectional, Dropout, Multiply, Add, 
                                     Concatenate, GlobalAveragePooling1D, MaxPooling1D,
                                     LayerNormalization, Activation)

# =====================================================
# 0. REPRODUCIBILITY BLOCK
# =====================================================
def set_seeds(seed=42):
    """Locks all random generators for identical results every run."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

set_seeds(42)
print("🚀 Reproducibility Locked (Seed=42). Starting Research Pipeline...")

# =====================================================
# 1. CONFIGURATION
# =====================================================
DATA_FILE = r"C:\Users\muthumaniraj\OneDrive\Documents\me research\Final _PV_FULL_DATASET\PV_MASTER_15MIN_2024_2025.csv"
TARGET = "Irradiance_Wm2"
EXOGENOUS = ["AmbientTemp_C", "CellTemp_C"] 
LOOKBACK = 24  
EPOCHS = 120
BATCH = 64

# Load Data
df = pd.read_csv(DATA_FILE)
df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
df = df.sort_values("Datetime")

features = [TARGET] + EXOGENOUS
data_scaled = MinMaxScaler().fit_transform(df[features].values)

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
# 2. ALGORITHM REPOSITORY (EXISTING & PROPOSED)
# =====================================================

def LSTM_Model(lb, nv):
    return Sequential([
        LSTM(64, return_sequences=True, input_shape=(lb, nv)), 
        LSTM(32), 
        Dense(1)
    ], name="LSTM_Baseline")

def GRU_Model(lb, nv):
    return Sequential([
        GRU(64, return_sequences=True, input_shape=(lb, nv)), 
        GRU(32), 
        Dense(1)
    ], name="GRU_Baseline")

def BiLSTM_Model(lb, nv):
    return Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(lb, nv)), 
        Bidirectional(LSTM(32)), 
        Dense(1)
    ], name="BiLSTM_Baseline")

def CNN_LSTM_Hybrid(lb, nv):
    return Sequential([
        Conv1D(64, 3, activation="relu", input_shape=(lb, nv)), 
        MaxPooling1D(2), 
        LSTM(50), 
        Dense(1)
    ], name="CNN_LSTM_Hybrid")

def LSTM_GRU_Combo(lb, nv):
    inputs = Input(shape=(lb, nv))
    p1 = LSTM(64)(inputs)
    p2 = GRU(64)(inputs)
    merged = Concatenate()([p1, p2])
    outputs = Dense(1)(Dense(32, activation='relu')(merged))
    model = Model(inputs, outputs, name="LSTM_GRU_Combo")
    return model

def BiLSTM_GRU_Combo(lb, nv):
    inputs = Input(shape=(lb, nv))
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = GRU(32)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs, name="BiLSTM_GRU_Combo")
    return model

def Gated_Residual_RNN(lb, nv):
    """CHAMPION MODEL: Gated Residual Sequential Network"""
    inputs = Input(shape=(lb, nv))
    # Recurrent Path
    x = Bidirectional(GRU(64, return_sequences=True))(inputs)
    # Residual Identity Path
    res = Conv1D(128, 1)(inputs) 
    x = Add()([x, res])
    x = LayerNormalization()(x)
    # Gating Mechanism (Adaptive Noise Filter)
    gate = Dense(128, activation='sigmoid')(x)
    x = Multiply()([x, gate])
    # Final Refinement
    x = GRU(64, return_sequences=False)(x)
    x = Activation('gelu')(Dense(64)(x))
    outputs = Dense(1)(x)
    model = Model(inputs, outputs, name="Gated_Residual_RNN_Proposed")
    # AdamW used for Champion's stability
    model.compile(loss="huber", optimizer=tf.keras.optimizers.AdamW(1e-3))
    return model

# =====================================================
# 3. UNIFIED TRAINING & EVALUATION
# =====================================================
n_vars = len(features)
rnn_models = {
    "LSTM": LSTM_Model(LOOKBACK, n_vars),
    "GRU": GRU_Model(LOOKBACK, n_vars),
    "BiLSTM": BiLSTM_Model(LOOKBACK, n_vars),
    "CNN-LSTM": CNN_LSTM_Hybrid(LOOKBACK, n_vars),
    "LSTM-GRU": LSTM_GRU_Combo(LOOKBACK, n_vars),
    "BiLSTM-GRU": BiLSTM_GRU_Combo(LOOKBACK, n_vars),
    "Gated-Residual-RNN": Gated_Residual_RNN(LOOKBACK, n_vars)
}

final_results = []

for name, model in rnn_models.items():
    print(f"\n📘 Training Architecture: {name}...")
    # Compile baselines with standard settings
    if "Gated" not in name:
        model.compile(loss="mse", optimizer="adam")
    
    es = tf.keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True)
    
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH, 
              verbose=0, validation_split=0.1, callbacks=[es])
    
    # Prediction and Metrics
    y_pred = model.predict(X_test, verbose=0).flatten()
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    # Precision within 10% error margin
    precision = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-7)) < 0.10)

    final_results.append({
        "Model": name, "R2": round(r2, 8), "RMSE": round(rmse, 8), 
        "MAE": round(mae, 8), "MSE": round(mean_squared_error(y_test, y_pred), 10), 
        "Precision*": round(precision, 8)
    })
    print(f"✅ {name} Done.")

# =====================================================
# 4. FINAL RESULTS REPORT
# =====================================================
report_df = pd.DataFrame(final_results).sort_values(by="R2", ascending=False)
print("\n" + "="*95)
print("🏁 FINAL CONSOLIDATED PERFORMANCE MATRIX")
print("="*95)
print(report_df.to_string(index=False))
print("="*95)

report_df.to_csv("Final_Stable_Results.csv", index=False)