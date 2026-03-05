import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Flatten, Add, LayerNormalization, 
                                     Conv1D, LSTM, GRU, Bidirectional, Dropout, 
                                     Concatenate, GlobalAveragePooling1D)

# =====================================================
# 1. DATA PREPROCESSING
# =====================================================
DATA_FILE = r"C:\Users\muthumaniraj\OneDrive\Documents\me research\Final _PV_FULL_DATASET\PV_MASTER_15MIN_2024_2025.csv"
TARGET = "Irradiance_Wm2"
EXOGENOUS = ["AmbientTemp_C", "CellTemp_C"] 
LOOKBACK = 24  
EPOCHS = 20
BATCH = 64

# Loading data
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
train_split = int(0.7 * len(X))
val_split = int(0.8 * len(X))

X_train, y_train = X[:train_split], y[:train_split]
X_val, y_val = X[train_split:val_split], y[train_split:val_split]
X_test, y_test = X[val_split:], y[val_split:]

# =====================================================
# 2. 2025/26 ALGORITHM ARCHITECTURES
# =====================================================

# 2.1 Bi-GRU Enhanced DL (Feb 2025 Trend)
def BiGRU_Hybrid_Model(lookback, n_vars):
    inputs = Input(shape=(lookback, n_vars))
    x = Bidirectional(GRU(64, return_sequences=True))(inputs)
    x = LSTM(64)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(loss="mse", optimizer="adam")
    return model

# 2.2 TCN - Temporal Convolutional Network (Aug 2025 Trend)
def TCN_Model(lookback, n_vars):
    inputs = Input(shape=(lookback, n_vars))
    # Dilated Causal Convolutions for long-term memory
    x = Conv1D(filters=64, kernel_size=3, padding='causal', dilation_rate=1, activation='relu')(inputs)
    x = Conv1D(filters=64, kernel_size=3, padding='causal', dilation_rate=2, activation='relu')(x)
    x = Conv1D(filters=64, kernel_size=3, padding='causal', dilation_rate=4, activation='relu')(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(loss="mse", optimizer="adam")
    return model

# 2.3 CNN-LSTM Spatial-Temporal (Nov 2025 Trend)
def CNN_LSTM_Model(lookback, n_vars):
    inputs = Input(shape=(lookback, n_vars))
    # Spatial feature extraction via CNN
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    # Temporal modeling via LSTM
    x = LSTM(64, return_sequences=False)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(loss="mse", optimizer="adam")
    return model

# =====================================================
# 3. EVALUATION SUITE
# =====================================================
models_to_run = {
    "Bi-GRU Hybrid": BiGRU_Hybrid_Model(LOOKBACK, len(features)),
    "TCN (Causal Conv)": TCN_Model(LOOKBACK, len(features)),
    "CNN-LSTM": CNN_LSTM_Model(LOOKBACK, len(features))
}

results = []

for name, model in models_to_run.items():
    print(f"\n--- Training {name} ---")
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH, verbose=0)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Regression Metrics
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    
    # Classification-Style Metrics (Accuracy logic)
    # Correct if within 10% relative error
    hits = np.abs((y_test - y_pred_test.flatten()) / (y_test + 1e-7)) < 0.10
    precision = np.sum(hits) / len(hits)
    recall = precision # In this evaluation context
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

    results.append({
        "Model": name,
        "Train_R2": round(r2_train, 6),
        "Test_R2": round(r2_test, 6),
        "MSE": round(mse, 6),
        "RMSE": round(rmse, 6),
        "MAE": round(mae, 6),
        "Precision*": round(precision, 6),
        "Recall*": round(recall, 4),
        "F1_Score*": round(f1, 4)
    })

# Final Matrix
report_df = pd.DataFrame(results)
print("\n" + "="*90)
print("ALGORITHM PERFORMANCE MATRIX (2025-2026 ARCHITECTURES)")
print("="*90)
print(report_df.to_string(index=False))
print("="*90)