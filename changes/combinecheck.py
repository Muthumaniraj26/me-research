import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, Dense, LSTM, GRU, Conv1D, Flatten, Reshape, 
                                     Add, Multiply, RNN, Bidirectional, Dropout, 
                                     MultiHeadAttention, LayerNormalization, 
                                     Permute, GlobalAveragePooling1D, MaxPooling1D, Concatenate)

print("📊 Initializing Deep Temporal-Relational Hybrid (DTRH) Research Framework")

# =====================================================
# 1. CONFIGURATION & DATA PREPARATION
# =====================================================
DATA_FILE = r"C:\Users\muthumaniraj\OneDrive\Documents\me research\Final _PV_FULL_DATASET\PV_MASTER_15MIN_2024_2025.csv"
TARGET = "Irradiance_Wm2"
EXOGENOUS = ["AmbientTemp_C", "CellTemp_C"] 
LOOKBACK = 24  
EPOCHS = 100
BATCH = 64

# Load and sort data
df = pd.read_csv(DATA_FILE)
df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
df = df.sort_values("Datetime")

features = [TARGET] + EXOGENOUS
data = df[features].values
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

def create_windowed_dataset(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size, :])
        y.append(data[i+window_size, 0])
    return np.array(X), np.array(y)

X, y = create_windowed_dataset(data_scaled, LOOKBACK)
train_split, val_split = int(0.7 * len(X)), int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_val, y_val = X[train_split:val_split], y[train_split:val_split]
X_test, y_test = X[val_split:], y[val_split:]

# =====================================================
# 2. PROPOSED CHAMPION: DEEP TEMPORAL-RELATIONAL HYBRID (DTRH)
# =====================================================

def DTRH_Model(lookback, n_vars):
    """
    Novel Architecture: Deep Temporal-Relational Hybrid.
    Integrates Dilated Multi-Scale Feature Fusion and Inverted Multi-Head Attention.
    """
    inputs = Input(shape=(lookback, n_vars))
    
    # Stage 1: Multi-Scale Spatio-Temporal Extraction
    # Local pattern extraction + dilated long-range dependency
    cnn_l = Conv1D(64, 3, padding='same', activation='relu')(inputs)
    cnn_d = Conv1D(64, 3, padding='same', dilation_rate=2, activation='relu')(inputs)
    cnn_fusion = Concatenate()([cnn_l, cnn_d])
    cnn_fusion = LayerNormalization()(cnn_fusion)
    
    # Stage 2: Inverted Relational Attention (iTransformer Logic)
    # Transposing dimensions to treat physical variables as tokens
    relational_map = Permute((2, 1))(cnn_fusion) 
    relational_map = Dense(128)(relational_map)
    attn_out = MultiHeadAttention(num_heads=8, key_dim=16)(relational_map, relational_map)
    relational_map = Add()([relational_map, attn_out])
    relational_map = LayerNormalization()(relational_map)
    
    # Stage 3: Adaptive Gating (Liquid Dynamics)
    gate = Dense(128, activation='sigmoid')(relational_map)
    gated_features = Multiply()([relational_map, gate])
    
    # Stage 4: Bidirectional Sequential Processing
    sequential_flow = Bidirectional(GRU(64, return_sequences=False))(gated_features)
    
    # Stage 5: High-Dimensional Nonlinear Projection
    x = Dense(256, activation='gelu')(sequential_flow)
    x = Dropout(0.1)(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs, outputs, name="DTRH_Hybrid")
    model.compile(loss="huber", optimizer=tf.keras.optimizers.AdamW(1e-3))
    return model

# =====================================================
# 3. BASELINE & SOTA ARCHITECTURES
# =====================================================

def ANN_Model(lookback, n_vars):
    return Sequential([
        Dense(64, activation="relu", input_shape=(lookback, n_vars)),
        Flatten(), Dense(32, activation="relu"), Dense(1)
    ], name="ANN")

def LSTM_Model(lookback, n_vars):
    return Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, n_vars)),
        LSTM(32), Dense(1)
    ], name="LSTM")

def BiLSTM_Model(lookback, n_vars):
    return Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(lookback, n_vars)),
        Bidirectional(LSTM(32)), Dense(1)
    ], name="BiLSTM")

def GRU_Model(lookback, n_vars):
    return Sequential([
        GRU(64, return_sequences=True, input_shape=(lookback, n_vars)),
        GRU(32), Dense(1)
    ], name="GRU")

def CNN_LSTM_Hybrid(lookback, n_vars):
    return Sequential([
        Conv1D(64, 3, activation="relu", input_shape=(lookback, n_vars)),
        MaxPooling1D(2), LSTM(50), Dense(1)
    ], name="CNN_LSTM")

def NBEATS_Model(lookback, n_vars):
    inputs = Input(shape=(lookback, n_vars))
    x = Flatten()(inputs)
    d1 = Dense(256, activation='relu')(x)
    f1 = Dense(1)(d1)
    res = Add()([x, -Dense(lookback * n_vars)(d1)])
    d2 = Dense(256, activation='relu')(res)
    f2 = Dense(1)(d2)
    outputs = Add()([f1, f2])
    return Model(inputs, outputs, name="NBEATS")

# =====================================================
# 4. TRAINING & METRIC CONSOLIDATION
# =====================================================

# Initialize Model Compilation
n_features = len(features)
baseline_models = {
    "ANN": ANN_Model(LOOKBACK, n_features),
    "LSTM": LSTM_Model(LOOKBACK, n_features),
    "BiLSTM": BiLSTM_Model(LOOKBACK, n_features),
    "GRU": GRU_Model(LOOKBACK, n_features),
    "CNN-LSTM": CNN_LSTM_Hybrid(LOOKBACK, n_features),
    "N-BEATS": NBEATS_Model(LOOKBACK, n_features),
    "DTRH (Proposed)": DTRH_Model(LOOKBACK, n_features)
}

final_results = []

for name, model in baseline_models.items():
    print(f"\n▶️ Training Architecture: {name}")
    model.compile(loss="mse", optimizer="adam")
    
    # Early stopping for scientific stability
    es = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    
    model.fit(X_train, y_train, validation_data=(X_val, y_val), 
              epochs=EPOCHS, batch_size=BATCH, verbose=0, callbacks=[es])
    
    # Evaluation
    y_p_test = model.predict(X_test, verbose=0)
    y_p_train = model.predict(X_train, verbose=0)
    
    # Metrics calculation
    mse = mean_squared_error(y_test, y_p_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_p_test)
    r2_te = r2_score(y_test, y_p_test)
    r2_tr = r2_score(y_train, y_p_train)
    
    # Hit Rate / Precision (Within 10% Margin)
    precision = np.mean(np.abs((y_test - y_p_test.flatten()) / (y_test + 1e-7)) < 0.10)

    final_results.append({
        "Model": name,
        "Train_Acc(R2)": round(r2_tr, 8),
        "Test_Acc(R2)": round(r2_te, 8),
        "RMSE": round(rmse, 8),
        "MAE": round(mae, 8),
        "MSE": round(mse, 10),
        "Precision*": round(precision, 8)
    })
    print(f"✔️ {name} Evaluation Complete.")

# =====================================================
# 5. RESEARCH OUTPUT GENERATION
# =====================================================
result_df = pd.DataFrame(final_results)
result_df = result_df.sort_values(by="Test_Acc(R2)", ascending=False)

print("\n" + "="*100)
print("📈 CONSOLIDATED EXPERIMENTAL PERFORMANCE MATRIX")
print("="*100)
print(result_df.to_string(index=False))
print("="*100)
print("*Precision* denotes the hit-rate of forecasts within a 10% relative error threshold.")

# Save to Excel/CSV for publication use
result_df.to_csv("Research_Performance_Matrix_2026.csv", index=False)