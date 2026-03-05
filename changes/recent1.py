import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Add, LayerNormalization, MultiHeadAttention, Permute

print("Deep Learning PV Forecasting Framework (2025-2026 Tier) - Full Metrics Suite")

# =====================================================
# 1. CONFIGURATION & DATA LOADING
# =====================================================
DATA_FILE = r"C:\Users\muthumaniraj\OneDrive\Documents\me research\Final _PV_FULL_DATASET\PV_MASTER_15MIN_2024_2025.csv"
TARGET = "Irradiance_Wm2"
EXOGENOUS = ["AmbientTemp_C", "CellTemp_C"] 
LOOKBACK = 24  
EPOCHS = 20
BATCH = 64

# Load and preprocess
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

# Splits: 70% Train, 10% Val, 20% Test
train_split = int(0.7 * len(X))
val_split = int(0.8 * len(X))

X_train, y_train = X[:train_split], y[:train_split]
X_val, y_val = X[train_split:val_split], y[train_split:val_split]
X_test, y_test = X[val_split:], y[val_split:]

# =====================================================
# 2. MODEL ARCHITECTURES
# =====================================================

def iTransformer_Model(lookback, n_vars):
    inputs = Input(shape=(lookback, n_vars))
    # Inversion: Permute to (Variables, Lookback)
    x = Permute((2, 1))(inputs) 
    x = Dense(128)(x)
    attn_out = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = Add()([x, attn_out])
    x = LayerNormalization()(x)
    x = Flatten()(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(loss="mse", optimizer="adam")
    return model

def Chronos_Inspired_Model(lookback, n_vars):
    inputs = Input(shape=(lookback, n_vars))
    x = Flatten()(inputs)
    for _ in range(3):
        res = x
        x = Dense(512, activation='swish')(x)
        x = Dense(lookback * n_vars)(x)
        x = Add()([res, x]) 
        x = LayerNormalization()(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(loss="mse", optimizer="adam")
    return model

# =====================================================
# 3. EXECUTION AND COMPREHENSIVE EVALUATION
# =====================================================
models_dict = {
    "iTransformer": iTransformer_Model(LOOKBACK, len(features)),
    "Chronos-Inspired": Chronos_Inspired_Model(LOOKBACK, len(features))
}

final_results = []

for name, model in models_dict.items():
    print(f"\n>>> Training {name}...")
    
    # Train
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val),
        epochs=EPOCHS, 
        batch_size=BATCH, 
        verbose=1
    )
    
    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # --- REGRESSION METRICS ---
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    
    # --- CLASSIFICATION-STYLE METRICS (Hit Rate Logic) ---
    # We define a 'Hit' as a prediction within 10% error of the actual value
    threshold = 0.10 
    # Use a small epsilon to avoid division by zero
    relative_error = np.abs((y_test - y_pred_test.flatten()) / (y_test + 1e-7))
    hits = relative_error < threshold
    
    precision = np.sum(hits) / len(hits) # Ratio of accurate forecasts
    recall = precision # In this context, Precision = Recall
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

    final_results.append({
        "Model": name,
        "Train_R2": round(r2_train, 6),
        "Test_R2": round(r2_test, 6),
        "MSE": round(mse, 6),
        "RMSE": round(rmse, 6),
        "MAE": round(mae, 6),
        "Precision*": round(precision, 6),
        "Recall*": round(recall, 6),
        "F1_Score*": round(f1, 6)
    })

# =====================================================
# 4. FINAL REPORT
# =====================================================
report_df = pd.DataFrame(final_results)

print("\n" + "="*80)
print("PV FORECASTING PERFORMANCE REPORT (2025)")
print("="*80)
print(report_df.to_string(index=False))
print("="*80)
print("*Note: Precision/Recall/F1 are calculated based on a 10% error tolerance.")

# Optional: Export to CSV
# report_df.to_csv("Model_Evaluation_Results.csv", index=False)