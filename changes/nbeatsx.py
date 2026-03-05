import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Add, Concatenate, MultiHeadAttention, LayerNormalization

print("Attentive N-BEATSx Forecasting Started")

# =====================================================
# CONFIG
# =====================================================
DATA_FILE = r"C:\Users\muthumaniraj\OneDrive\Documents\me research\Final _PV_FULL_DATASET\PV_MASTER_15MIN_2024_2025.csv"
TARGET = "Irradiance_Wm2"
EXOGENOUS = ["AmbientTemp_C"]  # Adding temperature as a driver
LOOKBACK = 24  
EPOCHS = 25
BATCH = 64

# =====================================================
# LOAD AND PREPROCESS DATA
# =====================================================
df = pd.read_csv(DATA_FILE)
df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
df = df.sort_values("Datetime")

# Selecting target and exogenous features
features = [TARGET] + EXOGENOUS
data = df[features].values

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

def make_multivariate_sequences(data, step):
    X, y = [], []
    for i in range(len(data) - step):
        X.append(data[i:i+step, :]) # All features for lookback
        y.append(data[i+step, 0])    # Only Irradiance for target
    return np.array(X), np.array(y)

X, y = make_multivariate_sequences(data_scaled, LOOKBACK)

# Split 70/10/20
train_split = int(0.7 * len(X))
val_split = int(0.8 * len(X))

X_train, y_train = X[:train_split], y[:train_split]
X_val, y_val = X[train_split:val_split], y[train_split:val_split]
X_test, y_test = X[val_split:], y[val_split:]

# =====================================================
# ATTENTIVE N-BEATSx ARCHITECTURE
# =====================================================

def nbeats_block(input_layer, lookback, num_features):
    # 1. Attention Mechanism (The Improvement)
    # Allows the block to weight different time steps differently
    attn = MultiHeadAttention(num_heads=4, key_dim=32)(input_layer, input_layer)
    attn = LayerNormalization()(attn)
    x = Flatten()(attn)
    
    # 2. Dense Expansion
    d1 = Dense(256, activation='relu')(x)
    d2 = Dense(256, activation='relu')(d1)
    d3 = Dense(256, activation='relu')(d2)
    
    # 3. Forecast/Backcast branches
    # Backcast predicts the input window to calculate residual
    backcast = Dense(lookback * num_features)(d3)
    backcast = Reshape((lookback, num_features))(backcast)
    
    # Forecast predicts the next step
    forecast = Dense(1)(d3)
    
    return backcast, forecast

def Attentive_NBEATSx_Model(lookback, num_features):
    inputs = Input(shape=(lookback, num_features))
    
    # Stack 1 (Trend/Global Patterns)
    backcast1, forecast1 = nbeats_block(inputs, lookback, num_features)
    res1 = Add()([inputs, -backcast1])
    
    # Stack 2 (Seasonality/Local Fluctuations)
    backcast2, forecast2 = nbeats_block(res1, lookback, num_features)
    
    # Final Output is the sum of both forecasts
    output = Add()([forecast1, forecast2])
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss="mse", optimizer="adam")
    return model

# =====================================================
# TRAINING AND EVALUATION
# =====================================================

model = Attentive_NBEATSx_Model(LOOKBACK, len(features))
print("Model Summary: Attentive N-BEATSx with Exogenous Inputs")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH,
    verbose=1
)

# Generate Predictions
train_preds = model.predict(X_train)
val_preds = model.predict(X_val)
test_preds = model.predict(X_test)

# Calculate Evaluation Matrix
report = {
    "Metric": ["Loss (MSE)", "Accuracy (R2)"],
    "Train": [history.history['loss'][-1], r2_score(y_train, train_preds)],
    "Val": [history.history['val_loss'][-1], r2_score(y_val, val_preds)],
    "Test": [mean_squared_error(y_test, test_preds), r2_score(y_test, test_preds)]
}

report_df = pd.DataFrame(report)
print("\nDETAILED ATTENTIVE N-BEATSx REPORT")
print(report_df.round(6).to_string(index=False))

# Save
report_df.to_csv("attentive_nbeatsx_report.csv", index=False)