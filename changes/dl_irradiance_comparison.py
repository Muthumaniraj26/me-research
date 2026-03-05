import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D, Flatten

print("🚀 Deep Learning Irradiance Forecast Comparison Started")

# =====================================================
# CONFIG
# =====================================================

DATA_FILE = r"C:\Users\muthumaniraj\OneDrive\Documents\me research\Final _PV_FULL_DATASET\PV_MASTER_15MIN_2024_2025.csv"   # Your master file
TARGET = "Irradiance_Wm2"

LOOKBACK = 24        # 24 steps = 6 hours (15min × 24)
EPOCHS = 25
BATCH = 64

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(DATA_FILE)

print("✅ Data Loaded:", df.shape)

# Sort time
df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
df = df.sort_values("Datetime")

# Keep only numeric
data = df[[TARGET]].values

# =====================================================
# NORMALIZE
# =====================================================

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# =====================================================
# CREATE SEQUENCES
# =====================================================

def make_sequences(data, step):
    X, y = [], []

    for i in range(len(data) - step):
        X.append(data[i:i+step])
        y.append(data[i+step])

    return np.array(X), np.array(y)


X, y = make_sequences(data_scaled, LOOKBACK)

print("✅ Sequences:", X.shape)

# =====================================================
# TRAIN TEST SPLIT (80/20)
# =====================================================

split = int(0.8 * len(X))

X_train = X[:split]
X_test  = X[split:]

y_train = y[:split]
y_test  = y[split:]

# =====================================================
# METRICS
# =====================================================

def evaluate(y_true, y_pred):

    y_true = scaler.inverse_transform(y_true)
    y_pred = scaler.inverse_transform(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)

    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    r2 = r2_score(y_true, y_pred)

    return rmse, mae, mape, r2, mse


# =====================================================
# MODEL DEFINITIONS
# =====================================================

# ANN
def ANN():

    model = Sequential([
        Dense(64, activation="relu", input_shape=(LOOKBACK,1)),
        Flatten(),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(loss="mse", optimizer="adam")

    return model


# LSTM
def LSTM_Model():

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LOOKBACK,1)),
        LSTM(32),
        Dense(1)
    ])

    model.compile(loss="mse", optimizer="adam")

    return model


# BiLSTM
def BiLSTM_Model():

    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(LOOKBACK,1)),
        Bidirectional(LSTM(32)),
        Dense(1)
    ])

    model.compile(loss="mse", optimizer="adam")

    return model


# GRU
def GRU_Model():

    model = Sequential([
        GRU(64, return_sequences=True, input_shape=(LOOKBACK,1)),
        GRU(32),
        Dense(1)
    ])

    model.compile(loss="mse", optimizer="adam")

    return model


# CNN-LSTM
def CNN_LSTM():

    model = Sequential([
        Conv1D(64, 3, activation="relu", input_shape=(LOOKBACK,1)),
        MaxPooling1D(2),

        LSTM(50),

        Dense(1)
    ])

    model.compile(loss="mse", optimizer="adam")

    return model


# =====================================================
# TRAIN & EVALUATE
# =====================================================

models = {
    "ANN": ANN(),
    "LSTM": LSTM_Model(),
    "BiLSTM": BiLSTM_Model(),
    "GRU": GRU_Model(),
    "CNN-LSTM": CNN_LSTM()
}


results = []


for name, model in models.items():

    print(f"\n📘 Training {name}...")

    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH,
        verbose=0
    )

    preds = model.predict(X_test)

    rmse, mae, mape, r2, mse = evaluate(y_test, preds)

    results.append([
        name,
        rmse,
        mae,
        mape,
        r2,
        mse
    ])

    print(f"✅ {name} Done")


# =====================================================
# RESULT TABLE
# =====================================================

result_df = pd.DataFrame(results, columns=[
    "Model","RMSE","MAE","MAPE(%)","R2","MSE"
])


print("\n📊 MODEL COMPARISON (IRRADIANCE TARGET)")
print(result_df.round(4))


# Save
result_df.to_csv("dl_model_comparison_irradiance.csv", index=False)

print("\n📁 Saved: dl_model_comparison_irradiance.csv")
print("🎯 Analysis Complete")
