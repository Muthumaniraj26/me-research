# =====================================================
# PV TIME-SERIES MODEL COMPARISON PIPELINE
# ARIMA + XGBoost + LSTM + CNN-LSTM
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten

# =====================================================
# CONFIG
# =====================================================

DATA_FILE = r"C:\Users\muthumaniraj\OneDrive\Documents\me research\Final _PV_FULL_DATASET\PV_MASTER_15MIN_2024_2025.csv"
TARGET = "Irradiance_Wm2"

TEST_RATIO = 0.2
SEQ_LEN = 24     # 24 timesteps (6 hours for 15min data)

import os

OUTPUT_DIR = "output_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULT_FILE = os.path.join(OUTPUT_DIR, "model_comparison.csv")
PLOT_FILE = os.path.join(OUTPUT_DIR, "model_comparison.png")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
# =====================================================
# METRICS
# =====================================================

def calc_metrics(y_true, y_pred):

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)

    return rmse, mae, mape, r2


# =====================================================
# LOAD DATA
# =====================================================

print("📂 Loading dataset...")

df = pd.read_csv(DATA_FILE)

df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
df = df.sort_values("Datetime")

df = df.dropna()

print("Rows:", len(df))


# =====================================================
# FEATURES
# =====================================================

FEATURES = [
    "Irradiance_Wm2",
    "AmbientTemp_C",
    "CellTemp_C",
    "Voc_V",
    "Isc_A",
    "Vm_V",
    "Im_A",
    "FF"
]

X = df[FEATURES].values
y = df[TARGET].values.reshape(-1,1)


# =====================================================
# NORMALIZATION
# =====================================================

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)


# =====================================================
# SEQUENCE CREATION
# =====================================================

def make_sequences(X, y, seq_len):

    Xs, ys = [], []

    for i in range(len(X)-seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])

    return np.array(Xs), np.array(ys)


X_seq, y_seq = make_sequences(X, y, SEQ_LEN)


# =====================================================
# TRAIN / TEST SPLIT
# =====================================================

split = int(len(X_seq)*(1-TEST_RATIO))

X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

print("Train:", X_train.shape)
print("Test :", X_test.shape)


# =====================================================
# RESULTS STORAGE
# =====================================================

results = []


# =====================================================
# 1. ARIMA (Univariate Baseline)
# =====================================================

print("🚀 Training ARIMA...")

y_arima = df[TARGET].values

train_arima = y_arima[:int(len(y_arima)*(1-TEST_RATIO))]
test_arima = y_arima[int(len(y_arima)*(1-TEST_RATIO)):]

model = ARIMA(train_arima, order=(5,1,0))
model_fit = model.fit()

pred_arima = model_fit.forecast(len(test_arima))

rmse, mae, mape, r2 = calc_metrics(test_arima, pred_arima)

results.append(["ARIMA", rmse, mae, mape, r2])


# =====================================================
# 2. XGBOOST
# =====================================================

print("🚀 Training XGBoost...")

Xgb_train = X_train.reshape(X_train.shape[0], -1)
Xgb_test = X_test.reshape(X_test.shape[0], -1)

y_train_xgb = y_train
y_test_xgb = y_test

model_xgb = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    device="cuda"
)

model_xgb.fit(Xgb_train, y_train_xgb)

pred_xgb = model_xgb.predict(Xgb_test)

pred_xgb = scaler_y.inverse_transform(pred_xgb.reshape(-1,1))
y_test_inv = scaler_y.inverse_transform(y_test_xgb)

rmse, mae, mape, r2 = calc_metrics(y_test_inv, pred_xgb)

results.append(["XGBoost", rmse, mae, mape, r2])


# =====================================================
# 3. LSTM
# =====================================================

print("🚀 Training LSTM...")

model_lstm = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, X_train.shape[2])),
    LSTM(32),
    Dense(1)
])

model_lstm.compile(
    loss="mse",
    optimizer="adam"
)

model_lstm.fit(
    X_train, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

pred_lstm = model_lstm.predict(X_test)

pred_lstm = scaler_y.inverse_transform(pred_lstm)
y_test_inv = scaler_y.inverse_transform(y_test)

rmse, mae, mape, r2 = calc_metrics(y_test_inv, pred_lstm)

results.append(["LSTM", rmse, mae, mape, r2])


# =====================================================
# 4. CNN-LSTM
# =====================================================

print("🚀 Training CNN-LSTM...")

model_cnn = Sequential([

    Conv1D(64, kernel_size=3, activation="relu",
           input_shape=(SEQ_LEN, X_train.shape[2])),

    MaxPooling1D(2),

    LSTM(64),

    Dense(1)
])

model_cnn.compile(
    loss="mse",
    optimizer="adam"
)

model_cnn.fit(
    X_train, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

pred_cnn = model_cnn.predict(X_test)

pred_cnn = scaler_y.inverse_transform(pred_cnn)

rmse, mae, mape, r2 = calc_metrics(y_test_inv, pred_cnn)

results.append(["CNN-LSTM", rmse, mae, mape, r2])


# =====================================================
# SAVE RESULTS
# =====================================================

df_res = pd.DataFrame(
    results,
    columns=["Model", "RMSE", "MAE", "MAPE(%)", "R2"]
)

df_res.to_csv(RESULT_FILE, index=False)

print("\n📊 MODEL COMPARISON")
print(df_res)


# =====================================================
# PLOT
# =====================================================

plt.figure(figsize=(10,6))

plt.bar(df_res["Model"], df_res["RMSE"])

plt.title("Model Comparison (RMSE)")
plt.ylabel("RMSE (W)")
plt.grid(True)

plt.savefig(PLOT_FILE)
plt.show()


print("\n✅ DONE")
print("Saved:", RESULT_FILE)
print("Saved:", PLOT_FILE)
