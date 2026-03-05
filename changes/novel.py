import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Flatten, Reshape, Multiply, Add, Layer, RNN

print("Deep Learning Advanced Forecasting Comparison Started")

# =====================================================
# CONFIG
# =====================================================
DATA_FILE = r"C:\Users\muthumaniraj\OneDrive\Documents\me research\Final _PV_FULL_DATASET\PV_MASTER_15MIN_2024_2025.csv"
TARGET = "Irradiance_Wm2"
LOOKBACK = 24  # 6 hours lookback
EPOCHS = 25
BATCH = 64

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(DATA_FILE)
print("Data Loaded:", df.shape)

df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
df = df.sort_values("Datetime")
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
print("Sequences:", X.shape)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# =====================================================
# METRICS
# =====================================================
def evaluate(y_true, y_pred):
    y_true = scaler.inverse_transform(y_true)
    y_pred = scaler.inverse_transform(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-7))) * 100
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, mape, r2, mse

# =====================================================
# ADVANCED MODEL DEFINITIONS
# =====================================================

# 1. Liquid Neural Network (LNN) - Adaptive Dynamics
class LiquidRNNCell(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units), initializer='glorot_uniform', name='kernel')
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units), initializer='orthogonal', name='recurrent_kernel')
        self.tau = self.add_weight(shape=(self.units,), initializer='ones', name='tau')
    def call(self, inputs, states):
        prev_h = states[0]
        gate = tf.nn.tanh(tf.matmul(inputs, self.kernel) + tf.matmul(prev_h, self.recurrent_kernel))
        h = prev_h + (1.0 / (tf.exp(self.tau) + 1.0)) * (-prev_h + gate)
        return h, [h]

def LNN_Model():
    inputs = Input(shape=(LOOKBACK, 1))
    cell = LiquidRNNCell(64)
    rnn_layer = RNN(cell)
    x = rnn_layer(inputs)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(loss="mse", optimizer="adam")
    return model

# 2. N-BEATS (Neural Basis Expansion Analysis)
def NBEATS_Model():
    inputs = Input(shape=(LOOKBACK, 1))
    x = Flatten()(inputs)
    d1 = Dense(128, activation='relu')(x)
    d2 = Dense(128, activation='relu')(d1)
    backcast1 = Dense(LOOKBACK)(d2)
    forecast1 = Dense(1)(d2)
    res = Add()([x, -backcast1])
    d3 = Dense(128, activation='relu')(res)
    d4 = Dense(128, activation='relu')(d3)
    forecast2 = Dense(1)(d4)
    out = Add()([forecast1, forecast2])
    model = Model(inputs, out)
    model.compile(loss="mse", optimizer="adam")
    return model

# 3. PatchTST (Segmented Time-Series Transformer approach)
def PatchTST_Model():
    patch_size = 4
    num_patches = LOOKBACK // patch_size
    inputs = Input(shape=(LOOKBACK, 1))
    x = Reshape((num_patches, patch_size))(inputs)
    x = Dense(64, activation='relu')(x)
    x = LSTM(64)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(loss="mse", optimizer="adam")
    return model

# 4. LBPT-26 (Novel Hybrid: N-BEATS + Patching + Liquid)
def LBPT_26_Model():
    inputs = Input(shape=(LOOKBACK, 1))
    flat = Flatten()(inputs)
    d1 = Dense(128, activation='relu')(flat)
    trend = Dense(1)(d1)
    res = Reshape((LOOKBACK, 1))(inputs)
    patch_size = 8
    num_patches = LOOKBACK // patch_size
    patches = Reshape((num_patches, patch_size))(res)
    cell = LiquidRNNCell(64)
    liquid_out = RNN(cell)(patches)
    residual_forecast = Dense(1)(liquid_out)
    out = Add()([trend, residual_forecast])
    model = Model(inputs, out)
    model.compile(loss="mse", optimizer="adam")
    return model

# =====================================================
# TRAIN & EVALUATE
# =====================================================
models = {
    "LNN": LNN_Model(),
    "N-BEATS": NBEATS_Model(),
    "PatchTST": PatchTST_Model(),
    "LBPT-26": LBPT_26_Model()
}

results = []
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH, verbose=0)
    preds = model.predict(X_test)
    rmse, mae, mape, r2, mse = evaluate(y_test, preds)
    results.append([name, rmse, mae, mape, r2, mse])
    print(f"{name} Done")

# =====================================================
# RESULT TABLE
# =====================================================
result_df = pd.DataFrame(results, columns=["Model","RMSE","MAE","MAPE(%)","R2","MSE"])
print("\nMODEL COMPARISON (IRRADIANCE TARGET)")
print(result_df.round(4))
result_df.to_csv("advanced_model_comparison_irradiance.csv", index=False)
print("\nSaved: advanced_model_comparison_irradiance.csv")
print("Analysis Complete")