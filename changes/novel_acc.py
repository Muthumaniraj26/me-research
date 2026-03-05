import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Flatten, Reshape, Multiply, Add, Layer, RNN

print("Deep Learning Advanced Forecasting with Full Metric Tracking Started")

# =====================================================
# CONFIG
# =====================================================
DATA_FILE = r"C:\Users\muthumaniraj\OneDrive\Documents\me research\Final _PV_FULL_DATASET\PV_MASTER_15MIN_2024_2025.csv"
TARGET = "Irradiance_Wm2"
LOOKBACK = 24  
EPOCHS = 25
BATCH = 64

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(DATA_FILE)
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

# =====================================================
# TRAIN / VAL / TEST SPLIT (70/10/20)
# =====================================================
total_len = len(X)
train_split = int(0.7 * total_len)
val_split = int(0.8 * total_len)

X_train, y_train = X[:train_split], y[:train_split]
X_val, y_val = X[train_split:val_split], y[train_split:val_split]
X_test, y_test = X[val_split:], y[val_split:]

print(f"Data Split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# =====================================================
# MODEL DEFINITIONS (LNN, N-BEATS, PatchTST, LBPT-26)
# =====================================================

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
    x = RNN(LiquidRNNCell(64))(inputs)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(loss="mse", optimizer="adam", metrics=['mae'])
    return model

def NBEATS_Model():
    inputs = Input(shape=(LOOKBACK, 1))
    x = Flatten()(inputs)
    d1 = Dense(128, activation='relu')(x)
    backcast = Dense(LOOKBACK)(d1)
    forecast1 = Dense(1)(d1)
    res = Add()([x, -backcast])
    d2 = Dense(128, activation='relu')(res)
    forecast2 = Dense(1)(d2)
    out = Add()([forecast1, forecast2])
    model = Model(inputs, out)
    model.compile(loss="mse", optimizer="adam", metrics=['mae'])
    return model

def PatchTST_Model():
    patch_size = 4
    num_patches = LOOKBACK // patch_size
    inputs = Input(shape=(LOOKBACK, 1))
    x = Reshape((num_patches, patch_size))(inputs)
    x = LSTM(64)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(loss="mse", optimizer="adam", metrics=['mae'])
    return model

def LBPT_26_Model():
    inputs = Input(shape=(LOOKBACK, 1))
    flat = Flatten()(inputs)
    trend = Dense(1)(Dense(128, activation='relu')(flat))
    res = Reshape((LOOKBACK // 8, 8))(inputs)
    liquid_out = RNN(LiquidRNNCell(64))(res)
    residual_forecast = Dense(1)(liquid_out)
    out = Add()([trend, residual_forecast])
    model = Model(inputs, out)
    model.compile(loss="mse", optimizer="adam", metrics=['mae'])
    return model

# =====================================================
# EXECUTION AND METRIC COLLECTION
# =====================================================

models = {
    "LNN": LNN_Model(),
    "N-BEATS": NBEATS_Model(),
    "PatchTST": PatchTST_Model(),
    "LBPT-26": LBPT_26_Model()
}

final_results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val),
        epochs=EPOCHS, 
        batch_size=BATCH, 
        verbose=0
    )
    
    # 1. Training Metrics (Final Epoch)
    train_loss = history.history['loss'][-1]
    
    # 2. Validation Metrics (Final Epoch)
    val_loss = history.history['val_loss'][-1]
    
    # 3. Testing Metrics
    test_preds = model.predict(X_test)
    test_loss = mean_squared_error(y_test, test_preds)
    
    # Accuracy Proxy (R2 Score) for Train, Val, and Test
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    train_r2 = r2_score(y_train, train_preds)
    val_r2 = r2_score(y_val, val_preds)
    test_r2 = r2_score(y_test, test_preds)
    
    final_results.append({
        "Model": name,
        "Train_Loss": train_loss,
        "Val_Loss": val_loss,
        "Test_Loss": test_loss,
        "Train_Acc(R2)": train_r2,
        "Val_Acc(R2)": val_r2,
        "Test_Acc(R2)": test_r2
    })
    print(f"{name} Evaluation Complete")

# =====================================================
# DISPLAY RESULTS
# =====================================================
results_df = pd.DataFrame(final_results)
print("\nDETAILED ACCURACY AND LOSS REPORT")
print(results_df.to_string(index=False))

results_df.to_csv("model_accuracy_loss_report.csv", index=False)
print("\nReport saved to model_accuracy_loss_report.csv")