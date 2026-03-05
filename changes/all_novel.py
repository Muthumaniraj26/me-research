import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, LSTM, GRU, Conv1D, Flatten, Reshape, 
                                     Add, Subtract, Multiply, RNN, Bidirectional, 
                                     Dropout, MultiHeadAttention, LayerNormalization, 
                                     Permute, GlobalAveragePooling1D)

# =====================================================
# 1. DATA PREPROCESSING & GLOBAL CONFIG
# =====================================================
DATA_FILE = r"C:\Users\muthumaniraj\OneDrive\Documents\me research\Final _PV_FULL_DATASET\PV_MASTER_15MIN_2024_2025.csv"
TARGET = "Irradiance_Wm2"
EXOGENOUS = ["AmbientTemp_C", "CellTemp_C"] 
LOOKBACK = 24  # Time steps back
EPOCHS = 15     # Reduced for quick multi-model comparison
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
# 2. TRADITIONAL DEEP LEARNING MODELS
# =====================================================

def Simple_LSTM(lookback, n_vars):
    inputs = Input(shape=(lookback, n_vars))
    x = LSTM(64)(inputs)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(loss="mse", optimizer="adam")
    return model

def BiGRU_Hybrid(lookback, n_vars):
    inputs = Input(shape=(lookback, n_vars))
    x = Bidirectional(GRU(64, return_sequences=True))(inputs)
    x = GRU(32)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(loss="mse", optimizer="adam")
    return model

# =====================================================
# 3. TRANSFORMER & ATTENTION MODELS
# =====================================================

def iTransformer_Model(lookback, n_vars):
    inputs = Input(shape=(lookback, n_vars))
    x = Permute((2, 1))(inputs) # Inversion Layer
    x = Dense(128)(x)
    attn = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = Add()([x, attn])
    x = LayerNormalization()(x)
    x = Flatten()(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(loss="mse", optimizer="adam")
    return model

# =====================================================
# 4. RECENT (2025/26) ALGORITHMS & RESIDUAL NETS
# =====================================================

def TCN_Model(lookback, n_vars):
    inputs = Input(shape=(lookback, n_vars))
    x = Conv1D(64, 3, padding='causal', dilation_rate=1, activation='relu')(inputs)
    x = Conv1D(64, 3, padding='causal', dilation_rate=2, activation='relu')(x)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(loss="mse", optimizer="adam")
    return model

def CNN_LSTM_SOTA(lookback, n_vars):
    inputs = Input(shape=(lookback, n_vars))
    x = Conv1D(64, 3, activation='relu')(inputs)
    x = LSTM(64)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(loss="mse", optimizer="adam")
    return model

# =====================================================
# 5. LIQUID & SEGMENTED MODELS (HYBRID)
# =====================================================

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

def Liquid_Neural_Net(lookback, n_vars):
    inputs = Input(shape=(lookback, n_vars))
    x = RNN(LiquidRNNCell(64))(inputs)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(loss="mse", optimizer="adam")
    return model

# =====================================================
# 6. PERFORMANCE EVALUATION SUITE
# =====================================================

models_to_test = {
    "TRADITIONAL: LSTM": Simple_LSTM(LOOKBACK, len(features)),
    "TRADITIONAL: Bi-GRU": BiGRU_Hybrid(LOOKBACK, len(features)),
    "TRANSFORMER: iTransformer": iTransformer_Model(LOOKBACK, len(features)),
    "RECENT: TCN": TCN_Model(LOOKBACK, len(features)),
    "RECENT: CNN-LSTM": CNN_LSTM_SOTA(LOOKBACK, len(features)),
    "HYBRID: Liquid-NN": Liquid_Neural_Net(LOOKBACK, len(features))
}

results = []

for name, model in models_to_test.items():
    print(f"\n>>> Running: {name}")
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH, verbose=0)
    
    y_pred_test = model.predict(X_test, verbose=0)
    y_pred_train = model.predict(X_train, verbose=0)
    
    # Core Regression Metrics
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    
    # Accuracy Logic (10% Threshold)
    hits = np.abs((y_test - y_pred_test.flatten()) / (y_test + 1e-7)) < 0.10
    precision = np.sum(hits) / len(hits)
    f1 = precision # For single-class hit rate, P=R=F1

    results.append({
        "Algorithm": name,
        "Train_R2": round(r2_train, 4),
        "Test_R2": round(r2_test, 4),
        "RMSE": round(rmse, 5),
        "MAE": round(mae, 5),
        "Precision*": round(precision, 4),
        "F1_Score*": round(f1, 4)
    })

# =====================================================
# 7. FINAL COMPARISON MATRIX
# =====================================================
final_df = pd.DataFrame(results)
print("\n" + "="*100)
print("PV IRRADIANCE FORECASTING RESEARCH MATRIX (2024-2026)")
print("="*100)
print(final_df.sort_values(by="Test_R2", ascending=False).to_string(index=False))
print("="*100)
print("*Note: Precision/F1 calculated at 10% error tolerance.")