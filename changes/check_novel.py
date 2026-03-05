import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Flatten, Reshape, Add, RNN, MultiHeadAttention, LayerNormalization, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

print("Deep Learning Advanced Forecasting with Full Metric Evaluation Started")

# =====================================================
# CONFIG
# =====================================================
DATA_FILE = r"C:\Users\muthumaniraj\OneDrive\Documents\me research\Final _PV_FULL_DATASET\PV_MASTER_15MIN_2024_2025.csv"
TARGET = "Irradiance_Wm2"
LOOKBACK = 96  
EPOCHS = 50
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
# LAYER DEFINITION: LIQUID RNN CELL
# =====================================================
# class LiquidRNNCell(tf.keras.layers.Layer):
#     def __init__(self, units, **kwargs):
#         self.units = units
#         self.state_size = units
#         super().__init__(**kwargs)
#     def build(self, input_shape):
#         self.kernel = self.add_weight(shape=(input_shape[-1], self.units), initializer='glorot_uniform', name='kernel')
#         self.recurrent_kernel = self.add_weight(shape=(self.units, self.units), initializer='orthogonal', name='recurrent_kernel')
#         self.tau = self.add_weight(shape=(self.units,), initializer='ones', name='tau')
#     def call(self, inputs, states):
#         prev_h = states[0]
#         gate = tf.nn.tanh(tf.matmul(inputs, self.kernel) + tf.matmul(prev_h, self.recurrent_kernel))
#         h = prev_h + (1.0 / (tf.exp(self.tau) + 1.0)) * (-prev_h + gate)
#         return h, [h]

# =====================================================
# MODEL ARCHITECTURES
# =====================================================

# def LNN_Model():
#     inputs = Input(shape=(LOOKBACK, 1))
#     x = RNN(LiquidRNNCell(64))(inputs)
#     outputs = Dense(1)(x)
#     model = Model(inputs, outputs)
#     model.compile(loss="mse", optimizer="adam")
#     return model

def nbeats_block(input_layer, lookback):
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
    backcast = Dense(lookback)(d3)
    backcast = Reshape((lookback, 1))(backcast)
    
    # Forecast predicts the next step
    forecast = Dense(1)(d3)
    
    return backcast, forecast

def Attentive_NBEATS_Model():
    inputs = Input(shape=(LOOKBACK, 1))
    
    # Stack 1 (Trend/Global Patterns)
    backcast1, forecast1 = nbeats_block(inputs, LOOKBACK)
    res1 = Add()([inputs, -backcast1])
    
    # Stack 2 (Seasonality/Local Fluctuations)
    backcast2, forecast2 = nbeats_block(res1, LOOKBACK)
    res2 = Add()([res1, -backcast2])

    # Stack 3 (High Frequency)
    backcast3, forecast3 = nbeats_block(res2, LOOKBACK)
    
    # Final Output is the sum of all forecasts
    output = Add()([forecast1, forecast2, forecast3])
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss="mse", optimizer="adam")
    return model

# def PatchTST_Model():
#     patch_size = 4
#     num_patches = LOOKBACK // patch_size
#     inputs = Input(shape=(LOOKBACK, 1))
#     x = Reshape((num_patches, patch_size))(inputs)
#     x = LSTM(64)(x)
#     outputs = Dense(1)(x)
#     model = Model(inputs, outputs)
#     model.compile(loss="mse", optimizer="adam")
#     return model

# def LBPT_26_Model():
#     inputs = Input(shape=(LOOKBACK, 1))
#     # N-BEATS Trend Extraction
#     flat = Flatten()(inputs)
#     trend = Dense(1)(Dense(128, activation='relu')(flat))
#     # Liquid Patching Residual
#     res = Reshape((LOOKBACK // 8, 8))(inputs)
#     liquid_out = RNN(LiquidRNNCell(64))(res)
#     residual_forecast = Dense(1)(liquid_out)
#     out = Add()([trend, residual_forecast])
#     model = Model(inputs, out)
#     model.compile(loss="mse", optimizer="adam")
#     return model

# =====================================================
# EVALUATION AND REPORTING
# =====================================================

models = {
    # "LNN": LNN_Model(),
    "Attentive_N-BEATS": Attentive_NBEATS_Model(),
    # "PatchTST": PatchTST_Model(),
    # "LBPT-26": LBPT_26_Model()
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val),
        epochs=EPOCHS, 
        batch_size=BATCH, 
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ],
        verbose=1
    )
    
    # Loss extraction (Scaled MSE)
    train_loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]
    
    # Predictions for Accuracy (R2 Score)
    train_preds = model.predict(X_train, verbose=0)
    val_preds = model.predict(X_val, verbose=0)
    test_preds = model.predict(X_test, verbose=0)
    
    # Accuracy Proxy calculation
    train_acc = r2_score(y_train, train_preds)
    val_acc = r2_score(y_val, val_preds)
    test_acc = r2_score(y_test, test_preds)
    
    # Testing Loss
    test_loss = mean_squared_error(y_test, test_preds)
    
    results.append({
        "Model": name,
        "Train_Loss": train_loss,
        "Val_Loss": val_loss,
        "Test_Loss": test_loss,
        "Train_Acc(R2)": train_acc,
        "Val_Acc(R2)": val_acc,
        "Test_Acc(R2)": test_acc
    })
    print(f"{name} Evaluation Complete")

# =====================================================
# FINAL REPORT
# =====================================================
results_df = pd.DataFrame(results)
print("\nDETAILED ACCURACY AND LOSS REPORT")
print(results_df.to_string(index=False))

results_df.to_csv("full_evaluation_matrix_irradiance.csv", index=False)
print("\nSaved: full_evaluation_matrix_irradiance.csv")