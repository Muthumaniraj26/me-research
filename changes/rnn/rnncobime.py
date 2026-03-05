import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, Dense, LSTM, GRU, Conv1D, Flatten, 
                                     Bidirectional, Dropout, Multiply, Add, 
                                     Concatenate, GlobalAveragePooling1D, MaxPooling1D, 
                                     LayerNormalization, Activation, GlobalMaxPooling1D,
                                     Reshape, Lambda)

print("🚀 RNN-Elite Framework: Fixed Champion Logic Started")

# =====================================================
# 1. CONFIGURATION
# =====================================================
DATA_FILE = r"C:\Users\muthumaniraj\OneDrive\Documents\me research\Final _PV_FULL_DATASET\Flat\93cm\TSSC\Flat_93cm_TSSC_15min.csv"
TARGET = "Irradiance_Wm2"
EXOGENOUS = ["AmbientTemp_C", "CellTemp_C"] 
LOOKBACK = 24  
EPOCHS = 120
BATCH = 64

# Data Loading
df = pd.read_csv(DATA_FILE)
df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
df = df.sort_values("Datetime")
features = [TARGET] + EXOGENOUS
data_scaled = MinMaxScaler().fit_transform(df[features].values)

def make_sequences(data, step):
    X, y = [], []
    for i in range(len(data) - step):
        X.append(data[i:i+step, :])
        y.append(data[i+step, 0])
    return np.array(X), np.array(y)

X, y = make_sequences(data_scaled, LOOKBACK)
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# =====================================================
# 2. ALGORITHM REPOSITORY
# =====================================================

# --- EXISTING / USED ALGORITHMS ---
def LSTM_Model(lb, nv):
    return Sequential([LSTM(64, return_sequences=True, input_shape=(lb, nv)), LSTM(32), Dense(1)], name="LSTM")

def GRU_Model(lb, nv):
    return Sequential([GRU(64, return_sequences=True, input_shape=(lb, nv)), GRU(32), Dense(1)], name="GRU")

def BiLSTM_Model(lb, nv):
    return Sequential([Bidirectional(LSTM(64, return_sequences=True), input_shape=(lb, nv)), Bidirectional(LSTM(32)), Dense(1)], name="BiLSTM")

def CNN_LSTM_Hybrid(lb, nv):
    return Sequential([Conv1D(64, 3, activation="relu", input_shape=(lb, nv)), MaxPooling1D(2), LSTM(50), Dense(1)], name="CNN_LSTM")

def LSTM_GRU_Combo(lb, nv):
    inputs = Input(shape=(lb, nv))
    p1, p2 = LSTM(64)(inputs), GRU(64)(inputs)
    merged = Concatenate()([p1, p2])
    return Model(inputs, Dense(1)(Dense(32, activation='relu')(merged)), name="LSTM_GRU")

# --- NEW PROPOSED CHAMPION: OPTIMIZED GR-RNN ---
def Optimized_GR_RNN(lb, nv):
    inputs = Input(shape=(lb, nv))
    
    # 1. Multi-Scale Feature Extraction
    cnn = Conv1D(128, 3, padding='same', activation='relu')(inputs)
    cnn = LayerNormalization()(cnn)
    
    # 2. Recurrent Core
    rnn = Bidirectional(GRU(64, return_sequences=True))(cnn)
    
    # 3. Gated Residual Logic (Fixed Identity Path)
    res = Conv1D(128, 1, padding='same')(inputs) 
    x = Add()([rnn, res])
    
    # 4. SE-Gating (FIXED: Using Reshape instead of tf.expand_dims)
    gate = GlobalAveragePooling1D()(x)
    gate = Dense(64, activation='relu')(gate)
    gate = Dense(128, activation='sigmoid')(gate)
    
    # Instead of tf.expand_dims, we use Reshape layer to match (1, 128)
    gate = Reshape((1, 128))(gate)
    
    x = Multiply()([x, gate])
    x = LayerNormalization()(x)
    
    # 5. Final Refinement
    x = GRU(64, return_sequences=False)(x)
    x = Dense(128)(x)
    x = Activation('gelu')(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs, outputs, name="Optimized_GR_RNN")
    # Using AdamW for better weight decay in long epochs
    model.compile(loss="huber", optimizer=tf.keras.optimizers.AdamW(1e-3, weight_decay=1e-4))
    return model

# =====================================================
# 3. EXECUTION LOOP
# =====================================================
n_vars = len(features)
models = {
    "LSTM": LSTM_Model(LOOKBACK, n_vars),
    "GRU": GRU_Model(LOOKBACK, n_vars),
    "BiLSTM": BiLSTM_Model(LOOKBACK, n_vars),
    "CNN-LSTM": CNN_LSTM_Hybrid(LOOKBACK, n_vars),
    "LSTM-GRU": LSTM_GRU_Combo(LOOKBACK, n_vars),
    "Optimized-GR-RNN (Champion)": Optimized_GR_RNN(LOOKBACK, n_vars)
}

final_results = []

for name, model in models.items():
    print(f"\n📘 Training Architecture: {name}...")
    if "Optimized" not in name:
        model.compile(loss="mse", optimizer="adam")
    
    es = tf.keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH, verbose=0, callbacks=[es])
    
    # Evaluation
    y_pred = model.predict(X_test, verbose=0)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    precision = np.mean(np.abs((y_test - y_pred.flatten()) / (y_test + 1e-7)) < 0.10)

    final_results.append({
        "Model": name, "R2": round(r2, 8), "RMSE": round(rmse, 8), 
        "MAE": round(mae, 8), "Precision*": round(precision, 8)
    })
    print(f"✅ {name} Done.")

# =====================================================
# 4. RESULTS
# =====================================================
report_df = pd.DataFrame(final_results).sort_values(by="R2", ascending=False)
print("\n" + "="*95)
print("🏁 FINAL PERFORMANCE MATRIX (IRRADIANCE)")
print("="*95)
print(report_df.to_string(index=False))
print("="*95)