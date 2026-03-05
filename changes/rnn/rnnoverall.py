import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, Dense, LSTM, GRU, Conv1D, Flatten, 
                                     Bidirectional, Dropout, Multiply, Add, 
                                     Concatenate, GlobalAveragePooling1D, MaxPooling1D,
                                     LayerNormalization, Activation, Reshape)

# =====================================================
# 0. REPRODUCIBILITY BLOCK (CRITICAL FOR RESEARCH)
# =====================================================
def set_seeds(seed=42):
    """Locks all random generators for identical results across all 28 files."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Ensures deterministic GPU behavior
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

set_seeds(42)
print("🚀 Seeds Set (42). Training is now Deterministic across all files.")

# =====================================================
# 1. CONFIGURATION & FILE LIST
# =====================================================
# The root path where your datasets are stored
BASE_PATH = r"C:\Users\muthumaniraj\OneDrive\Documents\me research\Final _PV_FULL_DATASET"

files = [
    "PV_MASTER_15MIN_2024_2025.csv",
    "Flat/62cm/Se-P/Flat_62cm_Se-P_15min.csv",
    "Flat/62cm/TCT/Flat_62cm_TCT_15min.csv",
    "Flat/62cm/TSSC/Flat_62cm_TSSC_15min.csv",
    "Flat/77cm/Se-P/Flat_77cm_Se-P_15min.csv",
    "Flat/77cm/TCT/Flat_77cm_TCT_15min.csv",
    "Flat/77cm/TSSC/Flat_77cm_TSSC_15min.csv",
    "Flat/93cm/Se-P/Flat_93cm_Se-P_15min.csv",
    "Flat/93cm/TCT/Flat_93cm_TCT_15min.csv",
    "Flat/93cm/TSSC/Flat_93cm_TSSC_15min.csv",
    "Inverted-V/62cm/Se-P/Inverted-V_62cm_Se-P_15min.csv",
    "Inverted-V/62cm/TCT/Inverted-V_62cm_TCT_15min.csv",
    "Inverted-V/62cm/TSSC/Inverted-V_62cm_TSSC_15min.csv",
    "Inverted-V/77cm/Se-P/Inverted-V_77cm_Se-P_15min.csv",
    "Inverted-V/77cm/TCT/Inverted-V_77cm_TCT_15min.csv",
    "Inverted-V/77cm/TSSC/Inverted-V_77cm_TSSC_15min.csv",
    "Inverted-V/93cm/Se-P/Inverted-V_93cm_Se-P_15min.csv",
    "Inverted-V/93cm/TCT/Inverted-V_93cm_TCT_15min.csv",
    "Inverted-V/93cm/TSSC/Inverted-V_93cm_TSSC_15min.csv",
    "V-Shape/62cm/Se-P/V-Shape_62cm_Se-P_15min.csv",
    "V-Shape/62cm/TCT/V-Shape_62cm_TCT_15min.csv",
    "V-Shape/62cm/TSSC/V-Shape_62cm_TSSC_15min.csv",
    "V-Shape/77cm/Se-P/V-Shape_77cm_Se-P_15min.csv",
    "V-Shape/77cm/TCT/V-Shape_77cm_TCT_15min.csv",
    "V-Shape/77cm/TSSC/V-Shape_77cm_TSSC_15min.csv",
    "V-Shape/93cm/Se-P/V-Shape_93cm_Se-P_15min.csv",
    "V-Shape/93cm/TCT/V-Shape_93cm_TCT_15min.csv",
    "V-Shape/93cm/TSSC/V-Shape_93cm_TSSC_15min.csv"
]

TARGET = "Irradiance_Wm2"
EXOGENOUS = ["AmbientTemp_C", "CellTemp_C"] 
LOOKBACK = 24  
EPOCHS = 120
BATCH = 64
OUTPUT_BASE_DIR = r"C:\Users\muthumaniraj\OneDrive\Documents\me research\changes\rnn\anothercheck"
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# =====================================================
# 2. MODEL DEFINITIONS
# =====================================================

def Gated_Residual_RNN(lb, nv):
    """
    Optimized Proposed Champion: 
    Dual-Path Temporal Residual Hybrid (DTRH)
    """
    inputs = Input(shape=(lb, nv))
    
    # Path 1: Deep Sequential Extraction (Strengthened)
    x = Bidirectional(GRU(64, return_sequences=True))(inputs)
    x = Bidirectional(GRU(64, return_sequences=True))(x) # Added depth for complex patterns
    
    # Path 2: Spatial Feature Extraction (Conv1D)
    res = Conv1D(128, 3, padding='same')(inputs) # Kernel size 3 captures temporal spikes better
    
    # Fusion Block: Residual Gating
    merged = Add()([x, res])
    merged = LayerNormalization()(merged)
    
    # Attention-based Gate
    gate = Dense(128, activation='sigmoid')(merged)
    refined = Multiply()([merged, gate])
    
    # Path 3: Final Global Refinement
    # This prevents the "Information Fade" seen in standard GRUs
    final_seq = GRU(64, return_sequences=False)(refined)
    
    # High-Precision Projection
    # Using AdamW and GELU for smoother gradient descent
    x = Dense(64)(final_seq)
    x = Activation('gelu')(x) 
    outputs = Dense(1)(x)
    
    model = Model(inputs, outputs, name="Proposed_DTRH_Hybrid")
    # AdamW handles the complexity of 120 epochs much better than standard Adam
    model.compile(loss="huber", optimizer=tf.keras.optimizers.AdamW(1e-3))
    return model

def build_baselines(lb, nv):
    """Standard Competitive Models"""
    models = {}
    models["LSTM"] = Sequential([LSTM(64, return_sequences=True, input_shape=(lb, nv)), LSTM(32), Dense(1)])
    models["GRU"] = Sequential([GRU(64, return_sequences=True, input_shape=(lb, nv)), GRU(32), Dense(1)])
    models["BiLSTM"] = Sequential([Bidirectional(LSTM(64, return_sequences=True), input_shape=(lb, nv)), Bidirectional(LSTM(32)), Dense(1)])
    models["CNN-LSTM"] = Sequential([Conv1D(64, 3, activation="relu", input_shape=(lb, nv)), MaxPooling1D(2), LSTM(50), Dense(1)])
    
    # LSTM-GRU Hybrid
    inp = Input(shape=(lb, nv))
    merged = Concatenate()([LSTM(64)(inp), GRU(64)(inp)])
    models["LSTM-GRU"] = Model(inp, Dense(1)(Dense(32, activation='relu')(merged)))
    
    # BiLSTM-GRU Hybrid
    inp2 = Input(shape=(lb, nv))
    x2 = Bidirectional(LSTM(64, return_sequences=True))(inp2)
    models["BiLSTM-GRU"] = Model(inp2, Dense(1)(GRU(32)(x2)))
    
    for name, m in models.items():
        m.compile(loss="mse", optimizer="adam")
    return models

# =====================================================
# 3. PROCESSING LOOP
# =====================================================

for file_rel_path in files:
    # Clean the path and build full string
    DATA_FILE = os.path.join(BASE_PATH, file_rel_path.strip())
    # Create a safe filename for the results
    file_id = file_rel_path.replace("/", "_").replace("\\", "_").replace(" ", "").replace(".csv", "")
    
    print(f"\n{'='*70}\n📂 CURRENT DATASET: {file_rel_path}\n{'='*70}")

    if not os.path.exists(DATA_FILE):
        print(f"⚠️ Warning: File not found at {DATA_FILE}. Skipping...")
        continue

    # Data Loading
    df = pd.read_csv(DATA_FILE)
    df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df = df.sort_values("Datetime")
    
    # Scaling
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[[TARGET] + EXOGENOUS].values)

    # Sequence Creation
    X, y = [], []
    for i in range(len(data_scaled) - LOOKBACK):
        X.append(data_scaled[i:i+LOOKBACK])
        y.append(data_scaled[i+LOOKBACK, 0])
    X, y = np.array(X), np.array(y)
    
    # Train-Test Split (80/20)
    split = int(0.8 * len(X))
    X_train, y_train, X_test, y_test = X[:split], y[:split], X[split:], y[split:]

    # Model Initialization
    n_vars = X.shape[2]
    all_models = build_baselines(LOOKBACK, n_vars)
    all_models["Gated-Residual-RNN"] = Gated_Residual_RNN(LOOKBACK, n_vars)

    file_results = []

    for name, model in all_models.items():
        print(f"📘 Training Model: {name}...")
        es = tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
        
        # Training
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH, verbose=0, 
                  validation_split=0.1, callbacks=[es])
        
        # Performance Evaluation (Test)
        y_pred = model.predict(X_test, verbose=0).flatten()
        r2_test = r2_score(y_test, y_pred)
        
        # Performance Evaluation (Train) - AS REQUESTED
        y_train_pred = model.predict(X_train, verbose=0).flatten()
        r2_train = r2_score(y_train, y_train_pred)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        precision = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-7)) < 0.10)

        file_results.append({
            "Model": name, 
            "Train_Acc(R2)": round(r2_train, 8), # New Column
            "Test_Acc(R2)": round(r2_test, 8),
            "RMSE": round(rmse, 8), 
            "MAE": round(mae, 8), 
            "Precision*": round(precision, 8)
        })
        print(f"✅ {name} Training Success.")

    # Generate Performance Matrix for this specific file
    report_df = pd.DataFrame(file_results).sort_values(by="Test_Acc(R2)", ascending=False)
    print(f"\n📊 COMPARISON TABLE FOR {file_id}:\n{report_df.to_string(index=False)}")
    
    # Save results
    save_path = os.path.join(OUTPUT_BASE_DIR, f"Results_{file_id}.csv")
    report_df.to_csv(save_path, index=False)
    print(f"💾 Results saved to: {save_path}")

print("\n🌟 ALL 28 RESEARCH DATASETS PROCESSED SUCCESSFULLY.")