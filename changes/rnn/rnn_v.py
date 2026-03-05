import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, Dense, LSTM, GRU, Conv1D, Flatten, 
                                     Bidirectional, Dropout, Multiply, Add, 
                                     Concatenate, GlobalAveragePooling1D, MaxPooling1D)

print("🚀 RNN-Elite Research Framework: Corrected Version Started")

# =====================================================
# 1. CONFIGURATION
# =====================================================
files=[
    "V-Shape/62cm/TCT/V-Shape_62cm_TCT_15min.csv",
    "V-Shape/62cm/TSSC/V-Shape_62cm_TSSC_15min.csv",
    "V-Shape/77cm/Se-P/V-Shape_77cm_Se-P_15min.csv",
    "V-Shape/77cm/TCT/V-Shape_77cm_TCT_15min.csv",
    "V-Shape/77cm/TSSC/V-Shape_77cm_TSSC_15min.csv",
    "V-Shape/93cm/Se-P/V-Shape_93cm_Se-P_15min.csv",
    "V-Shape/93cm/TCT/V-Shape_93cm_TCT_15min.csv",
    "V-Shape/93cm/TSSC/V-Shape_93cm_TSSC_15min.csv"
]
for i in range(len(files)):
    DATA_FILE = fr"C:\Users\muthumaniraj\OneDrive\Documents\me research\Final _PV_FULL_DATASET\{files[i]}"
    TARGET = "Irradiance_Wm2"
    EXOGENOUS = ["AmbientTemp_C", "CellTemp_C"] 
    LOOKBACK = 24  
    EPOCHS = 120
    BATCH = 64

    # Load and Preprocess
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
    # 2. FIXED RNN ALGORITHM REPOSITORY
    # =====================================================

    def LSTM_Model(lb, nv):
        m = Sequential([
            LSTM(64, return_sequences=True, input_shape=(lb, nv)), 
            LSTM(32), 
            Dense(1)
        ], name="LSTM_Model")
        m.compile(loss="mse", optimizer="adam")
        return m

    def GRU_Model(lb, nv):
        m = Sequential([
            GRU(64, return_sequences=True, input_shape=(lb, nv)), 
            GRU(32), 
            Dense(1)
        ], name="GRU_Model")
        m.compile(loss="mse", optimizer="adam")
        return m

    def BiLSTM_Model(lb, nv):
        m = Sequential([
            Bidirectional(LSTM(64, return_sequences=True), input_shape=(lb, nv)), 
            Bidirectional(LSTM(32)), 
            Dense(1)
        ], name="BiLSTM_Model")
        m.compile(loss="mse", optimizer="adam")
        return m

    def CNN_LSTM_Hybrid(lb, nv):
        m = Sequential([
            Conv1D(64, 3, activation="relu", input_shape=(lb, nv)), 
            MaxPooling1D(2), 
            LSTM(50), 
            Dense(1)
        ], name="CNN_LSTM_Hybrid")
        m.compile(loss="mse", optimizer="adam")
        return m

    def LSTM_GRU_Combo(lb, nv):
        inputs = Input(shape=(lb, nv))
        path1 = LSTM(64, return_sequences=False)(inputs)
        path2 = GRU(64, return_sequences=False)(inputs)
        merged = Concatenate()([path1, path2])
        outputs = Dense(1)(Dense(32, activation='relu')(merged))
        model = Model(inputs, outputs, name="LSTM_GRU_Combo")
        model.compile(loss="mse", optimizer="adam")
        return model

    def BiLSTM_GRU_Combo(lb, nv):
        inputs = Input(shape=(lb, nv))
        x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
        x = GRU(32, return_sequences=False)(x)
        outputs = Dense(1)(x)
        model = Model(inputs, outputs, name="BiLSTM_GRU_Combo")
        model.compile(loss="mse", optimizer="adam")
        return model

    def Gated_Residual_RNN(lb, nv):
        """Champion: Gated Residual Sequential Network"""
        inputs = Input(shape=(lb, nv))
        # Path 1: Recurrent Extraction
        x = Bidirectional(GRU(64, return_sequences=True))(inputs)
        # Path 2: Identity Residual (requires projection to match dims)
        res = Conv1D(128, 1)(inputs) 
        x = Add()([x, res])
        # Gating
        gate = Dense(128, activation='sigmoid')(x)
        x = Multiply()([x, gate])
        # Dimensional Reduction
        x = GRU(64, return_sequences=False)(x)
        outputs = Dense(1)(x)
        model = Model(inputs, outputs, name="Gated_Residual_RNN")
        model.compile(loss="huber", optimizer="adam")
        return model

    # =====================================================
    # 3. CONSOLIDATED TRAINING & EVALUATION
    # =====================================================

    n_vars = len(features)
    rnn_models = {
        "LSTM": LSTM_Model(LOOKBACK, n_vars),
        "GRU": GRU_Model(LOOKBACK, n_vars),
        "BiLSTM": BiLSTM_Model(LOOKBACK, n_vars),
        "CNN-LSTM": CNN_LSTM_Hybrid(LOOKBACK, n_vars),
        "LSTM-GRU": LSTM_GRU_Combo(LOOKBACK, n_vars),
        "BiLSTM-GRU": BiLSTM_GRU_Combo(LOOKBACK, n_vars),
        "Gated-Residual-RNN": Gated_Residual_RNN(LOOKBACK, n_vars)
    }

    final_results = []

    for name, model in rnn_models.items():
        print(f"\n📘 Training Architecture: {name}...")
        
        # EarlyStopping helps prevent overfitting in long 120-epoch runs
        es = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH, verbose=0, callbacks=[es])
        
        # Evaluation
        y_pred = model.predict(X_test, verbose=0)
        y_train_pred = model.predict(X_train, verbose=0)
        
        r2_te = r2_score(y_test, y_pred)
        r2_tr = r2_score(y_train, y_train_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        # Precision within 10% error margin
        precision = np.mean(np.abs((y_test - y_pred.flatten()) / (y_test + 1e-7)) < 0.10)

        final_results.append({
            "Model": name,
            "Train_Acc(R2)": round(r2_tr, 8),
            "Test_Acc(R2)": round(r2_te, 8),
            "RMSE": round(rmse, 8),
            "MAE": round(mae, 8),
            "MSE": round(mse, 10),
            "Precision*": round(precision, 8)
        })
        print(f"✅ {name} Completed Successfully.")

    # =====================================================
    # 4. FINAL REPORT GENERATION
    # =====================================================
    report_df = pd.DataFrame(final_results).sort_values(by="Test_Acc(R2)", ascending=False)
    print("\n" + "="*95)
    print(f"🏁 FINAL CONSOLIDATED PERFORMANCE MATRIX {files[i].split('/')} (IRRADIANCE)")
    print("="*95)
    print(report_df.to_string(index=False))
    print("="*95)

    report_df.to_csv("Final_RNN_Comparison_Results.csv", index=False)
    print("\n📁 Saved to: Final_RNN_Comparison_Results.csv")