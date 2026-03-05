import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import matplotlib.pyplot as plt


# =====================================================
# LOAD DATA
# =====================================================

print("📥 Loading dataset...")

df = pd.read_csv(r"C:\Users\muthumaniraj\OneDrive\Documents\me research\Final _PV_FULL_DATASET\PV_MASTER_15MIN_2024_2025.csv")

print("Rows:", len(df))


# =====================================================
# FEATURE ENGINEERING
# =====================================================

# Time features
df["Date"] = pd.to_datetime(df["Date"])

df["Hour"]  = df["Date"].dt.hour
df["Day"]   = df["Date"].dt.day
df["Month"] = df["Date"].dt.month


# Encode categorical
le_shape = LabelEncoder()
le_conn  = LabelEncoder()

df["Shape_enc"] = le_shape.fit_transform(df["Shape"])
df["Conn_enc"]  = le_conn.fit_transform(df["Connection"])


# =====================================================
# INPUT / OUTPUT
# =====================================================

X = df[[
    "Irradiance_Wm2",
    "AmbientTemp_C",
    "CellTemp_C",
    "Spacing_cm",
    "Shape_enc",
    "Conn_enc",
    "Hour",
    "Day",
    "Month"
]]

y = df["Pm_W"]


# =====================================================
# TIME-BASED SPLIT
# =====================================================

split = int(len(df) * 0.75)

X_train = X.iloc[:split]
X_test  = X.iloc[split:]

y_train = y.iloc[:split]
y_test  = y.iloc[split:]


# =====================================================
# SCALE
# =====================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


# =====================================================
# MODELS
# =====================================================

models = {

    "Linear": LinearRegression(),

    "RandomForest": RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective="reg:squarederror"
    )
}


# =====================================================
# TRAIN + EVALUATE
# =====================================================

results = []

predictions = {}

print("\n🚀 Training models...\n")

for name, model in models.items():

    print(f"Training {name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    MAE  = mean_absolute_error(y_test, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    MAPE = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    R2   = r2_score(y_test, y_pred)

    results.append([name, MAE, RMSE, MAPE, R2])

    predictions[name] = y_pred



# =====================================================
# RESULTS TABLE
# =====================================================

results_df = pd.DataFrame(
    results,
    columns=["Model", "MAE", "RMSE", "MAPE_%", "R2"]
)

print("\n📊 MODEL COMPARISON")
print(results_df)

import os
output_dir = "output_forecast"
os.makedirs(output_dir, exist_ok=True)

csv_path = os.path.join(output_dir, "forecast_results.csv")
results_df.to_csv(csv_path, index=False)


# =====================================================
# PLOT PREDICTION
# =====================================================

plt.figure(figsize=(14,6))

plt.plot(y_test.values[:500], label="Actual", linewidth=2)

for name, y_pred in predictions.items():
    plt.plot(y_pred[:500], label=name)

plt.title("PV Power Forecast (Sample Window)")
plt.xlabel("Time Step")
plt.ylabel("Power (W)")
plt.legend()
plt.grid(True)

plt.tight_layout()
png_path = os.path.join(output_dir, "forecast_comparison.png")
plt.savefig(png_path)
plt.show()


print("\n✅ Forecasting complete")
print(f"📁 Saved: {csv_path}")
print(f"📈 Saved: {png_path}")
