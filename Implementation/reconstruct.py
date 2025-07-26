# reconstruct.py

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# 1️⃣ Load your daily climate data
df = pd.read_csv("merged_pm25_weather_kathmandu.csv")

# Ensure consistent column names if needed
df.columns = df.columns.str.strip().str.lower()

# If 'district' column exists and filtering needed:
if 'district' in df.columns:
    df = df[df['district'].str.lower() == 'kathmandu']

# Ensure 'date' is datetime and sort
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# 2️⃣ Reconstruct features exactly as during training
lag_days = [1, 2, 3]

weather_cols = [
    'precip', 'pressure', 'humidity_2m', 'rh_2m', 'temp_2m', 'wetbulbtemp_2m',
    'maxtemp_2m', 'mintemp_2m', 'temprange_2m', 'earthskintemp',
    'windspeed_10m', 'maxwindspeed_10m', 'minwindspeed_10m', 'windspeedrange_10m',
    'windspeed_50m', 'maxwindspeed_50m', 'minwindspeed_50m', 'windspeedrange_50m'
]

# Create lagged PM2.5 features
for lag in lag_days:
    df[f'pm25_mean_lag_{lag}'] = df['pm25_mean'].shift(lag)

# Create lagged weather features
for col in weather_cols:
    for lag in lag_days:
        df[f'{col}_lag_{lag}'] = df[col].shift(lag)

# Drop rows with NaNs due to lagging
df = df.dropna().reset_index(drop=True)

# Prepare feature columns for model
feature_cols = weather_cols.copy()
for col in weather_cols:
    for lag in lag_days:
        feature_cols.append(f'{col}_lag_{lag}')
for lag in lag_days:
    feature_cols.append(f'pm25_mean_lag_{lag}')

# 3️⃣ Load scaler and scale X
scaler_X = joblib.load("scaler_X_kathmandu_with_lags.pkl")
X = df[feature_cols].values
X_scaled = scaler_X.transform(X)

# 4️⃣ Load trained DNN model
model = load_model("dnn_pm25_kathmandu_improved.keras")

# 5️⃣ Predict
y_pred_scaled = model.predict(X_scaled)

# 6️⃣ Inverse transform predictions
scaler_y = joblib.load("scaler_y_kathmandu_with_lags.pkl")
y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

# 7️⃣ Prepare output DataFrame
out_df = pd.DataFrame({
    "Date": df["date"],
    "Predicted_PM2.5": y_pred
})

# Print results to CMD
print(out_df)

# 8️⃣ Save to CSV
out_df.to_csv("reconstructed_pm25.csv", index=False)
print("\n✅ Reconstruction completed and saved to reconstructed_pm25.csv")
