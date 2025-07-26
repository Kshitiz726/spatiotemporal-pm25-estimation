import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib  # for saving scalers

# 1️⃣ Load merged CSV here
df = pd.read_csv("merged_pm25_weather_kathmandu.csv")

# Ensure date column is datetime and sort by date ascending
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# 2️⃣ Define lag days to create lag features
lag_days = [1, 2, 3]  # You can adjust or extend this list

# 3️⃣ Create lag features for target (pm25_mean)
for lag in lag_days:
    df[f'pm25_mean_lag_{lag}'] = df['pm25_mean'].shift(lag)

# 4️⃣ Create lag features for weather variables
weather_cols = [
    'Precip', 'Pressure', 'Humidity_2m', 'RH_2m', 'Temp_2m', 'WetBulbTemp_2m',
    'MaxTemp_2m', 'MinTemp_2m', 'TempRange_2m', 'EarthSkinTemp',
    'WindSpeed_10m', 'MaxWindSpeed_10m', 'MinWindSpeed_10m', 'WindSpeedRange_10m',
    'WindSpeed_50m', 'MaxWindSpeed_50m', 'MinWindSpeed_50m', 'WindSpeedRange_50m'
]

for col in weather_cols:
    for lag in lag_days:
        df[f'{col}_lag_{lag}'] = df[col].shift(lag)

# 5️⃣ Drop rows with any NaNs (created by lagging)
df = df.dropna().reset_index(drop=True)

# 6️⃣ Define feature columns to use for model input (current + lag features)
feature_cols = weather_cols.copy()  # current weather features
# Add lagged weather features
for col in weather_cols:
    for lag in lag_days:
        feature_cols.append(f'{col}_lag_{lag}')
# Add lagged PM2.5 features (exclude current pm25_mean as it's target)
for lag in lag_days:
    feature_cols.append(f'pm25_mean_lag_{lag}')

# 7️⃣ Extract X and y
X = df[feature_cols].values
y = df['pm25_mean'].values.reshape(-1, 1)

# 8️⃣ Scale features and target
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 9️⃣ Save scaled data and scalers for DNN training and inverse transforms later
np.save("X_scaled_kathmandu_with_lags.npy", X_scaled)
np.save("y_scaled_kathmandu_with_lags.npy", y_scaled)

joblib.dump(scaler_X, "scaler_X_kathmandu_with_lags.pkl")
joblib.dump(scaler_y, "scaler_y_kathmandu_with_lags.pkl")

print("✅ Z-scaling with lag features completed and saved:")
print("- X_scaled_kathmandu_with_lags.npy")
print("- y_scaled_kathmandu_with_lags.npy")
print("- scaler_X_kathmandu_with_lags.pkl")
print("- scaler_y_kathmandu_with_lags.pkl")
print("Ready for DNN training.")
