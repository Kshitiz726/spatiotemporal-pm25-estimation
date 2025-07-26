import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load hourly merged data
df = pd.read_csv("merged_pm25_weather_kathmandu_hourly.csv")
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

# Create cyclical encodings for hour and month
df['hour'] = df['datetime'].dt.hour
df['month'] = df['datetime'].dt.month
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Create lag features for pm25 and weather variables
lag_hours = [1, 2, 3]
weather_cols = df.columns.drop(['datetime', 'pm25', 'hour', 'month', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos'])

for lag in lag_hours:
    df[f'pm25_lag_{lag}'] = df['pm25'].shift(lag)
    for col in weather_cols:
        df[f'{col}_lag_{lag}'] = df[col].shift(lag)

# Drop rows with NaNs after lagging
df = df.dropna().reset_index(drop=True)

# Prepare features and target
feature_cols = df.columns.drop(['datetime', 'pm25'])
X = df[feature_cols].values
y = df['pm25'].values.reshape(-1, 1)

# Scale and save
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

np.save("X_scaled_kathmandu_hourly.npy", X_scaled)
np.save("y_scaled_kathmandu_hourly.npy", y_scaled)
joblib.dump(scaler_X, "scaler_X_kathmandu_hourly.pkl")
joblib.dump(scaler_y, "scaler_y_kathmandu_hourly.pkl")

print("✅ Hourly lag features, cyclical encodings, and scaling prepared and saved.")
print(f"Features used: {list(feature_cols)}")