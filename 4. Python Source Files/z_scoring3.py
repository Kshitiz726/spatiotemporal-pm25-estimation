import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load hourly merged data
df = pd.read_csv("final_cleaned_dataset.csv")
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

# Create cyclical encodings for hour and month
df['hour'] = df['datetime'].dt.hour
df['month'] = df['datetime'].dt.month
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Define lag hours and weather columns to lag
lag_hours = [1, 2, 3]
exclude_cols = ['datetime', 'pm25', 'hour', 'month', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
weather_cols = df.columns.difference(exclude_cols)

# Prepare all lagged features first, then concat once to avoid fragmentation
lag_dfs = []

for lag in lag_hours:
    # Lagged pm25
    pm25_lag = df['pm25'].shift(lag).rename(f'pm25_lag_{lag}')
    # Lagged weather features
    weather_lags = df[weather_cols].shift(lag).add_suffix(f'_lag_{lag}')
    # Combine for this lag
    lag_df = pd.concat([pm25_lag, weather_lags], axis=1)
    lag_dfs.append(lag_df)

# Concatenate all lagged features with original df
df = pd.concat([df] + lag_dfs, axis=1)

# Drop rows with NaNs created by lagging
df = df.dropna().reset_index(drop=True)

# Prepare features and target (exclude datetime and pm25)
feature_cols = df.columns.difference(['datetime', 'pm25'])
X = df[feature_cols].values
y = df['pm25'].values.reshape(-1, 1)

# Scale features and target
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Save scaled arrays
np.save("X_scaled_kathmandu_hourly_merra.npy", X_scaled)
np.save("y_scaled_kathmandu_hourly_merra.npy", y_scaled)

# Save scalers for inverse transform later
joblib.dump(scaler_X, "scaler_X_kathmandu_hourly_merra.pkl")
joblib.dump(scaler_y, "scaler_y_kathmandu_hourly_merra.pkl")

print("✅ Hourly lag features, cyclical encodings, and scaling prepared and saved.")
print(f"Features used: {list(feature_cols)}")
