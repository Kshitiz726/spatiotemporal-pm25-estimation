import pandas as pd
import numpy as np
import joblib

# Load your saved scalers and model
scaler_X = joblib.load("scaler_X_kathmandu_hourly_merra.pkl")
scaler_y = joblib.load("scaler_y_kathmandu_hourly_merra.pkl")
from tensorflow.keras.models import load_model
model = load_model("dnn_pm25_kathmandu_hourly_merra_fixed.keras")

# Load new data CSV (must include pm25 to compute lags)
df = pd.read_csv("final_cleaned_dataset.csv")
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

# Create cyclical time features
df['hour'] = df['datetime'].dt.hour
df['month'] = df['datetime'].dt.month
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Prepare lag features (lags 1,2,3) for pm25 and weather columns
lag_hours = [1, 2, 3]
exclude_cols = ['datetime', 'pm25', 'hour', 'month', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
weather_cols = df.columns.difference(exclude_cols)

lag_dfs = []
for lag in lag_hours:
    pm25_lag = df['pm25'].shift(lag).rename(f'pm25_lag_{lag}')
    weather_lags = df[weather_cols].shift(lag).add_suffix(f'_lag_{lag}')
    lag_df = pd.concat([pm25_lag, weather_lags], axis=1)
    lag_dfs.append(lag_df)

df = pd.concat([df] + lag_dfs, axis=1)
df = df.dropna().reset_index(drop=True)  # drop rows with NaN due to lagging

# Prepare final features (exclude datetime and pm25 target)
feature_cols = df.columns.difference(['datetime', 'pm25'])
X_new = df[feature_cols].values

# Scale input features with saved scaler
X_new_scaled = scaler_X.transform(X_new)

# Predict and inverse scale
y_pred_scaled = model.predict(X_new_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Save predictions back to dataframe or CSV
df['pm25_predicted'] = np.nan
df.loc[df.index, 'pm25_predicted'] = y_pred.flatten()

df.to_csv("pm25_reconstruction_output.csv", index=False)
print("✅ PM2.5 reconstructed predictions saved.")
