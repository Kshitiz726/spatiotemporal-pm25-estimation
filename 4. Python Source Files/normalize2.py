import pandas as pd
import numpy as np

# Load PM2.5 data
pm25_df = pd.read_csv("clean_hourly_pm25_kathmandu.csv")
pm25_df['datetime'] = pd.to_datetime(pm25_df['datetime'])  # keep original timezone and time as-is

# Replace negative PM2.5 values with NaN
pm25_df['pm25'] = pm25_df['pm25'].where(pm25_df['pm25'] >= 0, np.nan)

# Interpolate only on existing timestamps (no new timestamps created)
pm25_df['pm25'] = pm25_df['pm25'].interpolate(method='linear', limit_direction='both')

# After interpolation, ensure non-negative
pm25_df['pm25'] = pm25_df['pm25'].clip(lower=0)

# Save cleaned PM2.5 back
pm25_df.to_csv("clean_hourly_pm25_kathmandu_non_negative.csv", index=False)

print("✅ PM2.5 cleaned: non-null, non-negative, time kept unchanged.")
