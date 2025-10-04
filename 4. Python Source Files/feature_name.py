import pandas as pd
import numpy as np

df = pd.read_csv("merged_pm25_weather_kathmandu_hourly.csv")

exclude_cols = ['datetime', 'pm25', 'hour', 'month', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
weather_cols = list(df.columns.difference(exclude_cols))

lag_hours = [1, 2, 3]

feature_cols = weather_cols.copy()

for lag in lag_hours:
    feature_cols.append(f'pm25_lag_{lag}')
    feature_cols.extend([f"{col}_lag_{lag}" for col in weather_cols])

print(f"Total features: {len(feature_cols)}")

# Your top indices:
top_indices = [21, 37, 35, 57, 62, 49, 47, 46, 25, 62]

# Map indices to feature names
top_feature_names = [feature_cols[i] for i in top_indices]

print("Top 10 Features by name:")
for i, name in enumerate(top_feature_names, 1):
    print(f"{i}. {name}")
