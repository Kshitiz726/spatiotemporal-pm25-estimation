import pandas as pd

# Load PM2.5 data
pm25_df = pd.read_csv("clean_hourly_pm25_kathmandu_non_negative.csv")
pm25_df['datetime'] = pd.to_datetime(pm25_df['datetime'])
# Remove timezone to match weather_df
pm25_df['datetime'] = pm25_df['datetime'].dt.tz_localize(None)

# Load weather data
weather_df = pd.read_csv("kathmandu_hourly_weather_2017_2021.csv")
weather_df['time'] = pd.to_datetime(weather_df['time'], format="%Y-%m-%d %H:%M:%S")

# Merge on datetime (local time alignment)
merged_df = pd.merge(pm25_df, weather_df, left_on='datetime', right_on='time', how='inner')

# Drop redundant 'time' column (retain 'year' if needed)
merged_df = merged_df.drop(columns=['time'])

# Save merged dataset for model training
merged_df.to_csv("merged_pm25_weather_kathmandu_hourly.csv", index=False)

print("✅ Merged hourly PM2.5 and weather data saved as 'merged_pm25_weather_kathmandu_hourly.csv' and ready for ingestion.")
