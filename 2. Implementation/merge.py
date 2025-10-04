import pandas as pd

# Load data
pm25_df = pd.read_csv("daily_pm25_kathmandu_imputed.csv")
weather_df = pd.read_csv("kathmandu_daily_climate.csv")

# Convert date columns to datetime
pm25_df['date'] = pd.to_datetime(pm25_df['date'])
weather_df['Date'] = pd.to_datetime(weather_df['Date'])

# Merge on date
df = pd.merge(pm25_df, weather_df, left_on='date', right_on='Date', how='inner')

# Drop redundant columns from weather
df.drop(columns=['Date', 'Unnamed: 0', 'District', 'Latitude', 'Longitude'], inplace=True)

# Save merged data to CSV
output_path = "merged_pm25_weather_kathmandu.csv"
df.to_csv(output_path, index=False)

print(f"Merged data saved to {output_path}")
