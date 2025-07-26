import pandas as pd

# Load PM2.5 data
pm25 = pd.read_csv('clean_hourly_pm25_kathmandu_non_negative.csv', parse_dates=['time'])

# Load MERRA-2 weather data
merra = pd.read_csv('merra2_kathmandu_hourly_npt.csv', parse_dates=['time'])

# Build weather dataframe with first row blank and rest aligned with PM2.5
weather_to_merge = pd.concat([
    pd.DataFrame([{}]),  # Blank row for 5:00
    merra.iloc[:len(pm25) - 1].reset_index(drop=True)  # Start from MERRA's 6:15
], ignore_index=True)

# Drop 'time' column from weather data if it exists to avoid collision
weather_to_merge = weather_to_merge.drop(columns=['time'], errors='ignore')

# Merge PM2.5 and shifted weather data
result = pd.concat([pm25.reset_index(drop=True), weather_to_merge], axis=1)

# Save to CSV
result.to_csv('merged_shifted_output.csv', index=False)
