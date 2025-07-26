import pandas as pd
import numpy as np

# File paths - update as needed
phora_path = "D:/Data_Mining_Project/Data Sets/aq_kathmandu_us_diplomatic_post_phora_durbar_kathmandu.csv"
embassy_path = "D:/Data_Mining_Project/Data Sets/aq_kathmandu_us-diplomatic-post_embassy_kathmandu.csv"

# Load datasets
df_phora = pd.read_csv(phora_path)
df_embassy = pd.read_csv(embassy_path)

# Filter PM2.5 only
df_phora_pm25 = df_phora[df_phora["parameter"] == "pm25"].copy()
df_embassy_pm25 = df_embassy[df_embassy["parameter"] == "pm25"].copy()

# Parse timestamp and remove timezone info (make naive datetime)
df_phora_pm25["local_time"] = pd.to_datetime(df_phora_pm25["local"]).dt.tz_convert(None)
df_embassy_pm25["local_time"] = pd.to_datetime(df_embassy_pm25["local"]).dt.tz_convert(None)

# Set datetime as index for resampling
df_phora_pm25.set_index("local_time", inplace=True)
df_embassy_pm25.set_index("local_time", inplace=True)

# Resample to daily mean PM2.5
df_phora_daily = df_phora_pm25["value"].resample("D").mean().reset_index()
df_embassy_daily = df_embassy_pm25["value"].resample("D").mean().reset_index()

# Rename columns for clarity
df_phora_daily.rename(columns={"local_time": "date", "value": "pm25_phora"}, inplace=True)
df_embassy_daily.rename(columns={"local_time": "date", "value": "pm25_embassy"}, inplace=True)

# Merge both daily datasets on date (outer join to keep all dates)
df_daily_merged = pd.merge(df_phora_daily, df_embassy_daily, on="date", how="outer")

# Replace negative PM2.5 values with NaN (invalid data)
df_daily_merged.loc[df_daily_merged["pm25_phora"] < 0, "pm25_phora"] = np.nan
df_daily_merged.loc[df_daily_merged["pm25_embassy"] < 0, "pm25_embassy"] = np.nan

# Calculate mean PM2.5 from both stations, ignoring NaNs
df_daily_merged["pm25_mean"] = df_daily_merged[["pm25_phora", "pm25_embassy"]].mean(axis=1)

# Replace negative means with NaN too
df_daily_merged.loc[df_daily_merged["pm25_mean"] < 0, "pm25_mean"] = np.nan

# Sort by date
df_daily_merged.sort_values("date", inplace=True)

# Set 'date' as datetime index for time interpolation
df_daily_merged["date"] = pd.to_datetime(df_daily_merged["date"])
df_daily_merged.set_index("date", inplace=True)

# Interpolate missing PM2.5 values using time method
df_daily_merged["pm25_phora"] = df_daily_merged["pm25_phora"].interpolate(method='time')
df_daily_merged["pm25_embassy"] = df_daily_merged["pm25_embassy"].interpolate(method='time')
df_daily_merged["pm25_mean"] = df_daily_merged["pm25_mean"].interpolate(method='time')

# Forward-fill and backward-fill any remaining NaNs at start/end
df_daily_merged.fillna(method='ffill', inplace=True)
df_daily_merged.fillna(method='bfill', inplace=True)

# Reset index so 'date' becomes a column again
df_daily_merged.reset_index(inplace=True)

# Save cleaned daily PM2.5 data to CSV
output_path = "daily_pm25_kathmandu_imputed.csv"
df_daily_merged.to_csv(output_path, index=False)

print(f"Cleaned daily PM2.5 data saved to {output_path}")
