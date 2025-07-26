import pandas as pd

# Load your CSV
df = pd.read_csv("daily_pm25_kathmandu.csv", parse_dates=["date"])

# Replace negative values with NaN
pm_columns = ['pm25_phora', 'pm25_embassy', 'pm25_mean']
df[pm_columns] = df[pm_columns].applymap(lambda x: x if x >= 0 else None)

# Optional: Drop rows where all PM2.5 columns are NaN
df = df.dropna(subset=pm_columns, how='all')

# Save cleaned data
df.to_csv("cleaned_pm25.csv", index=False)

print("Cleaned data saved to cleaned_pm25.csv")
