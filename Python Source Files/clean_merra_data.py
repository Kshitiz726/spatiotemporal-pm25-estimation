import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("merged_shifted_output.csv", parse_dates=['time'])

# Show missing values before handling
print("Missing values before interpolation:\n", df.isnull().sum())

# Sort by time to ensure proper time-based interpolation
df = df.sort_values(by='time')

# Interpolate all numeric columns (time-aware if index is datetime)
df.interpolate(method='linear', inplace=True)

# Optional: Forward fill as a backup for any beginning NaNs
df.fillna(method='ffill', inplace=True)

# Optional: Drop any remaining rows with missing values
df.dropna(inplace=True)

# Show missing values after interpolation
print("\nMissing values after interpolation:\n", df.isnull().sum())

# Save the cleaned data
df.to_csv("final_cleaned_dataset.csv", index=False)
print("\n✅ Cleaned dataset saved as 'final_cleaned_dataset.csv'")
