import pandas as pd
from datetime import datetime, timedelta

# Load the NASA MERRA-2 CSV (no timestamp column)
df = pd.read_csv("merra_kathmandu_all_raw_filtered.csv")

# 1️⃣ Reconstruct UTC timestamps starting from 2017-03-03 00:30
start_time_utc = datetime(2017, 3, 3, 0, 30)
timestamps_utc = [start_time_utc + timedelta(hours=i) for i in range(len(df))]

# 2️⃣ Convert UTC → Nepal Time (UTC + 5:45)
timestamps_npt = [t + timedelta(hours=5, minutes=45) for t in timestamps_utc]

# 3️⃣ Add datetime column
df['time'] = timestamps_npt

# 4️⃣ Save to new CSV
df.to_csv("merra2_kathmandu_hourly_npt.csv", index=False)

print(f"✅ Added {len(df)} hourly timestamps in Nepal time.")
print("📁 Saved as 'merra2_kathmandu_hourly_npt.csv'")
