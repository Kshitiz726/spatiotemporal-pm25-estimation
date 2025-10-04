import pandas as pd

# 1️⃣ Load your dataset
input_path = "dailyclimate.csv"  # replace with your actual path
df = pd.read_csv(input_path)

# 2️⃣ Filter rows where District == "Kathmandu"
df_kathmandu = df[df["District"].str.strip().str.lower() == "kathmandu"]

# 3️⃣ Save to a new CSV
output_path = "kathmandu_daily_climate.csv"
df_kathmandu.to_csv(output_path, index=False)

print(f"✅ Extracted {len(df_kathmandu)} rows for Kathmandu and saved to {output_path}")
