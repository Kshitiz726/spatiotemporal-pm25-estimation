# Ready-to-run Python script to download 2017–2021 Kathmandu hourly PM2.5 and related air quality data from Open-Meteo, save year-wise CSVs, and concatenate for ML ingestion.

import requests
import pandas as pd
from time import sleep

latitude = 27.7172
longitude = 85.3240

hourly_vars = "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,european_aqi,us_aqi"

base_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

years = [2017, 2018, 2019, 2020, 2021]

all_dfs = []

for year in years:
    print(f"Downloading PM2.5 and AQ data for {year}...")
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": hourly_vars,
        "timezone": "auto"
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame(data['hourly'])
        df['time'] = pd.to_datetime(df['time'])
        df.insert(0, 'year', year)

        filename = f"kathmandu_hourly_pm25_aq_{year}.csv"
        df.to_csv(filename, index=False)
        print(f"Saved {filename}")

        all_dfs.append(df)

        sleep(1)

    except Exception as e:
        print(f"Failed for {year}: {e}")

combined_df = pd.concat(all_dfs, ignore_index=True)
combined_df.to_csv("kathmandu_hourly_pm25_aq_2017_2021.csv", index=False)
print("Saved combined CSV: kathmandu_hourly_pm25_aq_2017_2021.csv")
