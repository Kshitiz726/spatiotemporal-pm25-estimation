import os
import xarray as xr
import pandas as pd

lat_kathmandu = 27.7172
lon_kathmandu = 85.3240

data_folder = "D:/Data_Mining_Project/Implementation/NASA_MERRA_DATA_"
output_csv = "merra_kathmandu_all_raw_filtered.csv"

nc_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(".nc4")]
nc_files.sort()

df_list = []
for idx, file in enumerate(nc_files, 1):
    print(f"[{idx}/{len(nc_files)}] Processing: {file}")
    try:
        ds = xr.open_dataset(file)
        
        # Ensure we get data, using nearest point
        ds_subset = ds.sel(
            lat=lat_kathmandu,
            lon=lon_kathmandu,
            method="nearest"
        )

        df = ds_subset.to_dataframe().reset_index()
        print(f"✅ Loaded shape: {df.shape}")

        # Only append if there is data
        if not df.empty:
            df_list.append(df)
        else:
            print(f"⚠️ Empty data for {file}")

    except Exception as e:
        print(f"❌ Skipping {file} due to error: {e}")
        continue

print("🪄 Concatenating...")
if df_list:
    full_df = pd.concat(df_list, ignore_index=True)
    if os.path.exists(output_csv):
        # Load existing data and append
        existing_df = pd.read_csv(output_csv)
        combined_df = pd.concat([existing_df, full_df], ignore_index=True)
        combined_df.to_csv(output_csv, index=False)
        print(f"✅ Appended and saved to {output_csv} with shape {combined_df.shape}")
    else:
        full_df.to_csv(output_csv, index=False)
        print(f"✅ Created {output_csv} with shape {full_df.shape}")
