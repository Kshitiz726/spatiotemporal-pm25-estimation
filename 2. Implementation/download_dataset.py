import os
import requests

# === USER CONFIG ===
token = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6Imtld3JpZXBpZTIwMDMiLCJleHAiOjE3NTY4MTg1NTQsImlhdCI6MTc1MTYzNDU1NCwiaXNzIjoiaHR0cHM6Ly91cnMuZWFydGhkYXRhLm5hc2EuZ292IiwiaWRlbnRpdHlfcHJvdmlkZXIiOiJlZGxfb3BzIiwiYWNyIjoiZWRsIiwiYXNzdXJhbmNlX2xldmVsIjozfQ.y9SLNsnuiEFB3byBTLZmxreZTAnSXkv9hH3LcUzg8tUA6pQc7Qsna6AwCMC6wD9g7iLmKTEIHwgYQm_U7iQNlebTQAXYIGpdEq-goRB4e7REw3PNHb5fuwBRl3ujdhji3OFEvqnn0uBitxzHK6_InPsUy9wnMvHS7Y6xbSFTbpWR-AoKoE_-L01J204GDdK_pKAmhMCexlIX2bFpF6eM8Q4qlL5tckmUrHZ0o_0-LBqIFHrwf1avCCm-42pLGEEDwF93T8HcXeaXgXyX5OMdu-qXI9130zV6jqUbv0SwMe7KhrdRHQnpk2b51j8zOYl4bPSKO9p4C9QPJALDyfD5Hw"  # Replace with your Bearer token
url_file = "urls1.txt"       # Your .txt file with .nc4 links
output_dir = "NASA_MERRA_DATA_"  # Output folder to save downloads
# ===================

# Make sure output folder exists
os.makedirs(output_dir, exist_ok=True)

# Prepare headers for authentication
headers = {"Authorization": f"Bearer {token}"}

# Read URLs
with open(url_file, 'r') as f:
    urls = [line.strip() for line in f if line.strip()]

# Download loop
for url in urls:
    filename = url.split("/")[-1]
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath):
        print(f"✅ {filename} already exists. Skipping.")
        continue

    print(f"⬇️ Downloading {filename}...")
    try:
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()

        with open(filepath, 'wb') as f_out:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f_out.write(chunk)

        print(f"✅ Downloaded {filename}\n")

    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}\n")
