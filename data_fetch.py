import requests
import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np

# ---------------------------
# CONFIGURATION
# ---------------------------
API_KEY = "a85ea6bf7dcce14ef6c3531feb46d535"  # 👈 paste your key here in quotes
LAT, LON = 33.6844, 73.0479  # Islamabad coordinates
API_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
HISTORICAL_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"

os.makedirs("data", exist_ok=True)

# ---------------------------
# FETCH AIR QUALITY DATA
# ---------------------------
def fetch_aqi_data(hours=4500):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=hours)

    print(f"⏳ Fetching air quality data for Islamabad from {start_time} to {end_time} ...")

    # Convert to UNIX timestamps (required by API)
    start_ts = int(start_time.timestamp())
    end_ts = int(end_time.timestamp())

    params = {
        "lat": LAT,
        "lon": LON,
        "start": start_ts,
        "end": end_ts,
        "appid": API_KEY
    }

    response = requests.get(HISTORICAL_URL, params=params)

    if response.status_code != 200:
        print(f"❌ API request failed: {response.status_code}")
        print(response.text)
        return None

    data = response.json()
    if "list" not in data or len(data["list"]) == 0:
        print("⚠️ No data found from OpenWeatherMap API.")
        return None

    # Parse API response
    rows = []
    for entry in data["list"]:
        ts = datetime.utcfromtimestamp(entry["dt"])
        main = entry["main"]
        comps = entry["components"]

        rows.append({
            "datetime": ts,
            "aqi": main["aqi"],  # AQI index from 1 (good) to 5 (very poor)
            "co": comps.get("co", np.nan),
            "no": comps.get("no", np.nan),
            "no2": comps.get("no2", np.nan),
            "o3": comps.get("o3", np.nan),
            "so2": comps.get("so2", np.nan),
            "pm2_5": comps.get("pm2_5", np.nan),
            "pm10": comps.get("pm10", np.nan),
            "nh3": comps.get("nh3", np.nan)
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("datetime")

    df.to_csv("data/aqi_latest.csv", index=False)
    print(f"✅ Fetched {len(df)} rows. Saved to data/aqi_latest.csv.")
    return df


if __name__ == "__main__":
    fetch_aqi_data(hours=4500)
