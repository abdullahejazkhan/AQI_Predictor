# feature_engineering.py

import pandas as pd
import numpy as np
import os

def create_features(input_path="data/aqi_latest.csv", output_path="data/processed_features.csv"):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    print(f"📂 Loaded {len(df)} rows from {input_path}")

    # --- CLEAN & PREPARE ---
    # Ensure datetime column exists
    if "datetime" not in df.columns:
        raise KeyError("❌ 'datetime' column missing in dataset")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    # Ensure target column 'aqi' exists
    if "aqi" not in df.columns:
        raise KeyError("❌ 'aqi' column missing. Expected OpenWeatherMap data format.")

    # --- FEATURE ENGINEERING ---

    # Time-based features
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["weekday"] = df["datetime"].dt.weekday

    # Lag features (previous AQI values)
    df["aqi_lag1"] = df["aqi"].shift(1)
    df["aqi_lag2"] = df["aqi"].shift(2)

    # Rolling average features
    df["aqi_roll3"] = df["aqi"].rolling(window=3).mean()
    df["aqi_roll7"] = df["aqi"].rolling(window=7).mean()

    # Drop missing values from lags
    df = df.dropna()

    # --- OPTIONAL: Handle pollutant columns (fill missing values) ---
    pollutant_cols = ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
    for col in pollutant_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    # --- FINALIZE ---
    df = df.sort_values("datetime")
    df.to_csv(output_path, index=False)
    print(f"✅ Processed {len(df)} rows and saved to {output_path}")

if __name__ == "__main__":
    create_features()
