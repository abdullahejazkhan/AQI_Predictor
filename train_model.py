# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import matplotlib.pyplot as plt

# --- CONFIG ---
DATA_PATH = "data/processed_features.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "aqi_model.pkl")

# --- 1. LOAD DATA ---
df = pd.read_csv(DATA_PATH)

# Normalize column names (make lowercase)
df.columns = [c.lower() for c in df.columns]

# Define features and target
# Drop columns that aren’t numeric or useful
target_col = "aqi" if "aqi" in df.columns else "aqi" if "AQI" in df.columns else "aqi"
date_col = "date" if "date" in df.columns else "datetime"

# Clean
df = df.dropna()
X = df.drop(columns=[target_col, date_col])
y = df[target_col]

# --- 2. SPLIT DATA ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 3. TRAIN MODEL ---
model = RandomForestRegressor(
    n_estimators=150,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

joblib.dump(model, "models/aqi_model.pkl")


joblib.dump(X.columns.tolist(), "models/feature_names.pkl")

print("✅ Model and feature names saved to models/")

# --- 4. EVALUATE MODEL ---
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("✅ Model training complete.")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.2f}")

# --- 5. SAVE MODEL ---
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"💾 Model saved to {MODEL_PATH}")

# --- 6. OPTIONAL PLOT ---
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI")
plt.tight_layout()
plt.savefig("data/aqi_model_performance.png")
print("📊 Saved performance plot to data/aqi_model_performance.png")
