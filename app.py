# ---------------------------------------------------
# Islamabad AQI Prediction Dashboard — Final Stable Version
# ---------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from fpdf import FPDF
import plotly.graph_objects as go
import shap
import os
import matplotlib.pyplot as plt

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(
    page_title="Islamabad AQI Forecast",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Helper Functions
# -----------------------------
@st.cache_data
def load_model(path="models/aqi_model.pkl"):
    return joblib.load(path)

@st.cache_data
def load_feature_names(path="models/feature_names.pkl"):
    return joblib.load(path)

@st.cache_data
def load_data(path="data/processed_features.csv", nrows=None):
    df = pd.read_csv(path, nrows=nrows)
    df.columns = [c.lower() for c in df.columns]
    if "datetime" not in df.columns:
        for c in df.columns:
            if "date" in c or "time" in c:
                df["datetime"] = pd.to_datetime(df[c])
                break
    else:
        df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    return df

def aqi_category(aqi):
    if aqi <= 50: return "Good", "green"
    if aqi <= 100: return "Moderate", "yellow"
    if aqi <= 200: return "Unhealthy for Sensitive", "orange"
    if aqi <= 300: return "Unhealthy", "red"
    if aqi <= 400: return "Very Unhealthy", "#7e0023"
    return "Hazardous", "#7e0023"

def forecast_autoregressive(model, feature_names, history_df, steps=72):
    hist = history_df.copy().reset_index(drop=True)
    hist.columns = [c.lower() for c in hist.columns]
    numeric_cols = hist.select_dtypes(include=[np.number]).columns.tolist()
    if "aqi" in numeric_cols:
        numeric_cols.remove("aqi")

    out = []
    for i in range(steps):
        last = hist.iloc[-1:]
        dt = pd.to_datetime(last["datetime"].values[0]) + timedelta(hours=1)

        next_feat = {
            "hour": dt.hour,
            "day": dt.day,
            "month": dt.month,
            "weekday": dt.weekday(),
            "aqi_lag1": hist["aqi"].iloc[-1],
            "aqi_lag2": hist["aqi"].iloc[-2] if len(hist) > 1 else hist["aqi"].iloc[-1],
            "aqi_roll3": hist["aqi"].tail(3).mean(),
            "aqi_roll7": hist["aqi"].tail(7).mean(),
        }

        for c in numeric_cols:
            if c not in next_feat:
                next_feat[c] = last[c].values[0] if c in last.columns else 0.0

        x_next = pd.DataFrame([next_feat]).reindex(columns=feature_names, fill_value=0)
        y_pred = model.predict(x_next)[0]

        new_row = next_feat.copy()
        new_row["aqi"] = y_pred
        new_row["datetime"] = dt
        hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
        out.append({"datetime": dt, "predicted_aqi": float(y_pred)})

    forecast_df = pd.DataFrame(out)
    # clean to avoid frontend JSON errors
    forecast_df = forecast_df.replace([np.inf, -np.inf], np.nan).dropna()
    forecast_df["datetime"] = pd.to_datetime(forecast_df["datetime"], errors="coerce")
    forecast_df = forecast_df.dropna(subset=["datetime"])
    return forecast_df

def generate_report(aqi, forecast_df):
    os.makedirs("data", exist_ok=True)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Islamabad AQI Prediction Report", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Generated on: {datetime.now():%Y-%m-%d %H:%M}", ln=True)
    pdf.cell(0, 10, f"Current AQI: {aqi:.1f}", ln=True)
    pdf.cell(0, 10, "3-Day Forecast (first 10 rows):", ln=True)
    pdf.set_font("Arial", "", 10)
    for _, row in forecast_df.head(10).iterrows():
        pdf.cell(0, 8, f"{row['datetime']}  →  AQI {row['predicted_aqi']:.1f}", ln=True)
    pdf.output("data/aqi_report.pdf")

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.title("⚙️ Settings")
steps = st.sidebar.slider("Forecast horizon (hours)", 24, 72, 72, 24)
show_shap = st.sidebar.checkbox("Compute SHAP feature importance (slow)", False)

# -----------------------------
# Load Data & Model
# -----------------------------
model = load_model()
feature_names = load_feature_names()
df = load_data()

# -----------------------------
# Header and KPIs
# -----------------------------
st.title("🌤️ Islamabad Air Quality Forecast Dashboard")
st.markdown("Get **real-time AQI readings** and **3-day forecasts** powered by Machine Learning.")
st.markdown("---")

last_row = df.iloc[-1]
last_aqi = float(last_row["aqi"])
cat, color = aqi_category(last_aqi)
avg24 = df.tail(24)["aqi"].mean()

col1, col2, col3, col4 = st.columns([1.5, 1.2, 1.2, 1.2])
with col1:
    st.metric("Latest Observed AQI", f"{last_aqi:.1f}")
    st.markdown(f"<div style='font-size:13px;color:{color};font-weight:600'>{cat}</div>", unsafe_allow_html=True)
with col2:
    st.metric("24-Hour Avg AQI", f"{avg24:.1f}")
with col3:
    st.metric("Data Rows Used", f"{len(df):,}")
with col4:
    st.metric("Model", "RandomForest")

st.markdown("---")

# -----------------------------
# 3-Day Forecast
# -----------------------------
st.subheader("📈 3-Day (72-Hour) AQI Forecast")
history = df.tail(72)
forecast_df = forecast_autoregressive(model, feature_names, history, steps=steps)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=history["datetime"], y=history["aqi"],
    mode="lines+markers", name="Observed AQI", line=dict(color="green")
))
fig.add_trace(go.Scatter(
    x=forecast_df["datetime"], y=forecast_df["predicted_aqi"],
    mode="lines+markers", name="Forecasted AQI", line=dict(color="orange", dash="dot")
))
fig.update_layout(
    title="📆 3-Day AQI Forecast — Islamabad",
    xaxis_title="Datetime",
    yaxis_title="AQI",
    height=450,
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Forecast Tables
# -----------------------------
st.markdown("### Detailed 3-Day Forecast")
ft = forecast_df.copy()
ft["category"] = ft["predicted_aqi"].apply(lambda x: aqi_category(x)[0])
st.dataframe(ft, use_container_width=True)

st.markdown("### Daily Average Forecast")
ft["date"] = ft["datetime"].dt.date
daily = ft.groupby("date")["predicted_aqi"].mean().reset_index()
daily.columns = ["Date", "Average AQI"]
st.table(daily)

hazardous = ft[ft["predicted_aqi"] >= 300]
if not hazardous.empty:
    st.error(f"⚠️  {len(hazardous)} forecasted hour(s) predicted to be VERY UNHEALTHY or worse.")

st.markdown("---")

# -----------------------------
# Feature Importance (Optional)
# -----------------------------
st.subheader("Feature Importance (Optional)")
if show_shap:
    st.info("Computing SHAP values — this may take ~30s depending on your system speed.")
    X = df.drop(columns=["aqi", "datetime"]).tail(500)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig("data/shap_summary.png", dpi=150, bbox_inches="tight")
    st.image("data/shap_summary.png", use_container_width=True)
else:
    st.write("Toggle SHAP in the sidebar to compute feature importance (optional).")

st.markdown("---")

# -----------------------------
# Report Download
# -----------------------------
if st.button("📄 Generate & Download AQI Report"):
    generate_report(last_aqi, forecast_df)
    with open("data/aqi_report.pdf", "rb") as file:
        st.download_button("Download Report", file, file_name="AQI_Report.pdf")

st.caption("Built with Streamlit • Model: models/aqi_model.pkl • Data Source: processed local dataset")
