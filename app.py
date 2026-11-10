import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date
st.set_page_config(page_title="Bike Rental Prediction",page_icon="🚲",layout="centered")
st.title("🚲 Bike Rental Demand Prediction")
st.write("Enter the date, time and weather conditions "
    "to predict the number of bike rentals.")
artifact = joblib.load("bike_rental_model.joblib")
model = artifact["model"]
features = artifact["feature_columns"]
categories = artifact["categories"]
st.subheader("Enter Prediction Details")
with st.form("bike_prediction_form"):
    selected_date = st.date_input("Select Date",value=date(2012, 8, 7))
    hour = st.slider("Select Hour",min_value=0,max_value=23,value=8)
    season = st.selectbox("Season",categories["season"])
    holiday = st.selectbox("Holiday",categories["holiday"])
    workingday = st.selectbox("Working Day",categories["workingday"])
    weather = st.selectbox("Weather Situation",categories["weathersit"])
    temp = st.slider("Temperature",0.0, 1.0, 0.5)
    atemp = st.slider("Feeling Temperature",0.0, 1.0, 0.5)
    hum = st.slider("Humidity",0.0, 1.0, 0.6)
    windspeed = st.slider("Wind Speed",0.0, 1.0, 0.2)
    submitted = st.form_submit_button("Predict Bike Rentals")
if submitted:
    dt = pd.Timestamp(selected_date)
    month = dt.month
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    input_data = pd.DataFrame([{
        "season": season,
        "holiday": holiday,
        "workingday": workingday,
        "weathersit": weather,
        "yr": dt.year,
        "mnth": month,
        "day": dt.day,
        "dayofweek": dt.dayofweek,
        "hr": hour,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "temp": temp,
        "atemp": atemp,
        "hum": hum,
        "windspeed": windspeed}])
    input_data = input_data[features]
    prediction = float(model.predict(input_data)[0])
    prediction = max(0, prediction)
    st.success("Prediction completed!")
    st.metric("Predicted Hourly Bike Rentals", f"{prediction:,.0f} bikes")
    st.write("Selected Model:", artifact["model_name"])
    st.info("This is an estimate based on historical patterns "
        "learned by the trained machine learning model.")