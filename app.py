
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(
    page_title="Bike Rental Demand Predictor",
    page_icon="🚲",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM STYLING
# -------------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #101820;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1150px;
}
.hero {
    padding: 25px;
    border-radius: 16px;
    background: linear-gradient(120deg, #12372A, #205B45);
    margin-bottom: 25px;
}
.hero h1 {
    color: white;
    font-size: 36px;
}
.hero p {
    color: #E0F2E9;
    font-size: 17px;
}
.section-title {
    font-size: 22px;
    font-weight: 700;
    margin-top: 15px;
    margin-bottom: 12px;
}
.result-card {
    background: linear-gradient(120deg, #174B37, #237A55);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    color: white;
}
.result-number {
    font-size: 48px;
    font-weight: bold;
}
.small-note {
    color: #AAB7B8;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_artifact():
    return joblib.load("bike_rental_model.joblib")

artifact = load_artifact()

model = artifact["model"]
features = artifact["feature_columns"]
categories = artifact["categories"]

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🚲 Bike Rental Demand Prediction</h1>
    <p>
    Predict hourly bike rental demand using date,
    time and weather conditions with Machine Learning.
    </p>
</div>
""", unsafe_allow_html=True)

st.info(
    "Enter the conditions below to estimate the number "
    "of bikes that may be rented in one hour."
)

# -------------------------------------------------
# INPUT FORM
# -------------------------------------------------
with st.form("bike_prediction_form"):

    st.markdown(
        '<div class="section-title">📅 Date & Time</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        selected_date = st.date_input(
            "Select Date",
            value=date(2012, 8, 7),
            min_value=date(2011, 1, 1),
            max_value=date(2012, 12, 31)
        )

    with col2:
        hour = st.slider(
            "Select Hour",
            min_value=0,
            max_value=23,
            value=8,
            format="%d:00"
        )

    st.divider()

    st.markdown(
        '<div class="section-title">🌤️ Weather Conditions</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        season = st.selectbox(
            "Season",
            categories["season"]
        )

        holiday = st.selectbox(
            "Holiday",
            categories["holiday"]
        )

    with col2:
        workingday = st.selectbox(
            "Working Day",
            categories["workingday"]
        )

        weather = st.selectbox(
            "Weather Situation",
            categories["weathersit"]
        )

    st.divider()

    st.markdown(
        '<div class="section-title">🌡️ Environmental Measurements</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Enter actual measurements. The application "
        "automatically converts them into the normalized "
        "values required by the machine learning model."
    )

    col1, col2 = st.columns(2)

    with col1:
        temp_actual = st.slider(
            "Temperature (°C)",
            min_value=0.0,
            max_value=41.0,
            value=20.5,
            step=0.5
        )

        humidity_actual = st.slider(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            step=1.0
        )

    with col2:
        atemp_actual = st.slider(
            "Feeling Temperature (°C)",
            min_value=0.0,
            max_value=50.0,
            value=25.0,
            step=0.5
        )

        windspeed_actual = st.slider(
            "Wind Speed (dataset scale)",
            min_value=0.0,
            max_value=67.0,
            value=13.4,
            step=0.5
        )

    st.caption(
        "Wind speed uses the original dataset scale (0–67), "
        "before normalization."
    )

    st.divider()

    submitted = st.form_submit_button(
        "🚀 Predict Bike Rentals",
        use_container_width=True
    )

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
if submitted:

    dt = pd.Timestamp(selected_date)

    month = dt.month

    # Convert actual measurements to normalized values
    temp = temp_actual / 41.0
    atemp = atemp_actual / 50.0
    hum = humidity_actual / 100.0
    windspeed = windspeed_actual / 67.0

    # Cyclical feature engineering
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    # Dataset year encoding: 2011 = 0, 2012 = 1
    year_encoded = dt.year - 2011

    input_data = pd.DataFrame([{
        "season": season,
        "holiday": holiday,
        "workingday": workingday,
        "weathersit": weather,
        "yr": year_encoded,
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
        "windspeed": windspeed
    }])

    # Maintain original training feature order
    input_data = input_data[features]

    prediction = float(model.predict(input_data)[0])
    prediction = max(0, prediction)

    # -------------------------------------------------
    # DISPLAY RESULT
    # -------------------------------------------------
    st.divider()

    st.markdown(
        '<div class="section-title">📊 Prediction Results</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="result-card">
            <p>Estimated Hourly Bike Rentals</p>
            <div class="result-number">
                {prediction:,.0f}
            </div>
            <p>bikes per hour</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Selected Hour", f"{hour:02d}:00")

    with col2:
        st.metric("Temperature", f"{temp_actual:.1f} °C")

    with col3:
        st.metric("Humidity", f"{humidity_actual:.0f}%")

    st.success(
        "Prediction generated successfully using your trained model!"
    )

    st.caption(
        f"Machine Learning Model: {artifact['model_name']}"
    )

    with st.expander("View normalized model inputs"):
        st.dataframe(input_data, use_container_width=True)

    st.info(
        "Note: This prediction is an estimate based on "
        "historical bike rental patterns. Actual demand "
        "may differ."
    )

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.divider()

st.markdown(
    """
    <div style="text-align:center; color:gray;">
        Bike Rental Demand Prediction |
        Machine Learning Regression Project
    </div>
    """,
    unsafe_allow_html=True
)
