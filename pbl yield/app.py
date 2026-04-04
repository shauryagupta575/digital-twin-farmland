import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load saved model
model = joblib.load("model_output/best_model.pkl")
le    = joblib.load("model_output/label_encoder.pkl")

st.title("🌾 Crop Yield Prediction")

# Inputs
year = st.number_input("Year", 2000, 2100, 2024)
area = st.number_input("Area (ha)", 1.0)
N    = st.number_input("Nitrogen (kg/ha)", 0.0)
P    = st.number_input("Phosphorus (kg/ha)", 0.0)
K    = st.number_input("Potassium (kg/ha)", 0.0)
temp = st.number_input("Temperature (°C)", 0.0)
hum  = st.number_input("Humidity (%)", 0.0)
ph   = st.number_input("pH", 0.0)
rain = st.number_input("Rainfall (mm)", 0.0)
wind = st.number_input("Wind Speed (m/s)", 0.0)
solar= st.number_input("Solar Radiation", 0.0)

crop = st.selectbox("Crop", le.classes_)

# Predict
if st.button("Predict Yield"):
    crop_enc = le.transform([crop])[0]
    
    data = pd.DataFrame([[year, area, N, P, K,
                      temp, hum, ph, rain,
                      wind, solar, crop_enc]],
                    columns=['Year', 'Area_ha',
                             'N_req_kg_per_ha', 'P_req_kg_per_ha', 'K_req_kg_per_ha',
                             'Temperature_C', 'Humidity_%', 'pH',
                             'Rainfall_mm', 'Wind_Speed_m_s', 'Solar_Radiation_MJ_m2_day',
                             'Crop_enc'])

    prediction = model.predict(data)[0]

    st.success(f"🌱 Predicted Yield: {prediction:.2f} kg/ha")