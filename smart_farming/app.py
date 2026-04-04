import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page setup
# -----------------------------

st.set_page_config(page_title="Smart Farming Advisor", layout="wide")

st.title("🌾 Smart Farming Fertilizer Advisor")
st.write("Select your farm conditions to get the best fertilizer recommendation.")

# -----------------------------
# Load model and encoders
# -----------------------------

model = joblib.load("fertilizer_model.pkl")
encoders = joblib.load("encoders.pkl")

df = pd.read_csv("fertilizer_dataset.csv")

# ---------------------------------
# Farmer inputs (simple)
# ---------------------------------

st.header("Step 1: Select Crop")

crop = st.selectbox(
    "Crop",
    df["Crop_Type"].unique()
)

st.header("Step 2: Soil Type")

soil = st.selectbox(
    "Soil Type",
    df["Soil_Type"].unique()
)

st.header("Step 3: Irrigation Type")

irrigation = st.selectbox(
    "Irrigation Type",
    df["Irrigation_Type"].unique()
)

st.header("Step 4: Region")

region = st.selectbox(
    "Region",
    df["Region"].unique()
)

# ---------------------------------
# Prediction
# ---------------------------------

if st.button("🌱 Get Best Fertilizer"):

    input_data = {}

    for col in df.columns:

        if col == "Recommended_Fertilizer":
            continue

        elif col == "Crop_Type":
            input_data[col] = crop

        elif col == "Soil_Type":
            input_data[col] = soil

        elif col == "Irrigation_Type":
            input_data[col] = irrigation

        elif col == "Region":
            input_data[col] = region

        # fill hidden columns automatically
        elif df[col].dtype == "object":
            input_data[col] = df[col].iloc[0]

        else:
            input_data[col] = df[col].mean()

    input_df = pd.DataFrame([input_data])

    # encode categorical columns
    for col in encoders:
        if col in input_df.columns:
            input_df[col] = encoders[col].transform(input_df[col])

    prediction = model.predict(input_df)

    # convert encoded prediction to fertilizer name
    fertilizer_name = encoders["Recommended_Fertilizer"].inverse_transform(prediction)

    st.success(f"🌱 Recommended Fertilizer: {fertilizer_name[0]}")


# ---------------------------------
# Farming tips
# ---------------------------------

st.header("🌱 Farming Tips")

tips = [
    "Maintain soil pH between 6 and 7",
    "Use balanced fertilizer for healthy crops",
    "Rotate crops every season",
    "Ensure proper irrigation for best yield"
]

for tip in tips:
    st.write("✔", tip)