import os
import joblib
import streamlit as st
import pandas as pd
from train_model import train

st.set_page_config(page_title="HomeValue AI", page_icon="🏡", layout="wide")
st.title("🏡 Real Estate Housing Price Prediction")
st.write("Estimate a California district's median house value using a Ridge/Lasso regression model.")
st.info("Educational demonstration using the California Housing dataset. It does not provide current property listing prices or professional valuations.")

@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), "housing_model.pkl")
    if not os.path.exists(path):
        with st.spinner("First-time setup: downloading the public dataset and training models..."):
            train()
    return joblib.load(path)

try:
    model = load_model()
except Exception as e:
    st.error("Model setup failed. Check your internet connection (the dataset is downloaded on first run) and run: python train_model.py")
    st.exception(e)
    st.stop()

st.subheader("Enter area and household details")
c1, c2, c3 = st.columns(3)
with c1:
    longitude = st.number_input("Longitude", min_value=-125.0, max_value=-113.0, value=-118.0, step=0.1)
    latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=34.0, step=0.1)
    housing_median_age = st.number_input("Housing median age (years)", min_value=1.0, max_value=60.0, value=25.0)
with c2:
    total_rooms = st.number_input("Total rooms", min_value=1.0, value=2000.0)
    total_bedrooms = st.number_input("Total bedrooms", min_value=1.0, value=400.0)
    population = st.number_input("Population", min_value=1.0, value=1000.0)
with c3:
    households = st.number_input("Households", min_value=1.0, value=500.0)
    median_income = st.number_input("Median income (tens of thousands USD)", min_value=0.1, value=4.0, step=0.1)
    ave_occup = st.number_input("Average occupancy", min_value=0.1, value=3.0, step=0.1)

if st.button("Predict House Value", type="primary"):
    ave_rooms = total_rooms / households
    ave_bedrms = total_bedrooms / households
    row = pd.DataFrame([{
        "MedInc": median_income, "HouseAge": housing_median_age,
        "AveRooms": ave_rooms, "AveBedrms": ave_bedrms,
        "Population": population, "AveOccup": ave_occup,
        "Latitude": latitude, "Longitude": longitude,
        "RoomsPerHousehold": ave_rooms / ave_occup,
        "BedroomsPerRoom": ave_bedrms / ave_rooms if ave_rooms else 0,
        "PopulationPerHousehold": population / ave_occup
    }])
    predicted = float(model.predict(row)[0])
    st.success(f"Estimated median house value: ${predicted * 100000:,.0f}")
    st.caption(f"Model output: {predicted:.3f} × $100,000 (dataset target units).")

results_path = os.path.join(os.path.dirname(__file__), "housing_model_results.csv")
if os.path.exists(results_path):
    st.subheader("Model comparison")
    st.dataframe(pd.read_csv(results_path).round(4), use_container_width=True)
