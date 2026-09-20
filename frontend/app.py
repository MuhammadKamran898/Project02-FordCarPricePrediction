import requests
import streamlit as st
import os

API_BASE = os.getenv("API_URL", "http://127.0.0.1:8001")
API_URL = f"{API_BASE}/predict"


st.set_page_config(
    page_title="Ford Price Predictor",
    page_icon="🚗",
    layout="centered",
)


st.title("🚗 Ford Car Price Predictor")
st.write("Enter the vehicle details below to estimate its market price.")


# --------------------------------------------------
# Input fields
# --------------------------------------------------

model = st.selectbox(
    "Ford Model",
    [
        "Fiesta",
        "Focus",
        "Kuga",
        "Mondeo",
        "EcoSport",
        "C-MAX",
        "B-MAX",
        "Galaxy",
        "S-MAX",
        "Edge",
        "Puma",
        "Ranger",
        "Mustang",
        "Ka+",
        "KA",
        "Fusion",
        "Escort",
        "Streetka",
        "Tourneo Connect",
        "Grand C-MAX",
        "Grand Tourneo Connect",
        "Tourneo Custom",
        "Transit Tourneo",
    ],
)

year = st.number_input(
    "Year",
    min_value=1990,
    max_value=2026,
    value=2017,
    step=1,
)

transmission = st.selectbox(
    "Transmission",
    [
        "Manual",
        "Automatic",
        "Semi-Auto",
    ],
)

mileage = st.number_input(
    "Mileage",
    min_value=0,
    max_value=200000,
    value=15000,
    step=500,
)

fuel_type = st.selectbox(
    "Fuel Type",
    [
        "Petrol",
        "Diesel",
        "Hybrid",
        "Electric",
        "Other",
    ],
)

tax = st.number_input(
    "Tax",
    min_value=0,
    max_value=1000,
    value=150,
    step=5,
)

mpg = st.number_input(
    "MPG",
    min_value=1.0,
    max_value=250.0,
    value=50.0,
    step=0.1,
)

engine_size = st.number_input(
    "Engine Size",
    min_value=0.1,
    max_value=10.0,
    value=1.5,
    step=0.1,
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button(
    "Predict Price",
    type="primary",
    use_container_width=True,
):
    payload = {
        "model": model,
        "year": year,
        "transmission": transmission,
        "mileage": mileage,
        "fuelType": fuel_type,
        "tax": tax,
        "mpg": mpg,
        "engineSize": engine_size,
    }

    try:
        with st.spinner("Predicting vehicle price..."):
            response = requests.post(
                API_URL,
                json=payload,
                timeout=10,
            )

        if response.status_code == 200:
            result = response.json()

            predicted_price = result["predicted_price"]

            st.success("Prediction completed!")

            st.metric(
                label="Estimated Ford Price",
                value=f"£{predicted_price:,.0f}",
            )

        else:
            st.error(f"API error: {response.status_code}")

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the FastAPI server. Make sure the backend is running."
        )

    except requests.exceptions.Timeout:
        st.error("The prediction request timed out.")

    except Exception as exc:  # noqa: BLE001
        st.error(f"Unexpected error: {exc}")
