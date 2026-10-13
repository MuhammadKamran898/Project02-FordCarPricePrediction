import logging
import os

import requests
import streamlit as st


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ford-price-ui")

API_BASE = os.getenv("API_URL", "http://127.0.0.1:8001").rstrip("/")
API_URL = f"{API_BASE}/predict"

st.set_page_config(
    page_title="Ford Price Predictor",
    page_icon="🚗",
    layout="centered",
)

st.title("🚗 Ford Car Price Predictor")
st.write("Enter the vehicle details below to estimate its market price.")
st.caption(
    "Estimates are based on the available training data, not a guaranteed sale price. "
    "Some vehicle categories have very few examples and may have less reliable estimates."
)

# Category options reflect the cleaned training dataset. Keep these aligned with schemas_improved.py.
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
    min_value=1996,
    max_value=2020,
    value=2017,
    step=1,
)

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic", "Semi-Auto"],
)

mileage = st.number_input(
    "Mileage (miles)",
    min_value=1,
    max_value=177644,
    value=15000,
    step=500,
)

fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "Hybrid", "Electric", "Other"],
)

tax = st.number_input(
    "Tax (£ per year)",
    min_value=0,
    max_value=580,
    value=145,
    step=5,
)

mpg = st.number_input(
    "MPG (UK miles per gallon)",
    min_value=20.8,
    max_value=201.8,
    value=58.9,
    step=0.1,
)

engine_size = st.number_input(
    "Engine Size (litres)",
    min_value=0.1,
    max_value=5.0,
    value=1.2,
    step=0.1,
)

if st.button("Predict Price", type="primary", use_container_width=True):
    payload = {
        "model": model,
        "year": int(year),
        "transmission": transmission,
        "mileage": int(mileage),
        "fuelType": fuel_type,
        "tax": int(tax),
        "mpg": float(mpg),
        "engineSize": float(engine_size),
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
            predicted_price = float(result["predicted_price"])

            st.success("Prediction completed!")
            st.metric(
                label="Estimated Ford Price",
                value=f"£{predicted_price:,.2f}",
            )

        elif response.status_code == 422:
            st.error(
                "Some vehicle details are unsupported or outside the "
                "training-data range. Please check the entered values."
            )
            try:
                details = response.json().get("detail", [])
                with st.expander("Validation details"):
                    st.json(details)
            except ValueError:
                pass

        elif response.status_code == 503:
            st.error(
                "The prediction service is not ready. Please try again later."
            )

        else:
            st.error(
                "The prediction service encountered an error. Please try again later."
            )

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the FastAPI server. Check that the backend is running "
            "and API_URL points to the correct host."
        )
    except requests.exceptions.Timeout:
        st.error("The prediction request timed out. Please try again.")
    except requests.exceptions.RequestException:
        logger.exception("HTTP request failed while predicting a Ford price.")
        st.error("The prediction request failed. Please try again later.")
    except (ValueError, KeyError):
        logger.exception("The prediction API returned an invalid response.")
        st.error("The prediction service returned an invalid response.")
    except Exception:
        logger.exception("Unexpected prediction UI error.")
        st.error("An unexpected error occurred while predicting the price.")
