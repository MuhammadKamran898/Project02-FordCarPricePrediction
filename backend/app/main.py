from fastapi import FastAPI, HTTPException

from .model_service import FordPriceModel
from .schemas import FordCarInput, PredictionResponse

import logging


app = FastAPI(
    title="Ford Car Price Prediction API",
    description="Machine learning API for predicting Ford car prices.",
    version="1.0.0",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ford-price-api")

model_service = FordPriceModel()


@app.get("/")
def root():
    return {
        "message": "Ford Price Prediction API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict_price(car: FordCarInput):

    try:
        prediction = model_service.predict(
            model=car.model,
            year=car.year,
            transmission=car.transmission,
            mileage=car.mileage,
            fuelType=car.fuelType,
            tax=car.tax,
            mpg=car.mpg,
            engineSize=car.engineSize,
        )

        return PredictionResponse(predicted_price=round(prediction, 2))

    except Exception as exc:
        logger.exception("Prediction failed for payload: %s", car.model_dump())
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {type(exc).__name__}"
        ) from exc
