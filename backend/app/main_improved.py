import logging
import math
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from .model_service_improved import FordPriceModel
from .schemas_improved import FordCarInput, PredictionResponse


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ford-price-api")


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Keep the API process observable even if model loading fails."""
    try:
        application.state.model_service = FordPriceModel()
        logger.info("Ford price model is ready.")
    except Exception:
        application.state.model_service = None
        logger.exception("Failed to load the Ford price model.")
    yield


app = FastAPI(
    title="Ford Car Price Prediction API",
    description="Machine learning API for predicting Ford car prices.",
    version="1.1.0",
    lifespan=lifespan,
)


def _get_model_service(request: Request):
    return getattr(request.app.state, "model_service", None)


@app.get("/")
def root():
    return {
        "message": "Ford Price Prediction API",
        "status": "running",
    }


@app.get("/health")
def health(request: Request):
    """Backward-compatible health endpoint."""
    return {
        "status": "healthy" if _get_model_service(request) else "degraded"
    }


@app.get("/health/live")
def health_live():
    """Liveness check: the API process is responding."""
    return {"status": "alive"}


@app.get("/health/ready")
def health_ready(request: Request):
    """Readiness check: the model artifact has loaded successfully."""
    if _get_model_service(request) is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is not ready.",
        )
    return {"status": "ready"}


@app.post("/predict", response_model=PredictionResponse)
def predict_price(car: FordCarInput, request: Request):
    model_service = _get_model_service(request)
    if model_service is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is unavailable.",
        )

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

        if not math.isfinite(prediction) or prediction < 0:
            raise ValueError("The model returned an invalid price.")

        return PredictionResponse(predicted_price=round(prediction, 2))

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "Prediction failed for model=%s, year=%s",
            car.model,
            car.year,
        )
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please try again later.",
        ) from exc
