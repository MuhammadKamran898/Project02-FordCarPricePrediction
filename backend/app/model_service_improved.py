import logging
import math
import os
from pathlib import Path

import joblib
import pandas as pd


logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "ford_price_model.pkl"
)
MODEL_PATH = Path(
    os.getenv("FORD_MODEL_PATH", str(DEFAULT_MODEL_PATH))
).expanduser().resolve()

FEATURE_COLUMNS = [
    "model",
    "year",
    "transmission",
    "mileage",
    "fuelType",
    "tax",
    "mpg",
    "engineSize",
]


class FordPriceModel:
    """Load and serve the fitted Ford price prediction pipeline."""

    def __init__(self) -> None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Model artifact not found at '{MODEL_PATH}'. "
                "Train the improved notebook or set FORD_MODEL_PATH."
            )

        self.model = joblib.load(MODEL_PATH)

        if not callable(getattr(self.model, "predict", None)):
            raise TypeError(
                "The configured model artifact does not expose predict()."
            )

        trained_features = getattr(self.model, "feature_names_in_", None)
        if trained_features is not None and list(trained_features) != FEATURE_COLUMNS:
            raise ValueError(
                "The configured artifact was trained with an unexpected feature "
                f"schema. Expected {FEATURE_COLUMNS}, got {list(trained_features)}."
            )

        logger.info("Ford price model loaded from %s", MODEL_PATH)

    def predict(
        self,
        model: str,
        year: int,
        transmission: str,
        mileage: int,
        fuelType: str,
        tax: int,
        mpg: float,
        engineSize: float,
    ) -> float:
        """Predict a price using the same feature format as model training."""
        input_data = pd.DataFrame(
            [
                {
                    "model": model.strip().casefold(),
                    "year": year,
                    "transmission": transmission.strip().casefold(),
                    "mileage": mileage,
                    "fuelType": fuelType.strip().casefold(),
                    "tax": tax,
                    "mpg": mpg,
                    "engineSize": engineSize,
                }
            ],
            columns=FEATURE_COLUMNS,
        )

        prediction = float(self.model.predict(input_data)[0])
        if not math.isfinite(prediction) or prediction < 0:
            raise ValueError("The model returned an invalid price prediction.")

        return prediction
