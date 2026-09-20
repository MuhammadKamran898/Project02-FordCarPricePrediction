from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "ford_price_model.pkl"


class FordPriceModel:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)

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

        input_data = pd.DataFrame(
            [
                {
                    "model": model.strip(),
                    "year": year,
                    "transmission": transmission.strip(),
                    "mileage": mileage,
                    "fuelType": fuelType.strip(),
                    "tax": tax,
                    "mpg": mpg,
                    "engineSize": engineSize,
                }
            ]
        )

        prediction = self.model.predict(input_data)[0]

        return float(prediction)
