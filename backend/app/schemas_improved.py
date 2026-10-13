from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


FordModel = Literal[
    "b-max",
    "c-max",
    "ecosport",
    "edge",
    "escort",
    "fiesta",
    "focus",
    "fusion",
    "galaxy",
    "grand c-max",
    "grand tourneo connect",
    "ka",
    "ka+",
    "kuga",
    "mondeo",
    "mustang",
    "puma",
    "ranger",
    "s-max",
    "streetka",
    "tourneo connect",
    "tourneo custom",
    "transit tourneo",
]


class FordCarInput(BaseModel):
    """Input schema aligned with the cleaned Ford training data."""

    model_config = ConfigDict(extra="forbid")

    model: FordModel
    year: int = Field(..., ge=1996, le=2020)
    transmission: Literal["automatic", "manual", "semi-auto"]
    mileage: int = Field(..., ge=1, le=177644)
    fuelType: Literal["diesel", "electric", "hybrid", "other", "petrol"]
    tax: int = Field(..., ge=0, le=580)
    mpg: float = Field(..., ge=20.8, le=201.8, allow_inf_nan=False)
    engineSize: float = Field(..., gt=0, le=5.0, allow_inf_nan=False)

    @field_validator("model", "transmission", "fuelType", mode="before")
    @classmethod
    def normalize_category(cls, value):
        if isinstance(value, str):
            return value.strip().casefold()
        return value


class PredictionResponse(BaseModel):
    predicted_price: float = Field(..., ge=0, allow_inf_nan=False)
    currency: Literal["GBP"] = "GBP"
