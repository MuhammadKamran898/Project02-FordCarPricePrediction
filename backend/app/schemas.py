from pydantic import BaseModel, Field


class FordCarInput(BaseModel):
    model: str = Field(..., min_length=1)
    year: int = Field(..., ge=1990, le=2026)
    transmission: str = Field(..., min_length=1)
    mileage: int = Field(..., ge=0)
    fuelType: str = Field(..., min_length=1)
    tax: int = Field(..., ge=0)
    mpg: float = Field(..., gt=0)
    engineSize: float = Field(..., gt=0)


class PredictionResponse(BaseModel):
    predicted_price: float
    currency: str = "GBP"
