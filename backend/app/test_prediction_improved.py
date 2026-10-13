"""API contract tests for the improved Ford predictor.

These tests load the improved modules as a synthetic package so the relative
imports in main_improved.py work without assuming your repository's package name.
Run: python -m pytest test_prediction_improved.py -q
"""
import importlib
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent
PACKAGE_NAME = "ford_predictor_improved_test_package"

# Register a synthetic package whose modules live beside this test file.
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)
api_module = importlib.import_module(f"{PACKAGE_NAME}.main_improved")


VALID_PAYLOAD = {
    "model": "Fiesta",
    "year": 2017,
    "transmission": "Manual",
    "mileage": 15000,
    "fuelType": "Petrol",
    "tax": 145,
    "mpg": 58.9,
    "engineSize": 1.2,
}


class FakeModelService:
    def predict(self, **kwargs):
        assert kwargs["model"] == "fiesta"
        assert kwargs["transmission"] == "manual"
        assert kwargs["fuelType"] == "petrol"
        assert set(kwargs) == {
            "model", "year", "transmission", "mileage",
            "fuelType", "tax", "mpg", "engineSize",
        }
        return 12345.678


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api_module, "FordPriceModel", FakeModelService)
    with TestClient(api_module.app) as test_client:
        yield test_client


def test_prediction_normalizes_categories_and_returns_gbp(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert response.json() == {
        "predicted_price": 12345.68,
        "currency": "GBP",
    }


def test_unsupported_category_is_rejected(client):
    response = client.post(
        "/predict",
        json={**VALID_PAYLOAD, "model": "Unsupported Model"},
    )
    assert response.status_code == 422


def test_out_of_range_year_is_rejected(client):
    response = client.post(
        "/predict",
        json={**VALID_PAYLOAD, "year": 2060},
    )
    assert response.status_code == 422


def test_extra_fields_are_rejected(client):
    response = client.post(
        "/predict",
        json={**VALID_PAYLOAD, "unexpected": "value"},
    )
    assert response.status_code == 422


def test_liveness_endpoint(client):
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_readiness_endpoint(client):
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
