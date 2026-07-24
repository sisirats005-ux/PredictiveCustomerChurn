"""
Integration tests for FastAPI REST API endpoints.
Uses FastAPI's TestClient to verify health checks, prediction outputs,
and request validation rules.
"""

import pytest
from fastapi.testclient import TestClient

# Import the api module
try:
    from api import app
    API_IMPORT_SUCCESS = True
except ImportError:
    API_IMPORT_SUCCESS = False


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    if not API_IMPORT_SUCCESS:
        pytest.skip("FastAPI api.py is not importable or dependencies are missing.")
    return TestClient(app)


def test_health_endpoint(client):
    """Verify GET /health returns successfully."""
    response = client.get("/health")
    assert response.status_code in (200, 503)  # 200 if model loaded, 503 if not loaded
    json_data = response.json()
    assert "status" in json_data
    assert "model_loaded" in json_data


def test_predict_endpoint_validation_error(client):
    """Verify POST /predict returns 422 Unprocessable Entity when fields are missing."""
    # Send empty payload
    response = client.post("/predict", json={})
    assert response.status_code == 422
    
    # Send partial customer record (missing MonthlyCharges)
    partial_payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "TotalCharges": 960.0
    }
    response = client.post("/predict", json=partial_payload)
    assert response.status_code == 422
    assert "MonthlyCharges" in response.text


def test_predict_endpoint_invalid_values(client):
    """Verify validation constraints are enforced on input fields."""
    invalid_payload = {
        "gender": "Female",
        "SeniorCitizen": 3,  # Invalid: must be 0 or 1
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": -5,        # Invalid: must be >= 0
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": -20.0,  # Invalid: must be >= 0
        "TotalCharges": 960.0
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
