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
    assert response.status_code == 200  # Readiness details are represented in the response body
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


def test_predict_endpoint_success_returns_full_schema(client):
    """Verify a valid POST /predict returns 200 with every documented response field.

    Regression guard: response_model=ChurnResponse silently drops any key
    predict_churn() returns that isn't also declared on ChurnResponse, so a
    field can be computed correctly yet never reach the client without this
    test noticing.
    """
    from api import MODEL_LOADED
    if not MODEL_LOADED:
        pytest.skip("Model artifacts are not loaded in this environment.")

    payload = {
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
        "MonthlyCharges": 80.0,
        "TotalCharges": 960.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    expected_fields = {
        "churn_probability", "churn_prediction", "decision_threshold", "risk_tier",
        "clv", "billing_risk", "high_risk_profile", "persona", "top_risk_factors",
        "recommended_action", "model_confidence", "recommended_interventions",
    }
    assert expected_fields.issubset(data.keys())
    assert 0.5 <= data["model_confidence"] <= 1.0


def test_predict_endpoint_rejects_unknown_categories(client):
    """Verify strict categorical enums prevent unseen/business-invalid values at the API boundary."""
    payload = {
        "gender": "Unknown",
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
        "MonthlyCharges": 80.0,
        "TotalCharges": 960.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "gender" in response.text
