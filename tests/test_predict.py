"""
Tests for src/predict.py — the inference module used by the Streamlit app.

`prepare_customer_frame` is tested directly (pure pandas, no model
dependency). The full `predict_churn` path additionally requires the
serialized model artifacts in `models/` so those
tests are skipped automatically in environments where artifacts are unavailable
rather than failing the whole suite.
"""

import os

import pandas as pd
import pytest

from src.predict import prepare_customer_frame, load_artifacts, predict_churn

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
ARTIFACTS_AVAILABLE = all(
    os.path.exists(os.path.join(MODELS_DIR, f))
    for f in ("logistic_regression_model.joblib", "preprocessing_pipeline.joblib", "feature_names.joblib")
)

requires_artifacts = pytest.mark.skipif(
    not ARTIFACTS_AVAILABLE,
    reason="Serialized Logistic Regression artifacts are not available in this environment",
)


class TestPrepareCustomerFrame:
    def test_returns_single_row_engineered_frame(self, raw_customer_row):
        customer = {k: v for k, v in raw_customer_row.items() if k not in ("customerID", "Churn")}

        frame = prepare_customer_frame(customer)

        assert len(frame) == 1
        # Engineered columns from feature_engineering.create_features should be present.
        for col in ["ServiceCount", "TotalChargesPerTenure", "HasFamily", "LongTermCustomer", "AutoPayment"]:
            assert col in frame.columns

    def test_blank_total_charges_handled_like_clean_data(self, raw_customer_row):
        customer = {k: v for k, v in raw_customer_row.items() if k not in ("customerID", "Churn")}
        customer["tenure"] = 0
        customer["TotalCharges"] = "   "

        frame = prepare_customer_frame(customer)

        assert frame.loc[0, "TotalCharges"] == 0.0
        # tenure == 0 edge case: TotalChargesPerTenure should fall back to MonthlyCharges.
        assert frame.loc[0, "TotalChargesPerTenure"] == pytest.approx(customer["MonthlyCharges"])


@requires_artifacts
class TestPredictChurnEndToEnd:
    def test_predict_churn_returns_expected_keys_and_ranges(self, raw_customer_row):
        customer = {k: v for k, v in raw_customer_row.items() if k not in ("customerID", "Churn")}

        result = predict_churn(customer)
        expected_keys = {
            "churn_probability", "churn_prediction", "risk_tier",
            "clv", "billing_risk", "high_risk_profile", "persona", "model_confidence"
        }
        assert expected_keys.issubset(set(result.keys()))
        assert 0.0 <= result["churn_probability"] <= 1.0
        assert result["churn_prediction"] in {"Yes", "No"}
        assert result["risk_tier"] in {"Low", "Medium", "High"}
        assert 0.5 <= result["model_confidence"] <= 1.0

    def test_predict_churn_is_deterministic(self, raw_customer_row):
        customer = {k: v for k, v in raw_customer_row.items() if k not in ("customerID", "Churn")}
        model, preprocessor, feature_names = load_artifacts()

        result_1 = predict_churn(customer, model=model, preprocessor=preprocessor, feature_names=feature_names)
        result_2 = predict_churn(customer, model=model, preprocessor=preprocessor, feature_names=feature_names)

        assert result_1 == result_2
