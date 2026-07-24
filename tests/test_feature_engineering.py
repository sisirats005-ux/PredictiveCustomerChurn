"""
Unit tests for src/feature_engineering.py.
Each engineered feature (ServiceCount, TotalChargesPerTenure, HasFamily,
LongTermCustomer, AutoPayment, and the new advanced features) is checked 
against expected values on small, purpose-built rows.
"""

import pandas as pd
import pytest

from src.feature_engineering import create_features


def make_row(**overrides):
    """Baseline cleaned-data row with sensible defaults, overridable per test."""
    base = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 10,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 50.0,
        "TotalCharges": 500.0,
    }
    base.update(overrides)
    return base


class TestServiceCount:
    def test_counts_zero_for_no_services(self):
        df = pd.DataFrame([make_row(
            PhoneService="No", MultipleLines="No phone service", InternetService="No",
            OnlineSecurity="No internet service", OnlineBackup="No internet service",
            DeviceProtection="No internet service", TechSupport="No internet service",
            StreamingTV="No internet service", StreamingMovies="No internet service",
        )])
        result = create_features(df)
        assert result.loc[0, "ServiceCount"] == 0

    def test_counts_all_nine_services_when_fully_subscribed(self):
        df = pd.DataFrame([make_row(
            PhoneService="Yes", MultipleLines="Yes", InternetService="Fiber optic",
            OnlineSecurity="Yes", OnlineBackup="Yes", DeviceProtection="Yes",
            TechSupport="Yes", StreamingTV="Yes", StreamingMovies="Yes",
        )])
        result = create_features(df)
        assert result.loc[0, "ServiceCount"] == 9

    def test_partial_service_count(self):
        df = pd.DataFrame([make_row(
            PhoneService="Yes", MultipleLines="No", InternetService="DSL",
            OnlineSecurity="No", OnlineBackup="No", DeviceProtection="No",
            TechSupport="No", StreamingTV="No", StreamingMovies="No",
        )])
        result = create_features(df)
        assert result.loc[0, "ServiceCount"] == 2


class TestTotalChargesPerTenure:
    def test_normal_division(self):
        df = pd.DataFrame([make_row(tenure=10, TotalCharges=500.0, MonthlyCharges=45.0)])
        result = create_features(df)
        assert result.loc[0, "TotalChargesPerTenure"] == pytest.approx(50.0)

    def test_zero_tenure_falls_back_to_monthly_charges(self):
        df = pd.DataFrame([make_row(tenure=0, TotalCharges=0.0, MonthlyCharges=45.0)])
        result = create_features(df)
        assert result.loc[0, "TotalChargesPerTenure"] == pytest.approx(45.0)


class TestHasFamily:
    @pytest.mark.parametrize(
        "partner,dependents,expected",
        [
            ("Yes", "No", 1),
            ("No", "Yes", 1),
            ("Yes", "Yes", 1),
            ("No", "No", 0),
        ],
    )
    def test_has_family_logic(self, partner, dependents, expected):
        df = pd.DataFrame([make_row(Partner=partner, Dependents=dependents)])
        result = create_features(df)
        assert result.loc[0, "HasFamily"] == expected


class TestLongTermCustomer:
    @pytest.mark.parametrize("tenure,expected", [(0, 0), (24, 0), (25, 1), (72, 1)])
    def test_long_term_threshold(self, tenure, expected):
        df = pd.DataFrame([make_row(tenure=tenure)])
        result = create_features(df)
        assert result.loc[0, "LongTermCustomer"] == expected


class TestAutoPayment:
    @pytest.mark.parametrize(
        "payment_method,expected",
        [
            ("Bank transfer (automatic)", 1),
            ("Credit card (automatic)", 1),
            ("Electronic check", 0),
            ("Mailed check", 0),
        ],
    )
    def test_auto_payment_detection(self, payment_method, expected):
        df = pd.DataFrame([make_row(PaymentMethod=payment_method)])
        result = create_features(df)
        assert result.loc[0, "AutoPayment"] == expected


class TestAdvancedFeatures:
    def test_fiber_interactions(self):
        df = pd.DataFrame([
            make_row(InternetService="Fiber optic", Contract="Month-to-month", TechSupport="No"),
            make_row(InternetService="DSL", Contract="One year", TechSupport="Yes")
        ])
        result = create_features(df)
        assert result.loc[0, "Fiber_x_MonthToMonth"] == 1
        assert result.loc[0, "Fiber_x_NoTechSupport"] == 1
        assert result.loc[1, "Fiber_x_MonthToMonth"] == 0
        assert result.loc[1, "Fiber_x_NoTechSupport"] == 0

    def test_manual_billing_interactions(self):
        df = pd.DataFrame([
            make_row(Contract="Month-to-month", PaymentMethod="Electronic check"),
            make_row(Contract="Two year", PaymentMethod="Credit card (automatic)")
        ])
        result = create_features(df)
        assert result.loc[0, "MonthToMonth_x_NoAutoPay"] == 1
        assert result.loc[0, "HighRiskContractPay"] == 1
        assert result.loc[1, "MonthToMonth_x_NoAutoPay"] == 0
        assert result.loc[1, "HighRiskContractPay"] == 0

    def test_clv_calculation(self):
        df = pd.DataFrame([
            make_row(Contract="Month-to-month", MonthlyCharges=100.0, tenure=10), # 100 * (10 + 6) = 1600
            make_row(Contract="One year", MonthlyCharges=100.0, tenure=10),      # 100 * (10 + 12) = 2200
            make_row(Contract="Two year", MonthlyCharges=100.0, tenure=10),      # 100 * (10 + 24) = 3400
        ])
        result = create_features(df)
        assert result.loc[0, "CLV"] == pytest.approx(1600.0)
        assert result.loc[1, "CLV"] == pytest.approx(2200.0)
        assert result.loc[2, "CLV"] == pytest.approx(3400.0)

    def test_charges_ratio_and_billing_risk(self):
        df = pd.DataFrame([
            make_row(Contract="Month-to-month", MonthlyCharges=100.0, TotalCharges=400.0, PaymentMethod="Electronic check"),
            make_row(Contract="Two year", MonthlyCharges=100.0, TotalCharges=900.0, PaymentMethod="Credit card (automatic)")
        ])
        result = create_features(df)
        # ChargesRatio: 100 / (400 + 100) = 0.20
        assert result.loc[0, "ChargesRatio"] == pytest.approx(0.20)
        # BillingRisk: 100.0 * (1 - 0) * 1 = 100.0
        assert result.loc[0, "BillingRisk"] == pytest.approx(100.0)
        
        # ChargesRatio: 100 / (900 + 100) = 0.10
        assert result.loc[1, "ChargesRatio"] == pytest.approx(0.10)
        # BillingRisk: 100.0 * (1 - 1) * 0 = 0.0
        assert result.loc[1, "BillingRisk"] == pytest.approx(0.0)

    def test_risk_segmentation_profiles(self):
        df = pd.DataFrame([
            make_row(Contract="Month-to-month", PaymentMethod="Electronic check", InternetService="Fiber optic"),
            make_row(Contract="Two year", PaymentMethod="Credit card (automatic)", InternetService="DSL")
        ])
        result = create_features(df)
        assert result.loc[0, "HighRiskProfile"] == 1
        assert result.loc[1, "HighRiskProfile"] == 0

    def test_personas(self):
        df = pd.DataFrame([
            make_row(tenure=5, MonthlyCharges=80.0),  # New HighSpend
            make_row(tenure=20, MonthlyCharges=80.0), # Loyal HighSpend
            make_row(tenure=5, MonthlyCharges=40.0),  # LowSpend
        ])
        result = create_features(df)
        assert result.loc[0, "Persona_New_HighSpend"] == 1
        assert result.loc[0, "Persona_Loyal_HighSpend"] == 0
        assert result.loc[1, "Persona_New_HighSpend"] == 0
        assert result.loc[1, "Persona_Loyal_HighSpend"] == 1
        assert result.loc[2, "Persona_New_HighSpend"] == 0
        assert result.loc[2, "Persona_Loyal_HighSpend"] == 0


class TestCreateFeaturesGeneral:
    def test_preserves_original_columns(self):
        df = pd.DataFrame([make_row()])
        result = create_features(df)
        for col in df.columns:
            assert col in result.columns

    def test_does_not_mutate_input(self):
        df = pd.DataFrame([make_row()])
        original = df.copy(deep=True)
        create_features(df)
        pd.testing.assert_frame_equal(df, original)

    def test_works_on_multiple_rows(self):
        df = pd.DataFrame([make_row(tenure=5), make_row(tenure=30, Partner="Yes")])
        result = create_features(df)
        assert len(result) == 2
        assert result.loc[0, "LongTermCustomer"] == 0
        assert result.loc[1, "LongTermCustomer"] == 1
