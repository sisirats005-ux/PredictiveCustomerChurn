"""Unit tests for src/business_rules.py."""

import pytest
from src.business_rules import (
    risk_color,
    confidence_score,
    estimate_retention_roi,
    recommended_interventions,
    campaign_name,
    batch_roi,
    ARPU,
    ANNUAL_VALUE,
    CAC,
    OFFER_COST,
    CAMPAIGN_SUCCESS_RATE,
)


def test_risk_color():
    """Test UI color mapping for different risk tiers."""
    assert risk_color("High") == "#f87171"
    assert risk_color("Medium") == "#fbbf24"
    assert risk_color("Low") == "#34d399"
    assert risk_color("high") == "#f87171"
    assert risk_color("  Medium  ") == "#fbbf24"
    assert risk_color("Unknown") == "#8890a6"
    assert risk_color(None) == "#8890a6"


def test_confidence_score():
    """Test binary classification confidence calculation and validation."""
    assert confidence_score(0.8) == 0.8
    assert confidence_score(0.2) == 0.8
    assert confidence_score(0.5) == 0.5
    assert confidence_score(0.0) == 1.0
    assert confidence_score(1.0) == 1.0

    # Validation errors
    with pytest.raises(ValueError, match="Probability must be between 0.0 and 1.0"):
        confidence_score(-0.1)
    with pytest.raises(ValueError, match="Probability must be between 0.0 and 1.0"):
        confidence_score(1.1)
    with pytest.raises(TypeError, match="must be a numeric type"):
        confidence_score("not a float")


def test_estimate_retention_roi():
    """Test individual customer retention ROI calculations and validation."""
    # Test with custom arguments
    # expected_saved_value = probability * success_rate * (clv + CAC)
    # expected_saved_value = 0.8 * 0.25 * (1000.0 + 250.0) = 0.2 * 1250.0 = 250.0
    # net_roi = 250.0 - 50.0 = 200.0
    res = estimate_retention_roi(
        probability=0.8,
        clv=1000.0,
        offer_cost=50.0,
        success_rate=0.25,
    )
    assert res["expected_saved_value"] == 250.0
    assert res["offer_cost"] == 50.0
    assert res["net_roi"] == 200.0

    # Test with defaults (from configuration/business rules constants)
    res_default = estimate_retention_roi(probability=0.8, clv=1000.0)
    assert res_default["offer_cost"] == OFFER_COST
    expected_saved = 0.8 * CAMPAIGN_SUCCESS_RATE * (1000.0 + CAC)
    assert res_default["expected_saved_value"] == expected_saved
    assert res_default["net_roi"] == expected_saved - OFFER_COST

    # Validation errors
    with pytest.raises(ValueError, match="Probability must be between 0.0 and 1.0"):
        estimate_retention_roi(-0.1, 1000.0)
    with pytest.raises(ValueError, match="CLV must be non-negative"):
        estimate_retention_roi(0.5, -100.0)
    with pytest.raises(ValueError, match="Offer cost must be a non-negative number"):
        estimate_retention_roi(0.5, 100.0, offer_cost=-10.0)
    with pytest.raises(ValueError, match="Success rate must be between 0.0 and 1.0"):
        estimate_retention_roi(0.5, 100.0, success_rate=1.5)
    with pytest.raises(TypeError):
        estimate_retention_roi("string", 100.0)


def test_recommended_interventions():
    """Test CRM intervention list generation for various customer profiles."""
    # Low risk
    low_risk = recommended_interventions({}, "Low")
    assert "No retention action required" in low_risk[0]

    # Month-to-month contract
    res1 = recommended_interventions({"Contract": "Month-to-month"}, "High")
    assert len(res1) == 1
    assert "Month-to-Year Contract Migration" in res1[0]

    # Fiber optic
    res2 = recommended_interventions({"InternetService": "Fiber optic"}, "High")
    assert len(res2) == 1
    assert "Fiber Optic Quality & Loyalty Credit" in res2[0]

    # Electronic check
    res3 = recommended_interventions({"PaymentMethod": "Electronic check"}, "High")
    assert len(res3) == 1
    assert "Auto-Pay Promotion" in res3[0]

    # Short tenure
    res4 = recommended_interventions({"tenure": 6}, "High")
    assert len(res4) == 1
    assert "Early-Tenure Welcome Journey" in res4[0]

    # Multiple matching triggers
    res_multiple = recommended_interventions(
        {
            "Contract": "Month-to-month",
            "InternetService": "Fiber optic",
            "PaymentMethod": "Electronic check",
            "tenure": 6,
        },
        "High",
    )
    assert len(res_multiple) == 4

    # No specific triggers but high risk
    res_default = recommended_interventions(
        {
            "Contract": "Two year",
            "InternetService": "DSL",
            "PaymentMethod": "Mailed check",
            "tenure": 48,
        },
        "Medium",
    )
    assert len(res_default) == 1
    assert "Proactive Customer Care" in res_default[0]

    # None/empty safety check
    res_none = recommended_interventions(None, "High")
    assert len(res_none) == 1
    assert "Proactive Customer Care" in res_none[0]


def test_campaign_name():
    """Test short campaign shorthand mapping logic."""
    assert campaign_name({}, "Low") == "Standard lifecycle nurture"
    assert campaign_name({"Contract": "Month-to-month", "InternetService": "Fiber optic"}, "High") == "Fiber annual-plan save offer"
    assert campaign_name({"PaymentMethod": "Electronic check"}, "High") == "Auto-pay migration offer"
    assert campaign_name({"tenure": 6}, "High") == "New-customer concierge save"
    
    # Fallback default
    assert campaign_name({"Contract": "Two year", "tenure": 48}, "Medium") == "Proactive care outreach"
    
    # None/empty safety check
    assert campaign_name(None, "High") == "Proactive care outreach"


def test_batch_roi():
    """Test cohort campaign aggregate economics model calculations."""
    # Test calculations
    # flagged_customers = 1000
    # precision_estimate = 0.8
    # true_churners = 800
    # retained = 800 * 0.25 = 200
    # revenue_saved = 200 * (780.0 + 250.0) = 200 * 1030 = 206000
    # campaign_cost = 1000 * 50.0 = 50000
    # net_benefit = 206000 - 50000 = 156000
    res = batch_roi(
        flagged_customers=1000,
        precision_estimate=0.8,
        annual_value=780.0,
        cac=250.0,
        offer_cost=50.0,
        success_rate=0.25,
    )
    assert res["estimated_true_churners"] == 800.0
    assert res["customers_retained"] == 200.0
    assert res["revenue_saved"] == 206000.0
    assert res["campaign_cost"] == 50000.0
    assert res["net_benefit"] == 156000.0

    # Test parameter validation errors
    with pytest.raises(ValueError, match="Flagged customers must be a non-negative number"):
        batch_roi(-100, 0.8)
    with pytest.raises(ValueError, match="Precision estimate must be between 0.0 and 1.0"):
        batch_roi(1000, 1.2)
    with pytest.raises(ValueError, match="Annual value must be non-negative"):
        batch_roi(1000, 0.8, annual_value=-50.0)
    with pytest.raises(ValueError, match="Customer acquisition cost .* must be non-negative"):
        batch_roi(1000, 0.8, cac=-10.0)
    with pytest.raises(ValueError, match="Offer cost must be non-negative"):
        batch_roi(1000, 0.8, offer_cost=-5.0)
    with pytest.raises(ValueError, match="Success rate must be between 0.0 and 1.0"):
        batch_roi(1000, 0.8, success_rate=-0.1)
