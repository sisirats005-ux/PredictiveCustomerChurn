"""Business-facing retention rules and ROI calculations for churn predictions."""

import os
import yaml
from src.utils import load_config

# Load business parameters from central configuration with safe fallback defaults
try:
    _config = load_config()
    _business_config = _config.get("business", {})
except Exception:
    _business_config = {}

# Centralized Business Constants (single source of truth in config.yaml)
ARPU = float(_business_config.get("arpu", 65.0))
ANNUAL_VALUE = ARPU * 12.0
CAC = float(_business_config.get("cac", 250.0))
OFFER_COST = float(_business_config.get("retention_offer_cost", 50.0))
CAMPAIGN_SUCCESS_RATE = float(_business_config.get("campaign_success_rate", 0.25))


def risk_color(risk_tier):
    """Return a consistent UI color for a risk tier.

    Parameters
    ----------
    risk_tier : str or None
        The customer's risk tier, e.g. 'High', 'Medium', 'Low'.

    Returns
    -------
    str
        Hex color code for UI display.
    """
    if not risk_tier:
        return "#8890a6"
    cleaned_tier = str(risk_tier).strip().title()
    return {"High": "#f87171", "Medium": "#fbbf24", "Low": "#34d399"}.get(cleaned_tier, "#8890a6")


def confidence_score(probability):
    """Estimate binary-classification confidence as distance from maximum uncertainty.

    Parameters
    ----------
    probability : float
        Predicted probability of churn (0.0 to 1.0).

    Returns
    -------
    float
        Confidence score between 0.5 and 1.0.
    """
    if not isinstance(probability, (int, float)):
        raise TypeError(f"Probability must be a numeric type, got {type(probability).__name__}")
    if not (0.0 <= probability <= 1.0):
        raise ValueError(f"Probability must be between 0.0 and 1.0, got {probability}")
    return max(probability, 1.0 - probability)


def estimate_retention_roi(probability, clv, offer_cost=OFFER_COST, success_rate=CAMPAIGN_SUCCESS_RATE):
    """Estimate expected net value of making a retention offer to one customer.

    Parameters
    ----------
    probability : float
        Predicted probability of churn (0.0 to 1.0).
    clv : float
        Customer Lifetime Value ($).
    offer_cost : float, optional
        Cost of promotional offer ($), defaults to config values.
    success_rate : float, optional
        Success rate of campaign (0.0 to 1.0), defaults to config values.

    Returns
    -------
    dict
        Dictionary containing expected saved value, offer cost, and net ROI.
    """
    if not isinstance(probability, (int, float)):
        raise TypeError(f"Probability must be numeric, got {type(probability).__name__}")
    if not (0.0 <= probability <= 1.0):
        raise ValueError(f"Probability must be between 0.0 and 1.0, got {probability}")
    if not isinstance(clv, (int, float)):
        raise TypeError(f"CLV must be numeric, got {type(clv).__name__}")
    if clv < 0:
        raise ValueError(f"CLV must be non-negative, got {clv}")
    if not isinstance(offer_cost, (int, float)) or offer_cost < 0:
        raise ValueError(f"Offer cost must be a non-negative number, got {offer_cost}")
    if not isinstance(success_rate, (int, float)) or not (0.0 <= success_rate <= 1.0):
        raise ValueError(f"Success rate must be between 0.0 and 1.0, got {success_rate}")

    expected_saved_value = probability * success_rate * (clv + CAC)
    return {
        "expected_saved_value": expected_saved_value,
        "offer_cost": offer_cost,
        "net_roi": expected_saved_value - offer_cost,
    }


def recommended_interventions(customer, risk_tier):
    """Return prioritized retention campaigns for an individual customer profile.

    Parameters
    ----------
    customer : dict or None
        Customer feature record.
    risk_tier : str
        Risk tier ('Low', 'Medium', or 'High').

    Returns
    -------
    list of str
        List of priority campaign intervention descriptions.
    """
    if not customer:
        customer = {}
    
    cleaned_tier = str(risk_tier).strip().title() if risk_tier else "Low"
    if cleaned_tier == "Low":
        return ["No retention action required: customer exhibits a stable customer profile."]

    interventions = []
    if customer.get("Contract") == "Month-to-month":
        interventions.append(
            "Month-to-Year Contract Migration: Offer a 10% monthly discount or a free broadband speed bump "
            "in exchange for converting to a 1-year contract."
        )
    if customer.get("InternetService") == "Fiber optic":
        interventions.append(
            "Fiber Optic Quality & Loyalty Credit: deploy a technical connection audit and apply a proactive "
            "$5/month statement credit for 6 months."
        )
    if customer.get("PaymentMethod") == "Electronic check":
        interventions.append(
            "Auto-Pay Promotion: offer a one-time $10 statement credit to transition from Electronic Check "
            "to automatic Credit Card or Bank Transfer billing."
        )
    if customer.get("tenure", 99) <= 12:
        interventions.append(
            "Early-Tenure Welcome Journey: schedule a loyalty representative check-in and dispatch a "
            "first-year anniversary bonus offer."
        )

    return interventions or [
        "Proactive Customer Care: engage via outbound satisfaction survey and offer a general 5% loyalty discount."
    ]


def campaign_name(customer, risk_tier):
    """Return a short campaign label for reporting and UI cards.

    Parameters
    ----------
    customer : dict or None
        Customer feature record.
    risk_tier : str
        Risk tier ('Low', 'Medium', or 'High').

    Returns
    -------
    str
        The name of the campaign.
    """
    if not customer:
        customer = {}

    cleaned_tier = str(risk_tier).strip().title() if risk_tier else "Low"
    if cleaned_tier == "Low":
        return "Standard lifecycle nurture"
    if customer.get("Contract") == "Month-to-month" and customer.get("InternetService") == "Fiber optic":
        return "Fiber annual-plan save offer"
    if customer.get("PaymentMethod") == "Electronic check":
        return "Auto-pay migration offer"
    if customer.get("tenure", 99) <= 12:
        return "New-customer concierge save"
    return "Proactive care outreach"


def batch_roi(flagged_customers, precision_estimate, annual_value=ANNUAL_VALUE, cac=CAC,
              offer_cost=OFFER_COST, success_rate=CAMPAIGN_SUCCESS_RATE):
    """Estimate aggregate retention economics for a scored campaign audience.

    Parameters
    ----------
    flagged_customers : int
        Number of targeted customers.
    precision_estimate : float
        Precision estimate of model (0.0 to 1.0).
    annual_value : float, optional
        Annual customer value ($), defaults to config values.
    cac : float, optional
        Customer acquisition cost ($), defaults to config values.
    offer_cost : float, optional
        Offer cost per targeted customer ($), defaults to config values.
    success_rate : float, optional
        Expected conversion rate of targeted customers (0.0 to 1.0), defaults to config values.

    Returns
    -------
    dict
        Dictionary of campaign economics metrics.
    """
    if not isinstance(flagged_customers, (int, float)) or flagged_customers < 0:
        raise ValueError("Flagged customers must be a non-negative number.")
    if not isinstance(precision_estimate, (int, float)) or not (0.0 <= precision_estimate <= 1.0):
        raise ValueError("Precision estimate must be between 0.0 and 1.0.")
    if not isinstance(annual_value, (int, float)) or annual_value < 0:
        raise ValueError("Annual value must be non-negative.")
    if not isinstance(cac, (int, float)) or cac < 0:
        raise ValueError("Customer acquisition cost (CAC) must be non-negative.")
    if not isinstance(offer_cost, (int, float)) or offer_cost < 0:
        raise ValueError("Offer cost must be non-negative.")
    if not isinstance(success_rate, (int, float)) or not (0.0 <= success_rate <= 1.0):
        raise ValueError("Success rate must be between 0.0 and 1.0.")

    estimated_true_churners = flagged_customers * precision_estimate
    customers_retained = estimated_true_churners * success_rate
    revenue_saved = customers_retained * (annual_value + cac)
    campaign_cost = flagged_customers * offer_cost
    return {
        "estimated_true_churners": estimated_true_churners,
        "customers_retained": customers_retained,
        "revenue_saved": revenue_saved,
        "campaign_cost": campaign_cost,
        "net_benefit": revenue_saved - campaign_cost,
    }

