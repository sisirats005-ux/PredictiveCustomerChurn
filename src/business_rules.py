"""Business-facing retention rules and ROI calculations for churn predictions."""

ANNUAL_VALUE = 780.0
CAC = 250.0
OFFER_COST = 50.0
CAMPAIGN_SUCCESS_RATE = 0.25


def risk_color(risk_tier):
    """Return a consistent UI color for a risk tier."""
    return {"High": "#dc2626", "Medium": "#f59e0b", "Low": "#16a34a"}.get(risk_tier, "#64748b")


def confidence_score(probability):
    """Estimate binary-classification confidence as distance from maximum uncertainty."""
    return max(probability, 1.0 - probability)


def estimate_retention_roi(probability, clv, offer_cost=OFFER_COST, success_rate=CAMPAIGN_SUCCESS_RATE):
    """Estimate expected net value of making a retention offer to one customer."""
    expected_saved_value = probability * success_rate * (clv + CAC)
    return {
        "expected_saved_value": expected_saved_value,
        "offer_cost": offer_cost,
        "net_roi": expected_saved_value - offer_cost,
    }


def recommended_interventions(customer, risk_tier):
    """Return prioritized retention campaigns for an individual customer profile."""
    if risk_tier == "Low":
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
    """Return a short campaign label for reporting and UI cards."""
    if risk_tier == "Low":
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
    """Estimate aggregate retention economics for a scored campaign audience."""
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
