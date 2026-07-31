"""
FastAPI REST API for the ConnectTel Customer Churn Prediction model.
Provides endpoints for scoring customer profiles and retrieving retention recommendations.
"""

from enum import Enum
from typing import List, Union

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import uvicorn

from src.business_rules import recommended_interventions
from src.predict import load_artifacts, predict_churn
from src.utils import setup_logger

logger = setup_logger(name="churn_project.api")

app = FastAPI(
    title="ConnectTel Churn Prediction API",
    description="REST API to score customer churn risk and obtain proactive business interventions.",
    version="1.1.0",
)

# Load model artifacts globally so cold-start failures are surfaced by /health.
try:
    model, preprocessor, feature_names = load_artifacts()
    MODEL_LOADED = True
    MODEL_LOAD_ERROR = None
    logger.info("API startup: model artifacts loaded successfully.")
except Exception as exc:  # noqa: BLE001 - health endpoint reports startup artifact failures.
    model, preprocessor, feature_names = None, None, None
    MODEL_LOADED = False
    MODEL_LOAD_ERROR = str(exc)
    logger.critical("API startup: model artifacts failed to load, /predict will return 503: %s", exc)


class YesNo(str, Enum):
    yes = "Yes"
    no = "No"


class Gender(str, Enum):
    female = "Female"
    male = "Male"


class MultipleLinesEnum(str, Enum):
    no = "No"
    yes = "Yes"
    no_phone_service = "No phone service"


class InternetServiceEnum(str, Enum):
    dsl = "DSL"
    fiber = "Fiber optic"
    no = "No"


class InternetAddOn(str, Enum):
    no = "No"
    yes = "Yes"
    no_internet_service = "No internet service"


class ContractEnum(str, Enum):
    month_to_month = "Month-to-month"
    one_year = "One year"
    two_year = "Two year"


class PaymentMethodEnum(str, Enum):
    electronic_check = "Electronic check"
    mailed_check = "Mailed check"
    bank_transfer = "Bank transfer (automatic)"
    credit_card = "Credit card (automatic)"


class CustomerData(BaseModel):
    gender: Gender = Field(..., description="Customer gender")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Senior citizen indicator (0 or 1)")
    Partner: YesNo = Field(..., description="Whether the customer has a partner")
    Dependents: YesNo = Field(..., description="Whether the customer has dependents")
    tenure: int = Field(..., ge=0, le=100, description="Months of customer tenure")
    PhoneService: YesNo = Field(..., description="Whether customer has phone service")
    MultipleLines: MultipleLinesEnum = Field(..., description="Phone line structure")
    InternetService: InternetServiceEnum = Field(..., description="Internet service type")
    OnlineSecurity: InternetAddOn = Field(..., description="Online security service")
    OnlineBackup: InternetAddOn = Field(..., description="Online backup service")
    DeviceProtection: InternetAddOn = Field(..., description="Device protection service")
    TechSupport: InternetAddOn = Field(..., description="Technical support service")
    StreamingTV: InternetAddOn = Field(..., description="TV streaming service")
    StreamingMovies: InternetAddOn = Field(..., description="Movies streaming service")
    Contract: ContractEnum = Field(..., description="Contract term")
    PaperlessBilling: YesNo = Field(..., description="Paperless billing")
    PaymentMethod: PaymentMethodEnum = Field(..., description="Payment channel")
    MonthlyCharges: float = Field(..., ge=0.0, description="Monthly recurring charge amount")
    TotalCharges: Union[float, str] = Field(
        ..., description="Total cumulative charge amount (numeric float or blank string for new customers)"
    )


class ChurnResponse(BaseModel):
    # `model_confidence` below triggers a spurious Pydantic UserWarning
    # ("conflict with protected namespace 'model_'") on every process
    # startup -- it's a false positive (we don't use Pydantic's own
    # `model_*` methods on this field), so it's disabled here rather than
    # renaming the field and breaking the public API contract.
    model_config = {"protected_namespaces": ()}

    churn_probability: float = Field(..., description="Model calculated probability of churn (0.0 to 1.0)")
    churn_prediction: str = Field(..., description="Binary churn decision ('Yes' or 'No')")
    decision_threshold: float = Field(..., description="Decision threshold used for the binary churn prediction")
    risk_tier: str = Field(..., description="Calculated risk tier ('Low', 'Medium', 'High')")
    clv: float = Field(..., description="Estimated Customer Lifetime Value ($)")
    billing_risk: float = Field(..., description="Billing cash-collection friction risk indicator")
    high_risk_profile: int = Field(..., description="1 if account represents high risk profile, 0 otherwise")
    persona: str = Field(..., description="Identified customer persona cluster")
    top_risk_factors: List[str] = Field(..., description="Human-readable factors driving the risk assessment")
    recommended_action: str = Field(..., description="Priority retention action for this customer")
    model_confidence: float = Field(..., description="Confidence in the binary model decision")
    recommended_interventions: List[str] = Field(
        ..., description="Priority customer retention actions based on profile indicators"
    )


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Verify API availability and model loading status for container health probes."""
    payload = {"status": "healthy" if MODEL_LOADED else "unhealthy", "model_loaded": MODEL_LOADED}
    if MODEL_LOAD_ERROR:
        payload["error"] = MODEL_LOAD_ERROR
    return payload


@app.post("/predict", response_model=ChurnResponse)
def predict(customer: CustomerData):
    """Score one customer profile and return tailored CRM retention actions."""
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service unavailable: Model artifacts could not be loaded.",
        )

    customer_dict = customer.model_dump(mode="json")

    try:
        scoring = predict_churn(
            customer=customer_dict,
            model=model,
            preprocessor=preprocessor,
            feature_names=feature_names,
        )

        interventions = recommended_interventions(customer_dict, scoring["risk_tier"])

        return {**scoring, "recommended_interventions": interventions}

    except Exception as exc:  # noqa: BLE001 - converts runtime inference issues into API errors.
        logger.exception("POST /predict failed during inference: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline execution error: {exc}",
        ) from exc


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)