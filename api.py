"""
FastAPI REST API for the ConnectTel Customer Churn Prediction model.
Provides endpoints for scoring customer profiles and retrieving retention recommendations.
"""

import os
from typing import List, Optional, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from src.predict import load_artifacts, predict_churn

app = FastAPI(
    title="ConnectTel Churn Prediction API",
    description="REST API to score customer churn risk and obtain proactive business interventions.",
    version="1.0.0"
)

# Load model artifacts globally
try:
    model, preprocessor, feature_names = load_artifacts()
    MODEL_LOADED = True
except Exception as e:
    model, preprocessor, feature_names = None, None, None
    MODEL_LOADED = False


class CustomerData(BaseModel):
    gender: str = Field(..., description="Customer gender ('Male', 'Female')")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Senior citizen indicator (0 or 1)")
    Partner: str = Field(..., description="Whether the customer has a partner ('Yes', 'No')")
    Dependents: str = Field(..., description="Whether the customer has dependents ('Yes', 'No')")
    tenure: int = Field(..., ge=0, le=100, description="Months of customer tenure")
    PhoneService: str = Field(..., description="Whether customer has phone service ('Yes', 'No')")
    MultipleLines: str = Field(..., description="Phone line structure ('No', 'Yes', 'No phone service')")
    InternetService: str = Field(..., description="Internet service type ('DSL', 'Fiber optic', 'No')")
    OnlineSecurity: str = Field(..., description="Online security service ('No', 'Yes', 'No internet service')")
    OnlineBackup: str = Field(..., description="Online backup service ('No', 'Yes', 'No internet service')")
    DeviceProtection: str = Field(..., description="Device protection service ('No', 'Yes', 'No internet service')")
    TechSupport: str = Field(..., description="Technical support service ('No', 'Yes', 'No internet service')")
    StreamingTV: str = Field(..., description="TV streaming service ('No', 'Yes', 'No internet service')")
    StreamingMovies: str = Field(..., description="Movies streaming service ('No', 'Yes', 'No internet service')")
    Contract: str = Field(..., description="Contract term ('Month-to-month', 'One year', 'Two year')")
    PaperlessBilling: str = Field(..., description="Paperless billing ('Yes', 'No')")
    PaymentMethod: str = Field(..., description="Payment channel ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)')")
    MonthlyCharges: float = Field(..., ge=0.0, description="Monthly recurring charge amount")
    TotalCharges: Union[float, str] = Field(..., description="Total cumulative charge amount (numeric float or blank string for new customers)")


class ChurnResponse(BaseModel):
    churn_probability: float = Field(..., description="Model calculated probability of churn (0.0 to 1.0)")
    churn_prediction: str = Field(..., description="Binary churn decision ('Yes' or 'No')")
    risk_tier: str = Field(..., description="Calculated risk tier ('Low', 'Medium', 'High')")
    clv: float = Field(..., description="Estimated Customer Lifetime Value ($)")
    billing_risk: float = Field(..., description="Billing cash-collection friction risk indicator")
    high_risk_profile: int = Field(..., description="1 if account represents high risk profile, 0 otherwise")
    persona: str = Field(..., description="Identified customer persona cluster")
    recommended_interventions: List[str] = Field(..., description="Priority customer retention actions based on profile indicators")


@app.get("/health")
def health_check():
    """Verify API availability and model loading status."""
    if MODEL_LOADED:
        return {"status": "healthy", "model_loaded": True}
    else:
        return {"status": "unhealthy", "model_loaded": False, "error": "Model files missing in models/ registry."}


@app.post("/predict", response_model=ChurnResponse)
def predict(customer: CustomerData):
    """
    Score a single customer's profile to predict churn probability and return
    tailored CRM retention actions.
    """
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Prediction service unavailable: Model artifacts could not be loaded."
        )

    # Convert Pydantic object to dictionary
    customer_dict = customer.model_dump()

    try:
        # Generate model scoring output
        scoring = predict_churn(
            customer=customer_dict,
            model=model,
            preprocessor=preprocessor,
            feature_names=feature_names
        )

        # Generate targeted rule-based recommendations for medium/high risk
        interventions = []
        if scoring["risk_tier"] in ["Medium", "High"]:
            if customer.Contract == "Month-to-month":
                interventions.append(
                    "Month-to-Year Contract Migration: Offer a 10% monthly discount or a free broadband speed bump "
                    "in exchange for converting to a 1-year contract terms."
                )
            if customer.InternetService == "Fiber optic":
                interventions.append(
                    "Fiber Optic Quality & Loyalty Credit: Deploy technical connection audit in customer's area "
                    "and apply a proactive $5/month statement credit for 6 months."
                )
            if customer.PaymentMethod == "Electronic check":
                interventions.append(
                    "Auto-Pay Promotion: Offer a one-time $10 statement credit to transition customer from "
                    "manual Electronic Check billing to automatic Credit Card/Bank Transfer billing."
                )
            if customer.tenure <= 12:
                interventions.append(
                    "Early-Tenure Welcoming CRM Journey: Arrange a priority outreach call by customer loyalty representative "
                    "at month 3/6 and dispatch a first-year anniversary bonus offer."
                )
            
            # Default recommendation if no rules triggered but customer is flagged
            if not interventions:
                interventions.append(
                    "Proactive Customer Care: Engage customer via outbound satisfaction survey and offer a general 5% loyalty discount."
                )
        else:
            interventions.append("No retention action required: customer exhibits stable customer profile.")

        return {
            "churn_probability": scoring["churn_probability"],
            "churn_prediction": scoring["churn_prediction"],
            "risk_tier": scoring["risk_tier"],
            "clv": scoring["clv"],
            "billing_risk": scoring["billing_risk"],
            "high_risk_profile": scoring["high_risk_profile"],
            "persona": scoring["persona"],
            "recommended_interventions": interventions
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference pipeline execution error: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
