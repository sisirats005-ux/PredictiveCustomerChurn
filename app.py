"""
ConnectTel Customer Churn Prediction - Streamlit Deployment App.

An interactive dashboard that loads the serialized Logistic Regression model
and preprocessing pipeline, and provides:
- A single-customer scoring tool (Churn Probability, Risk Tier, CLV, Billing
  Risk, top SHAP-aligned risk factors, and a recommended retention action)
- An Executive Dashboard summarizing risk distribution and expected revenue
  saved across the full customer base
"""

import json
import os

import pandas as pd
import streamlit as st

from src.predict import load_artifacts, predict_churn, get_decision_threshold
from src.preprocessing import load_data, clean_data
from src.feature_engineering import create_features
from src.utils import load_config

st.set_page_config(page_title="ConnectTel Churn Dashboard", page_icon="📡", layout="centered")

# Business assumptions reused from reports/business_insight_report.md, Section 5,
# so every dollar figure shown in this app is consistent with the ROI model.
ANNUAL_VALUE = 780.0   # ARPU annual value per retained customer
CAC = 250.0            # Customer acquisition cost avoided by retention
OFFER_COST = 50.0      # Average retention offer cost
CAMPAIGN_SUCCESS_RATE = 0.25  # Fraction of targeted true churners who accept and stay


@st.cache_resource
def get_artifacts():
    """Load and cache the model, preprocessor, and feature names for the app session."""
    return load_artifacts()


@st.cache_data
def get_model_metadata():
    """Load Logistic Regression metadata (metrics, threshold, params) for display, if present."""
    metadata_path = os.path.join("models", "logistic_regression_metadata.json")
    if not os.path.exists(metadata_path):
        return None
    try:
        with open(metadata_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data
def score_full_customer_base(_model, _preprocessor, _feature_names, threshold):
    """
    Batch-score the entire historical dataset with the deployed model, for
    the Executive Dashboard tab. Returns a DataFrame with churn probability
    and risk tier per customer, plus aggregate KPI figures.
    """
    config = load_config()
    df_raw = load_data(config["data"]["raw_path"])
    df_clean, _ = clean_data(df_raw)
    df_eng = create_features(df_clean)

    X = df_eng.drop(columns=[config["data"]["target_col"]])
    X_trans = _preprocessor.transform(X)
    X_df = pd.DataFrame(X_trans, columns=_feature_names, index=X.index)

    probs = _model.predict_proba(X_df)[:, 1]
    preds = (probs >= threshold).astype(int)

    risk_tier = pd.cut(
        probs, bins=[-0.01, 0.3, 0.6, 1.01], labels=["Low", "Medium", "High"]
    )

    scored = pd.DataFrame({
        "churn_probability": probs,
        "churn_prediction": preds,
        "risk_tier": risk_tier,
    })
    return scored


st.title("📡 ConnectTel Customer Churn Predictor")
st.caption(
    "Interactive deployment of the Logistic Regression model (final selected model per the project report). "
    "Input customer attributes below to score risk and calculate retention ROI."
)

metadata = get_model_metadata()
if metadata and "metrics" in metadata:
    with st.expander("ℹ️ Model Info (matches project report)"):
        m = metadata["metrics"]
        i1, i2, i3, i4 = st.columns(4)
        i1.metric("Accuracy", f"{m.get('Accuracy', 0):.3f}")
        i2.metric("Precision", f"{m.get('Precision', 0):.3f}")
        i3.metric("Recall", f"{m.get('Recall', 0):.3f}")
        i4.metric("ROC AUC", f"{m.get('ROC_AUC', 0):.3f}")
        st.caption(
            f"Model type: {metadata.get('model_type', 'Logistic Regression')} · "
            f"scikit-learn: {metadata.get('sklearn_version', 'n/a')} · "
            f"random_state: {metadata.get('random_state', 'n/a')}"
        )
        thr_info = metadata.get("threshold_optimization")
        if thr_info:
            st.markdown(
                f"**Decision threshold: {thr_info['recommended_threshold']}** "
                "(cost-sensitive optimization, not the naive 0.5 default — see the "
                "Threshold Optimization section of the project report for the full rationale)."
            )
        st.caption(
            "Since churn datasets are imbalanced, Accuracy alone can be misleading. "
            "ROC-AUC, Precision, Recall, and F1 were used as the primary evaluation metrics, "
            "and the Logistic Regression model was trained with class_weight='balanced' to "
            "correct for the ~73%/27% class imbalance without needing synthetic oversampling (SMOTE)."
        )

try:
    model, preprocessor, feature_names = get_artifacts()
except FileNotFoundError as e:
    st.error(
        f"Model artifacts not found under `models/`.\n\n**Details:** {e}\n\n"
        "Ensure `models/logistic_regression_model.joblib`, "
        "`models/preprocessing_pipeline.joblib`, and `models/feature_names.joblib` "
        "are present alongside `app.py`."
    )
    st.stop()
except RuntimeError as e:
    st.error(f"❌ Could not load the Logistic Regression model.\n\n**Details:** {e}")
    st.stop()
except Exception as e:  # noqa: BLE001
    st.error(f"❌ An unexpected error occurred while loading the model: {e}")
    st.stop()

decision_threshold = get_decision_threshold()

predict_tab, dashboard_tab = st.tabs(["🔮 Predict Churn Risk", "📊 Executive Dashboard"])

# --------------------------------------------------------------------------
# Tab 1: Single-customer prediction
# --------------------------------------------------------------------------
with predict_tab:
    with st.form("customer_form"):
        st.subheader("Account Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        with col2:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=0.5)
        with col3:
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=840.0, step=1.0)

        col4, col5, col6 = st.columns(3)
        with col4:
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        with col5:
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        with col6:
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )

        st.subheader("Customer Demographics")
        col7, col8, col9 = st.columns(3)
        with col7:
            gender = st.selectbox("Gender", ["Female", "Male"])
        with col8:
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        with col9:
            partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])

        st.subheader("Subscribed Services")
        col10, col11 = st.columns(2)
        with col10:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        with col11:
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

        col12, col13 = st.columns(2)
        with col12:
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        with col13:
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        submitted = st.form_submit_button("Run Risk Prediction", type="primary")

    if submitted:
        customer = {
            "gender": gender,
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        # Predict using inference module (re-calculates new advanced features automatically)
        try:
            result = predict_churn(customer, model=model, preprocessor=preprocessor, feature_names=feature_names)
        except Exception as e:  # noqa: BLE001
            st.error(f"❌ Prediction failed: {e}")
            st.stop()

        st.divider()
        st.header("Risk Analytics & Diagnostics")

        m1, m2, m3 = st.columns(3)
        m1.metric(label="Churn Probability", value=f"{result['churn_probability']:.1%}")
        m2.metric(label="Risk Tier", value=result["risk_tier"])
        m3.metric(label="Estimated Customer LTV", value=f"${result['clv']:,.2f}")

        st.progress(min(max(result["churn_probability"], 0.0), 1.0))
        st.caption(
            f"Decision boundary: customer is classified **{result['churn_prediction']}** "
            f"at the cost-optimized threshold of {result['decision_threshold']:.2f} "
            "(not the naive 0.5 default)."
        )

        st.markdown("### Profile Segment Analysis")
        c1, c2 = st.columns(2)
        c1.info(f"**Identified Persona:**  \n{result['persona']}")

        risk_level = "Elevated Billing Risk" if result["billing_risk"] > 50.0 else "Stable Billing Friction"
        c2.warning(f"**Billing Risk Rating:**  \n{risk_level} (${result['billing_risk']:.2f})")

        st.divider()
        st.header("Explainable AI: Why This Prediction?")
        st.markdown("**Top Risk Factors**")
        for factor in result["top_risk_factors"]:
            st.markdown(f"- {factor}")
        st.markdown("**Recommended Action**")
        st.success(result["recommended_action"])

        st.divider()
        st.header("Recommended Retention Interventions")

        interventions = []
        if result["risk_tier"] in ["Medium", "High"]:
            if contract == "Month-to-month":
                interventions.append(
                    "👉 **Month-to-Year Contract Migration**: Offer a 10% monthly discount or a free speed bump "
                    "in exchange for converting to a 1-year contract."
                )
            if internet_service == "Fiber optic":
                interventions.append(
                    "👉 **Fiber Quality Stability Audit**: Dispatch technical connection check in customer's area "
                    "and apply a proactive $5/month loyalty credit for 6 months."
                )
            if payment_method == "Electronic check":
                interventions.append(
                    "👉 **Auto-Pay Enrollment Campaign**: Offer a one-time $10 statement credit to transition customer "
                    "to Credit Card or Bank Auto-Pay billing."
                )
            if tenure <= 12:
                interventions.append(
                    "👉 **Early-Tenure CRM Welcome Journey**: Arrange a priority check-in call by a loyalty agent "
                    "at month 3/6 and dispatch a first-year anniversary bonus offer."
                )

            if not interventions:
                interventions.append("👉 **Outbound Care Outreach**: Call customer to assess satisfaction and offer a general 5% loyalty discount.")

            st.error("⚠️ **Proactive Retention Action Required**")
            for campaign in interventions:
                st.markdown(campaign)
        else:
            st.success("✅ **Stable Account Profile**")
            st.write("Customer exhibits low churn probability. Maintain standard service sequence; no promotional interventions required.")

# --------------------------------------------------------------------------
# Tab 2: Executive Dashboard
# --------------------------------------------------------------------------
with dashboard_tab:
    st.header("📊 Executive Risk & Revenue Dashboard")
    st.caption(
        "Full customer base scored with the deployed Logistic Regression model "
        f"at the cost-optimized decision threshold ({decision_threshold:.2f})."
    )

    try:
        scored = score_full_customer_base(model, preprocessor, feature_names, decision_threshold)
    except Exception as e:  # noqa: BLE001
        st.error(f"❌ Could not score the full customer base: {e}")
        st.stop()

    total_customers = len(scored)
    high_risk = int((scored["risk_tier"] == "High").sum())
    medium_risk = int((scored["risk_tier"] == "Medium").sum())
    low_risk = int((scored["risk_tier"] == "Low").sum())
    flagged_churners = int(scored["churn_prediction"].sum())

    # Expected revenue saved: only customers flagged "Yes" at the deployed
    # threshold are targeted; a fraction (CAMPAIGN_SUCCESS_RATE) of the true
    # churners among them are assumed to accept the offer and stay. This
    # mirrors the same methodology as reports/business_insight_report.md
    # Section 5, scaled to this dataset instead of the illustrative 100k base.
    # Precision at the deployed threshold (from the metadata threshold sweep)
    # estimates what fraction of flagged customers are true churners.
    precision_estimate = 0.434 if abs(decision_threshold - 0.30) < 1e-6 else 0.504
    estimated_true_churners = flagged_churners * precision_estimate
    customers_retained = estimated_true_churners * CAMPAIGN_SUCCESS_RATE
    revenue_saved = customers_retained * (ANNUAL_VALUE + CAC)
    campaign_cost = flagged_churners * OFFER_COST
    net_benefit = revenue_saved - campaign_cost

    st.markdown("#### Customer Risk Distribution")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customers", f"{total_customers:,}")
    k2.metric("🔴 High Risk", f"{high_risk:,}")
    k3.metric("🟠 Medium Risk", f"{medium_risk:,}")
    k4.metric("🟢 Low Risk", f"{low_risk:,}")

    st.markdown("#### Retention Campaign Impact (Estimated)")
    r1, r2, r3 = st.columns(3)
    r1.metric("Customers Flagged for Outreach", f"{flagged_churners:,}")
    r2.metric("Est. Customers Retained", f"{customers_retained:,.0f}")
    r3.metric("Est. Revenue Saved", f"${revenue_saved:,.0f}")

    st.caption(
        f"Assumptions: ARPU annual value ${ANNUAL_VALUE:.0f}, CAC avoided ${CAC:.0f}, "
        f"retention offer cost ${OFFER_COST:.0f}, campaign success rate {CAMPAIGN_SUCCESS_RATE:.0%}, "
        f"estimated Precision at this threshold {precision_estimate:.1%} "
        "(see reports/business_insight_report.md, Section 5, for the full ROI methodology). "
        f"Estimated campaign cost: ${campaign_cost:,.0f} · Estimated net benefit: ${net_benefit:,.0f}."
    )

    st.markdown("#### Risk Tier Breakdown")
    tier_counts = scored["risk_tier"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
    st.bar_chart(tier_counts)

st.divider()
st.caption(
    "⚠️ This interface runs predictions using the containerized MLOps pipeline. "
    "See `api.py` and `Dockerfile` for REST endpoint deployment parameters. "
    "Engineering: Dockerized deployment · REST API · Automated testing (pytest) · "
    "GitHub Actions CI/CD · Modular architecture · Centralized configuration (config.yaml)."
)
