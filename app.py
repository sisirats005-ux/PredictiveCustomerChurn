"""Enterprise Streamlit dashboard for ConnectTel churn prediction."""

import json
import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.business_rules import (
    ANNUAL_VALUE,
    CAC,
    CAMPAIGN_SUCCESS_RATE,
    OFFER_COST,
    batch_roi,
    campaign_name,
    estimate_retention_roi,
    recommended_interventions,
    risk_color,
)
from src.feature_engineering import create_features
from src.predict import get_decision_threshold, load_artifacts, predict_churn
from src.preprocessing import clean_data, load_data
from src.utils import load_config

st.set_page_config(page_title="ConnectTel AI Churn Studio", page_icon="📡", layout="wide")

RISK_ORDER = ["Low", "Medium", "High"]
DEFAULT_CUSTOMER = {
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
LOW_RISK_CUSTOMER = {
    **DEFAULT_CUSTOMER,
    "tenure": 48,
    "Contract": "Two year",
    "InternetService": "DSL",
    "TechSupport": "Yes",
    "PaymentMethod": "Credit card (automatic)",
    "MonthlyCharges": 55.0,
    "TotalCharges": 2640.0,
}


st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .hero {background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #06b6d4 100%); padding: 2rem; border-radius: 24px; color: white; margin-bottom: 1rem;}
    .hero h1 {font-size: 2.6rem; margin: 0 0 .4rem 0; letter-spacing: -0.04em;}
    .hero p {font-size: 1.05rem; opacity: .92; max-width: 900px;}
    .kpi-card {background: white; border: 1px solid #e2e8f0; border-radius: 18px; padding: 1.1rem; box-shadow: 0 10px 28px rgba(15,23,42,.08);}
    .kpi-label {color: #64748b; font-size: .82rem; text-transform: uppercase; letter-spacing: .08em; margin-bottom: .35rem;}
    .kpi-value {font-size: 1.7rem; font-weight: 800; color: #0f172a;}
    .risk-badge {display: inline-block; color: white; padding: .45rem .8rem; border-radius: 999px; font-weight: 800; letter-spacing: .02em;}
    .panel {background: #ffffff; border: 1px solid #e2e8f0; border-radius: 20px; padding: 1.2rem; box-shadow: 0 8px 22px rgba(15,23,42,.06); margin-bottom: 1rem;}
    .muted {color: #64748b; font-size: .92rem;}
    .action-card {border-left: 6px solid #2563eb; background: #eff6ff; padding: 1rem; border-radius: 16px; margin-bottom: .7rem;}
    .stButton > button {border-radius: 999px; font-weight: 700; padding: .55rem 1rem;}
    div[data-testid="stMetric"] {background: white; border: 1px solid #e2e8f0; border-radius: 16px; padding: 1rem; box-shadow: 0 8px 22px rgba(15,23,42,.05);}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_artifacts():
    """Load and cache model artifacts once per Streamlit session."""
    return load_artifacts()


@st.cache_data(show_spinner=False)
def get_model_metadata():
    """Load deployed model metadata for KPI display."""
    metadata_path = os.path.join("models", "logistic_regression_metadata.json")
    if not os.path.exists(metadata_path):
        return {}
    with open(metadata_path, "r") as f:
        return json.load(f)


@st.cache_data(show_spinner="Scoring historical customer base...")
def score_full_customer_base(_model, _preprocessor, _feature_names, threshold):
    """Batch-score the historical base for executive dashboard visualizations."""
    config = load_config()
    df_raw = load_data(config["data"]["raw_path"])
    df_clean, _ = clean_data(df_raw)
    df_eng = create_features(df_clean)
    X = df_eng.drop(columns=[config["data"]["target_col"]])
    X_trans = _preprocessor.transform(X)
    if X_trans.shape[1] != len(_feature_names):
        raise RuntimeError(
            f"Feature mismatch while scoring portfolio: got {X_trans.shape[1]} columns, "
            f"expected {len(_feature_names)}."
        )
    X_df = pd.DataFrame(X_trans, columns=_feature_names, index=X.index)
    probs = _model.predict_proba(X_df)[:, 1]
    return pd.DataFrame(
        {
            "churn_probability": probs,
            "churn_prediction": (probs >= threshold).astype(int),
            "risk_tier": pd.cut(probs, bins=[-0.01, 0.3, 0.6, 1.01], labels=RISK_ORDER),
            "Contract": X["Contract"].values,
            "InternetService": X["InternetService"].values,
            "PaymentMethod": X["PaymentMethod"].values,
            "MonthlyCharges": X["MonthlyCharges"].values,
        }
    )


def reset_customer():
    st.session_state.customer_defaults = DEFAULT_CUSTOMER.copy()
    st.session_state.prediction_result = None
    st.session_state.prediction_customer = None


def load_example_customer():
    st.session_state.customer_defaults = DEFAULT_CUSTOMER.copy()


def load_low_risk_customer():
    st.session_state.customer_defaults = LOW_RISK_CUSTOMER.copy()


def build_customer_from_form():
    """Build a validated raw customer payload from Streamlit widget state."""
    if st.session_state.total_charges < 0 or st.session_state.monthly_charges < 0:
        raise ValueError("Charges must be non-negative.")
    if st.session_state.tenure == 0 and st.session_state.total_charges > 0:
        raise ValueError("A zero-tenure customer should not have positive Total Charges.")
    return {
        "gender": st.session_state.gender,
        "SeniorCitizen": 1 if st.session_state.senior_citizen == "Yes" else 0,
        "Partner": st.session_state.partner,
        "Dependents": st.session_state.dependents,
        "tenure": int(st.session_state.tenure),
        "PhoneService": st.session_state.phone_service,
        "MultipleLines": st.session_state.multiple_lines,
        "InternetService": st.session_state.internet_service,
        "OnlineSecurity": st.session_state.online_security,
        "OnlineBackup": st.session_state.online_backup,
        "DeviceProtection": st.session_state.device_protection,
        "TechSupport": st.session_state.tech_support,
        "StreamingTV": st.session_state.streaming_tv,
        "StreamingMovies": st.session_state.streaming_movies,
        "Contract": st.session_state.contract,
        "PaperlessBilling": st.session_state.paperless_billing,
        "PaymentMethod": st.session_state.payment_method,
        "MonthlyCharges": float(st.session_state.monthly_charges),
        "TotalCharges": float(st.session_state.total_charges),
    }


def probability_gauge(probability, threshold):
    """Build an interactive Plotly churn probability gauge."""
    return go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 42}},
            delta={"reference": threshold * 100, "suffix": " pts vs threshold"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": risk_color("High" if probability >= 0.6 else "Medium" if probability >= 0.3 else "Low")},
                "steps": [
                    {"range": [0, 30], "color": "#dcfce7"},
                    {"range": [30, 60], "color": "#fef3c7"},
                    {"range": [60, 100], "color": "#fee2e2"},
                ],
                "threshold": {"line": {"color": "#0f172a", "width": 4}, "value": threshold * 100},
            },
            title={"text": "Churn Probability"},
        )
    ).update_layout(height=300, margin=dict(l=20, r=20, t=50, b=10))


def factors_chart(factors):
    """Create a ranked visual for explanation factors when live SHAP values are unavailable."""
    scores = list(range(len(factors), 0, -1))
    fig = px.bar(
        x=scores,
        y=factors,
        orientation="h",
        labels={"x": "Relative priority", "y": "Risk factor"},
        color=scores,
        color_continuous_scale="Blues",
    )
    fig.update_layout(showlegend=False, height=280, margin=dict(l=10, r=10, t=20, b=10), yaxis={"autorange": "reversed"})
    return fig


def report_dataframe(customer, result, roi, campaign):
    """Create a one-row downloadable prediction report."""
    return pd.DataFrame([
        {
            "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
            "churn_probability": result["churn_probability"],
            "risk_tier": result["risk_tier"],
            "churn_prediction": result["churn_prediction"],
            "model_confidence": result["model_confidence"],
            "customer_lifetime_value": result["clv"],
            "expected_saved_value": roi["expected_saved_value"],
            "offer_cost": roi["offer_cost"],
            "estimated_net_roi": roi["net_roi"],
            "campaign": campaign,
            "recommended_action": result["recommended_action"],
            "top_risk_factors": " | ".join(result["top_risk_factors"]),
            **customer,
        }
    ])


def render_prediction_result(result, customer):
    """Render the complete prediction results experience."""
    probability = result["churn_probability"]
    confidence = result.get("model_confidence", max(probability, 1 - probability))
    roi = estimate_retention_roi(probability, result["clv"])
    campaign = campaign_name(customer, result["risk_tier"])
    interventions = recommended_interventions(customer, result["risk_tier"])

    st.toast("Prediction completed successfully", icon="✅")
    color = risk_color(result["risk_tier"])
    st.markdown(
        f"<span class='risk-badge' style='background:{color}'>{result['risk_tier']} Risk</span>",
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Churn Probability", f"{probability:.1%}")
    k2.metric("Model Confidence", f"{confidence:.1%}")
    k3.metric("Estimated CLV", f"${result['clv']:,.0f}")
    k4.metric("Estimated Net ROI", f"${roi['net_roi']:,.0f}")

    left, right = st.columns([1.05, 1])
    with left:
        st.plotly_chart(probability_gauge(probability, result["decision_threshold"]), use_container_width=True)
        st.caption(
            f"Decision threshold: {result['decision_threshold']:.2f}. Prediction: "
            f"**{result['churn_prediction']}**. Confidence is measured as distance from binary uncertainty."
        )
    with right:
        st.markdown("<div class='panel'><h3>Interactive Customer Summary</h3>", unsafe_allow_html=True)
        s1, s2 = st.columns(2)
        s1.write(f"**Persona:** {result['persona']}")
        s1.write(f"**Contract:** {customer['Contract']}")
        s1.write(f"**Tenure:** {customer['tenure']} months")
        s2.write(f"**Internet:** {customer['InternetService']}")
        s2.write(f"**Payment:** {customer['PaymentMethod']}")
        s2.write(f"**Billing Risk:** ${result['billing_risk']:,.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

    explain_col, action_col = st.columns([1, 1])
    with explain_col:
        st.markdown("### AI Explanation Panel")
        st.info("Live SHAP values are computed offline for model governance; this panel shows SHAP-aligned drivers used by the deployed rules layer.")
        st.plotly_chart(factors_chart(result["top_risk_factors"]), use_container_width=True)
        for factor in result["top_risk_factors"]:
            st.markdown(f"- {factor}")
        shap_path = os.path.join("outputs", "plots", "shap_feature_importance_plot.png")
        if os.path.exists(shap_path):
            st.image(shap_path, caption="Offline SHAP global feature importance artifact", use_container_width=True)
    with action_col:
        st.markdown("### Recommended Retention Action")
        st.markdown(f"<div class='action-card'><b>Suggested campaign:</b><br>{campaign}</div>", unsafe_allow_html=True)
        st.success(result["recommended_action"])
        for item in interventions:
            st.markdown(f"<div class='action-card'>{item}</div>", unsafe_allow_html=True)

    st.markdown("### ROI Summary")
    r1, r2, r3 = st.columns(3)
    r1.metric("Expected Saved Value", f"${roi['expected_saved_value']:,.0f}")
    r2.metric("Offer Cost", f"${roi['offer_cost']:,.0f}")
    r3.metric("Net Retention ROI", f"${roi['net_roi']:,.0f}")

    csv = report_dataframe(customer, result, roi, campaign).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download prediction report (CSV)",
        data=csv,
        file_name="connecttel_churn_prediction_report.csv",
        mime="text/csv",
        type="primary",
    )


def render_customer_form():
    """Render the prediction input form with example/reset controls."""
    defaults = st.session_state.customer_defaults
    st.markdown("### Score a Customer")
    st.caption("Use the example buttons or adjust the fields to score an individual account in real time.")

    cta1, cta2, cta3 = st.columns([1, 1, 4])
    cta1.button("Example high risk", on_click=load_example_customer)
    cta2.button("Reset form", on_click=reset_customer)
    cta3.button("Example low risk", on_click=load_low_risk_customer)

    with st.form("customer_form", clear_on_submit=False):
        st.markdown("#### Account & Billing")
        a1, a2, a3 = st.columns(3)
        a1.number_input("Tenure (months)", min_value=0, max_value=100, value=int(defaults["tenure"]), key="tenure")
        a2.number_input("Monthly Charges ($)", min_value=0.0, value=float(defaults["MonthlyCharges"]), step=0.5, key="monthly_charges")
        a3.number_input("Total Charges ($)", min_value=0.0, value=float(defaults["TotalCharges"]), step=1.0, key="total_charges")

        b1, b2, b3 = st.columns(3)
        b1.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(defaults["Contract"]), key="contract")
        b2.selectbox("Paperless Billing", ["Yes", "No"], index=["Yes", "No"].index(defaults["PaperlessBilling"]), key="paperless_billing")
        b3.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(defaults["PaymentMethod"]), key="payment_method")

        st.markdown("#### Demographics")
        d1, d2, d3, d4 = st.columns(4)
        d1.selectbox("Gender", ["Female", "Male"], index=["Female", "Male"].index(defaults["gender"]), key="gender")
        d2.selectbox("Senior Citizen", ["No", "Yes"], index=defaults["SeniorCitizen"], key="senior_citizen")
        d3.selectbox("Partner", ["Yes", "No"], index=["Yes", "No"].index(defaults["Partner"]), key="partner")
        d4.selectbox("Dependents", ["No", "Yes"], index=["No", "Yes"].index(defaults["Dependents"]), key="dependents")

        st.markdown("#### Services")
        svc1, svc2, svc3 = st.columns(3)
        svc1.selectbox("Phone Service", ["Yes", "No"], index=["Yes", "No"].index(defaults["PhoneService"]), key="phone_service")
        svc1.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=["No", "Yes", "No phone service"].index(defaults["MultipleLines"]), key="multiple_lines")
        svc2.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=["DSL", "Fiber optic", "No"].index(defaults["InternetService"]), key="internet_service")
        svc2.selectbox("Online Security", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["OnlineSecurity"]), key="online_security")
        svc2.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["OnlineBackup"]), key="online_backup")
        svc3.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["DeviceProtection"]), key="device_protection")
        svc3.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["TechSupport"]), key="tech_support")
        svc3.selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["StreamingTV"]), key="streaming_tv")
        st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["StreamingMovies"]), key="streaming_movies")

        submitted = st.form_submit_button("Run Risk Prediction", type="primary", use_container_width=True)

    return submitted


def render_executive_dashboard(model, preprocessor, feature_names, decision_threshold):
    """Render the executive portfolio dashboard with interactive Plotly charts."""
    st.markdown("### Executive Risk & Revenue Dashboard")
    st.caption(f"Full customer base scored at the deployed cost-sensitive decision threshold ({decision_threshold:.2f}).")

    scored = score_full_customer_base(model, preprocessor, feature_names, decision_threshold)
    total_customers = len(scored)
    tier_counts = scored["risk_tier"].value_counts().reindex(RISK_ORDER).fillna(0).astype(int)
    flagged_churners = int(scored["churn_prediction"].sum())
    precision_estimate = 0.434 if abs(decision_threshold - 0.30) < 1e-6 else 0.504
    roi = batch_roi(flagged_churners, precision_estimate)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customers", f"{total_customers:,}")
    k2.metric("High Risk", f"{tier_counts['High']:,}")
    k3.metric("Flagged for Outreach", f"{flagged_churners:,}")
    k4.metric("Est. Net Benefit", f"${roi['net_benefit']:,.0f}")

    chart1, chart2 = st.columns([1, 1])
    with chart1:
        fig = px.pie(
            names=tier_counts.index,
            values=tier_counts.values,
            hole=0.55,
            color=tier_counts.index,
            color_discrete_map={"Low": "#16a34a", "Medium": "#f59e0b", "High": "#dc2626"},
            title="Risk Tier Mix",
        )
        fig.update_layout(height=390, legend_title_text="Risk")
        st.plotly_chart(fig, use_container_width=True)
    with chart2:
        hist = px.histogram(
            scored,
            x="churn_probability",
            nbins=30,
            color="risk_tier",
            color_discrete_map={"Low": "#16a34a", "Medium": "#f59e0b", "High": "#dc2626"},
            title="Churn Probability Distribution",
        )
        hist.add_vline(x=decision_threshold, line_dash="dash", line_color="#0f172a")
        hist.update_layout(height=390, xaxis_tickformat=".0%", bargap=0.03)
        st.plotly_chart(hist, use_container_width=True)

    segment = (
        scored.groupby(["Contract", "risk_tier"], observed=False)
        .size()
        .reset_index(name="customers")
    )
    fig_segment = px.bar(
        segment,
        x="Contract",
        y="customers",
        color="risk_tier",
        barmode="group",
        color_discrete_map={"Low": "#16a34a", "Medium": "#f59e0b", "High": "#dc2626"},
        title="Risk Concentration by Contract Type",
    )
    fig_segment.update_layout(height=420)
    st.plotly_chart(fig_segment, use_container_width=True)

    st.markdown("### Retention ROI Model")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Est. True Churners", f"{roi['estimated_true_churners']:,.0f}")
    r2.metric("Est. Customers Retained", f"{roi['customers_retained']:,.0f}")
    r3.metric("Revenue Saved", f"${roi['revenue_saved']:,.0f}")
    r4.metric("Campaign Cost", f"${roi['campaign_cost']:,.0f}")
    st.caption(
        f"Assumptions: annual ARPU value ${ANNUAL_VALUE:.0f}, CAC avoided ${CAC:.0f}, "
        f"offer cost ${OFFER_COST:.0f}, campaign success {CAMPAIGN_SUCCESS_RATE:.0%}, "
        f"precision estimate {precision_estimate:.1%}."
    )


def render_model_info(metadata):
    """Render model governance metrics from metadata."""
    with st.expander("Model governance and deployment metadata", expanded=False):
        metrics = metadata.get("metrics", {})
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics.get('Accuracy', 0):.3f}")
        m2.metric("Precision", f"{metrics.get('Precision', 0):.3f}")
        m3.metric("Recall", f"{metrics.get('Recall', 0):.3f}")
        m4.metric("ROC AUC", f"{metrics.get('ROC_AUC', 0):.3f}")
        threshold = metadata.get("threshold_optimization", {}).get("recommended_threshold", get_decision_threshold())
        st.write(
            f"Deployed model: **{metadata.get('model_type', 'LogisticRegression')}** · "
            f"scikit-learn artifact version: **{metadata.get('sklearn_version', 'unknown')}** · "
            f"decision threshold: **{threshold:.2f}**"
        )
        st.caption("SHAP plots remain available under outputs/plots for offline governance and audit review.")


if "customer_defaults" not in st.session_state:
    st.session_state.customer_defaults = DEFAULT_CUSTOMER.copy()
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "prediction_customer" not in st.session_state:
    st.session_state.prediction_customer = None

st.markdown(
    """
    <div class='hero'>
      <h1>ConnectTel AI Churn Studio</h1>
      <p>Enterprise retention intelligence for real-time churn scoring, explainable AI diagnostics, and campaign ROI planning.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model, preprocessor, feature_names = get_artifacts()
    st.success("Model artifacts loaded and validated.", icon="✅")
except Exception as exc:  # noqa: BLE001
    st.error(f"Prediction service is unavailable: {exc}")
    st.info("Run the app in the pinned requirements.txt environment or retrain artifacts with the active runtime.")
    st.stop()

metadata = get_model_metadata()
decision_threshold = get_decision_threshold()
render_model_info(metadata)

predict_tab, dashboard_tab = st.tabs(["Predict", "Executive Dashboard"])

with predict_tab:
    form_col, result_col = st.columns([0.95, 1.05], gap="large")
    with form_col:
        submitted = render_customer_form()
        if submitted:
            try:
                customer = build_customer_from_form()
                with st.spinner("Running model inference and ROI analysis..."):
                    st.session_state.prediction_result = predict_churn(
                        customer, model=model, preprocessor=preprocessor, feature_names=feature_names
                    )
                    st.session_state.prediction_customer = customer
            except ValueError as exc:
                st.warning(str(exc), icon="⚠️")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Prediction failed: {exc}", icon="🚨")
    with result_col:
        if st.session_state.prediction_result and st.session_state.prediction_customer:
            render_prediction_result(st.session_state.prediction_result, st.session_state.prediction_customer)
        else:
            st.markdown("<div class='panel'><h3>Ready for prediction</h3><p class='muted'>Submit a customer profile to see churn probability, confidence, explanations, retention actions, and ROI.</p></div>", unsafe_allow_html=True)
            st.plotly_chart(probability_gauge(0.0, decision_threshold), use_container_width=True)

with dashboard_tab:
    try:
        render_executive_dashboard(model, preprocessor, feature_names, decision_threshold)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not render executive dashboard: {exc}")

st.caption(
    "Production stack: Streamlit enterprise dashboard · FastAPI REST service · Docker · SHAP artifacts · "
    "pytest regression tests · GitHub Actions CI/CD · centralized config.yaml."
)
