"""
Enterprise Streamlit dashboard for ConnectTel churn prediction ("ConnectTel
AI Churn Studio").

Renders three screens (routed via `st.session_state.active_tab`, see the
"MAIN PAGE RENDER" section at the bottom of this file):
  - Predict:             single-customer scoring form + result panels.
  - Explain:              coefficient-based local explanation + offline SHAP plots.
  - Executive Dashboard:  portfolio-wide batch scoring and ROI summary.

All inference goes through `src.predict.predict_churn`, which loads
`models/logistic_regression_model.joblib` -- this file only handles
presentation, input collection, and layout. See README.md, Section 5
("Production & MLOps Infrastructure") for the full architecture.
"""
print("========== APP.PY EXECUTED ==========")
import json
import logging
import os
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from components.sidebar import (
    render_sidebar_drawer
)
from src.business_rules import (
    ANNUAL_VALUE,
    CAC,
    CAMPAIGN_SUCCESS_RATE,
    OFFER_COST,
    batch_roi,
    campaign_name,
    confidence_score,
    estimate_retention_roi,
    recommended_interventions,
    risk_color,
)
from src.feature_engineering import create_features
from src.predict import get_decision_threshold, load_artifacts, predict_churn, prepare_customer_frame
from src.preprocessing import clean_data, load_data
from src.utils import load_config, setup_logger

st.set_page_config(page_title="ConnectTel AI Churn Studio", layout="wide", initial_sidebar_state="expanded")

# Module-level logger: mirrors the training-pipeline logger convention in
# src/utils.py (console + reports/pipeline.log) so Streamlit-side failures
# (model load errors, per-customer inference errors, executive dashboard
# batch-scoring errors) land in the same log trail as the training run,
# instead of only ever being visible to whoever is looking at the browser
# tab when the error banner renders.
logger = setup_logger(name="churn_project.app")


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
def render_metric_card_component(
    title,
    value,
    accent="#3B82F6",
    badge="",
    element_id="metric",
    value_color="#FFFFFF",
    prefix="",
    suffix="",
    numeric_target=None,
):
    """
    Render a KPI card.
    Temporary recovery implementation.
    """

    st.markdown(
        f"""
        <div style="
            background:#172033;
            border-left:5px solid {accent};
            border-radius:12px;
            padding:18px;
            margin-bottom:12px;
            box-shadow:0 6px 20px rgba(0,0,0,.25);
        ">

            <div style="
                font-size:12px;
                color:#94A3B8;
                font-weight:600;
                letter-spacing:.05em;
                text-transform:uppercase;
            ">
                {title}
            </div>

            <div style="
                margin-top:8px;
                font-size:30px;
                font-weight:700;
                color:{value_color};
            ">
                {value}
            </div>

            <div style="
                margin-top:6px;
                color:#CBD5E1;
                font-size:12px;
            ">
                {badge}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards(result=None, customer=None):
    """Render the 4 horizontal KPI cards with polished motion and clean value animation."""
    if result and customer:
        probability = result["churn_probability"]
        confidence = result.get("model_confidence", confidence_score(probability))
        roi = estimate_retention_roi(probability, result["clv"])
        
        risk_tier = result["risk_tier"]
        prob_color = risk_color(risk_tier)
        prob_bg = "rgba(239, 68, 68, 0.12)" if risk_tier == "High" else "rgba(245, 158, 11, 0.12)" if risk_tier == "Medium" else "rgba(16, 185, 129, 0.12)"
        prob_badge_color = "#F87171" if risk_tier == "High" else "#FBBF24" if risk_tier == "Medium" else "#34D399"
        prob_status = f'<span class="kpi-status-badge" style="background: {prob_bg}; color: {prob_badge_color};">{risk_tier} Risk</span>'

        conf_bg = "rgba(34, 211, 238, 0.12)"
        conf_badge_color = "#22D3EE"
        conf_status_text = "Very High" if confidence >= 0.8 else "High" if confidence >= 0.6 else "Medium" if confidence >= 0.4 else "Low"
        conf_status = f'<span class="kpi-status-badge" style="background: {conf_bg}; color: {conf_badge_color};">{conf_status_text}</span>'

        clv_bg = "rgba(167, 139, 250, 0.12)"
        clv_badge_color = "#A78BFA"
        clv_status = f'<span class="kpi-status-badge" style="background: {clv_bg}; color: {clv_badge_color};">CLV Target</span>'

        roi_bg = "rgba(16, 185, 129, 0.12)"
        roi_badge_color = "#10B981"
        roi_status = f'<span class="kpi-status-badge" style="background: {roi_bg}; color: {roi_badge_color};">Expected ROI</span>'
    else:
        prob_status = '<span class="kpi-status-badge" style="background: rgba(255,255,255,0.06); color: #8890A6;">No score</span>'
        conf_status = '<span class="kpi-status-badge" style="background: rgba(255,255,255,0.06); color: #8890A6;">N/A</span>'
        clv_status = '<span class="kpi-status-badge" style="background: rgba(255,255,255,0.06); color: #8890A6;">CLV Target</span>'
        roi_status = '<span class="kpi-status-badge" style="background: rgba(255,255,255,0.06); color: #8890A6;">ROI Target</span>'

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        if result and customer:
            render_metric_card_component("Churn Probability", f"{probability:.1%}", "#EF4444", prob_status, "kpi-prob", value_color=prob_color, prefix="", suffix="", numeric_target=probability * 100)
        else:
            render_metric_card_component("Churn Probability", "—", "#EF4444", prob_status, "kpi-prob", value_color="#A2AAC0")
    with k2:
        if result and customer:
            render_metric_card_component("Model Confidence", f"{confidence:.1%}", "#22D3EE", conf_status, "kpi-conf", value_color="#EDEFF5", prefix="", suffix="", numeric_target=confidence * 100)
        else:
            render_metric_card_component("Model Confidence", "—", "#22D3EE", conf_status, "kpi-conf", value_color="#EDEFF5")
    with k3:
        if result and customer:
            render_metric_card_component("Estimated CLV", f"${result['clv']:,.0f}", "#A78BFA", clv_status, "kpi-clv", value_color="#EDEFF5", prefix="$", suffix="", numeric_target=float(result['clv']))
        else:
            render_metric_card_component("Estimated CLV", "—", "#A78BFA", clv_status, "kpi-clv", value_color="#EDEFF5")
    with k4:
        if result and customer:
            render_metric_card_component("Expected ROI", f"${roi['net_roi']:,.0f}", "#10B981", roi_status, "kpi-roi", value_color="#10B981" if roi['net_roi'] >= 0 else "#A2AAC0", prefix="$", suffix="", numeric_target=float(roi['net_roi']))
        else:
            render_metric_card_component("Expected ROI", "—", "#10B981", roi_status, "kpi-roi", value_color="#A2AAC0")


def build_ai_executive_summary(result, customer, roi):
    """Create a concise account summary for the prediction view."""
    risk_tier = result["risk_tier"]
    if risk_tier == "High":
        headline = "Immediate outreach is recommended."
        body = ""
        badges = ["Urgent", f"ROI ${roi['net_roi']:,.0f}", "Fast outreach"]
    elif risk_tier == "Medium":
        headline = "Proactive follow-up is advisable."
        body = ""
        badges = ["Watch", f"ROI ${roi['net_roi']:,.0f}", "Proactive"]
    else:
        headline = "Current retention risk is low."
        body = ""
        badges = ["Healthy", f"ROI ${roi['net_roi']:,.0f}", "Maintain"]
    return {"headline": headline, "body": body, "badges": badges}


def render_prediction_result(result, customer):
    """Render the prediction results using radial gauge, customer profile, SHAP, and CRM recommendation timeline."""
    probability = result["churn_probability"]
    roi = estimate_retention_roi(probability, result["clv"])
    campaign = campaign_name(customer, result["risk_tier"])
    interventions = recommended_interventions(customer, result["risk_tier"])
    summary = build_ai_executive_summary(result, customer, roi)

    color = risk_color(result["risk_tier"])
    st.markdown(
        f"<span class='risk-badge' style='background:{color}'>{result['risk_tier']} Risk Account</span>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='panel ai-summary-panel'>", unsafe_allow_html=True)
    st.markdown(f"<div class='ai-summary-headline'>{summary['headline']}</div>", unsafe_allow_html=True)
    if summary["body"]:
        st.markdown(f"<div class='ai-summary-copy'>{summary['body']}</div>", unsafe_allow_html=True)
    st.markdown("<div class='ai-summary-row'>" + "".join([f"<span class='mini-badge'>{badge}</span>" for badge in summary["badges"]]) + "</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Row 2: Radial Gauge (left 45%) and Customer Profile (right 55%) side-by-side (equal height)
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    row2_left, row2_right = st.columns([0.45, 0.55], gap="small")
    with row2_left:
        st.markdown("<div class='panel' style='min-height: 140px !important;'>", unsafe_allow_html=True)
        st.markdown("<h5 class='panel-title'>Risk gauge</h5>", unsafe_allow_html=True)
        st.plotly_chart(probability_gauge(probability, result["decision_threshold"]), width="stretch", config={'responsive': True})
        pred_label = "CHURN (High Churn Risk)" if result['churn_prediction'] == 1 else "RETAIN (Low Churn Risk)"
        pred_color = "#f87171" if result['churn_prediction'] == 1 else "#34d399"
        st.markdown(
            f"<div style='font-size: 0.8rem; color: #A2AAC0; margin-top: 0.2rem; text-align: center;'>"
            f"Decision threshold: <strong>{result['decision_threshold']:.2f}</strong> · "
            f"Prediction: <strong style='color: {pred_color};'>{pred_label}</strong>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with row2_right:
        st.markdown("<div class='panel' style='min-height: 140px !important;'>", unsafe_allow_html=True)
        st.markdown("<h5 class='panel-title'>Profile</h5>", unsafe_allow_html=True)
        chips = [
            f"Persona: {result['persona']}",
            f"Contract: {customer['Contract']}",
            f"Internet: {customer['InternetService']}",
            f"Risk: {result['risk_tier']}",
            f"Tenure: {customer['tenure']}m",
            f"Pay: {customer['PaymentMethod']}",
        ]
        st.markdown("<div class='chip-wrap'>" + "".join([f"<span class='chip-pill active'>{chip}</span>" for chip in chips]) + "</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Row 3: SHAP Feature Importance (left 45%) and CRM Recommendation Timeline (right 55%) side-by-side (equal height)
    row3_left, row3_right = st.columns([0.45, 0.55], gap="small")
    with row3_left:
        st.markdown("<div class='panel' style='min-height: 158px !important;'>", unsafe_allow_html=True)
        st.markdown("<h5 class='panel-title'>Drivers</h5>", unsafe_allow_html=True)
        st.plotly_chart(factors_chart(result["top_risk_factors"]), width="stretch", config={'responsive': True})
        drivers_html = " ".join([f"<span class='mini-badge'>{factor}</span>" for factor in result["top_risk_factors"][:3]])
        st.markdown(f"<div class='ai-summary-row'>{drivers_html}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with row3_right:
        st.markdown("<div class='panel' style='min-height: 158px !important;'>", unsafe_allow_html=True)
        st.markdown("<h5 class='panel-title'>Retention plan</h5>", unsafe_allow_html=True)
        strategy_cards = [
            {
                "title": "Priority Outreach",
                "copy": "Best fit for high-risk month-to-month accounts that need a fast touchpoint.",
                "value": roi["expected_saved_value"] * 0.9,
                "cost": roi["offer_cost"] * 0.6,
                "roi": 6.1,
                "accent": "#22D3EE"
            },
            {
                "title": "Contract Stabilization",
                "copy": "A contract or billing adjustment is the best bet when friction is a major driver.",
                "value": roi["expected_saved_value"] * 0.8,
                "cost": roi["offer_cost"] * 0.7,
                "roi": 5.1,
                "accent": "#10B981"
            },
            {
                "title": "Service Recovery",
                "copy": "Use proactive support and product guidance when service quality is the main issue.",
                "value": roi["expected_saved_value"] * 0.7,
                "cost": roi["offer_cost"] * 0.5,
                "roi": 4.4,
                "accent": "#F59E0B"
            },
        ]
        for card in strategy_cards:
            st.markdown(
                f"""
                <div class="strategy-card" style="--accent: {card['accent']};">
                    <div class="strategy-title">{card['title']}</div>
                    <div class="strategy-metrics">
                        <div><strong>ROI</strong><br>{card['roi']:.1f}x</div>
                        <div><strong>Cost</strong><br>${card['cost']:,.0f}</div>
                        <div><strong>Success</strong><br>{int(card['roi'] * 8)}%</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # Row 4: ROI Economic Cards
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<h5 class='panel-title'>Impact</h5>", unsafe_allow_html=True)
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f"""
        <div class="roi-card" style="--accent: #22D3EE;">
            <div class="roi-info">
                <div class="roi-label">Expected Saved Value</div>
                <div class="roi-val">${roi['expected_saved_value']:,.0f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="roi-card" style="--accent: #F59E0B;">
            <div class="roi-info">
                <div class="roi-label">Campaign Cost</div>
                <div class="roi-val">${roi['offer_cost']:,.0f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with r3:
        roi_txt = "#34D399" if roi['net_roi'] >= 0 else "#F87171"
        st.markdown(f"""
        <div class="roi-card" style="--accent: {roi_txt};">
            <div class="roi-info">
                <div class="roi-label">Net Retention ROI</div>
                <div class="roi-val" style="color: {roi_txt};">${roi['net_roi']:,.0f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    csv = report_dataframe(customer, result, roi, campaign).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download prediction report (CSV)",
        data=csv,
        file_name="connecttel_churn_prediction_report.csv",
        mime="text/csv",
        type="primary",
        width="stretch"
    )


def render_executive_dashboard(model, preprocessor, feature_names, decision_threshold):
    """Render the executive portfolio dashboard inside a cohesive single-sheet BI report container."""
    st.markdown("<div class='bi-report-container'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: space-between; gap: 0.75rem; margin-bottom: 0.1rem;">
            <h2 style="color: #EDEFF5; font-size: 1.04rem; font-weight: 800; margin: 0; letter-spacing: -0.01em;">Executive Risk &amp; Revenue Portfolio Sheet</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    scored = score_full_customer_base(model, preprocessor, feature_names, decision_threshold)
    total_customers = len(scored)
    tier_counts = scored["risk_tier"].value_counts().reindex(RISK_ORDER).fillna(0).astype(int)
    flagged_churners = int(scored["churn_prediction"].sum())
    precision_estimate = 0.434 if abs(decision_threshold - 0.30) < 1e-6 else 0.504
    roi = batch_roi(flagged_churners, precision_estimate)

    # Level 1: Modern Business Cards inside the report container
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4, gap="small")
    
    high_risk_pct = tier_counts["High"] / total_customers if total_customers > 0 else 0.0
    retention_pct = flagged_churners / total_customers if total_customers > 0 else 0.0
    
    with k1:
        st.markdown(
            f"""
            <div class="kpi-card-horizontal" style="box-shadow: none !important; border: 1px solid rgba(255,255,255,0.08) !important; border-left: 3px solid #22D3EE !important; display: flex; flex-direction: column; align-items: flex-start; justify-content: center; height: 71px;">
                <div class="kpi-metric-name">Customer Base</div>
                <div class="kpi-metric-value" style="color: #22D3EE; font-size: 1.45rem; font-weight: 800;">{total_customers:,}</div>
                <div style="font-size: 0.72rem; color: #8890A6; margin-top: 0.2rem;">Total active subscribers</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="kpi-card-horizontal" style="box-shadow: none !important; border: 1px solid rgba(255,255,255,0.08) !important; border-left: 3px solid #EF4444 !important; display: flex; flex-direction: column; align-items: flex-start; justify-content: center; height: 71px;">
                <div class="kpi-metric-name">High Risk Customers</div>
                <div class="kpi-metric-value" style="color: #EF4444; font-size: 1.45rem; font-weight: 800;">{tier_counts["High"]:,}</div>
                <div style="font-size: 0.72rem; color: #8890A6; margin-top: 0.2rem;">{high_risk_pct:.1%} of customer base</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="kpi-card-horizontal" style="box-shadow: none !important; border: 1px solid rgba(255,255,255,0.08) !important; border-left: 3px solid #F59E0B !important; display: flex; flex-direction: column; align-items: flex-start; justify-content: center; height: 71px;">
                <div class="kpi-metric-name">Recommended Retention List</div>
                <div class="kpi-metric-value" style="color: #F59E0B; font-size: 1.45rem; font-weight: 800;">{flagged_churners:,}</div>
                <div style="font-size: 0.72rem; color: #8890A6; margin-top: 0.2rem;">{retention_pct:.1%} targeted for outreach</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
            <div class="kpi-card-horizontal" style="box-shadow: none !important; border: 1px solid rgba(255,255,255,0.08) !important; border-left: 3px solid #10B981 !important; display: flex; flex-direction: column; align-items: flex-start; justify-content: center; height: 71px;">
                <div class="kpi-metric-name">Projected Revenue Saved</div>
                <div class="kpi-metric-value" style="color: #10B981; font-size: 1.45rem; font-weight: 800;">${roi["revenue_saved"]:,.0f}</div>
                <div style="font-size: 0.72rem; color: #8890A6; margin-top: 0.2rem;">Based on 25% campaign conversion</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Executive Summary and AI Insights Section
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<h4 class='panel-title' style='color: #22D3EE; font-size: 1.05rem;'>Portfolio snapshot</h4>", unsafe_allow_html=True)
    
    ins1, ins2 = st.columns([0.6, 0.4], gap="small")
    with ins1:
        st.markdown(
            f"""
            <div style="font-size: 0.82rem; line-height: 1.4; color: #EDEFF5;">
                <p style="margin: 0;">
                    The current portfolio shows <strong style="color: #EF4444;">{tier_counts['High']:,}</strong> high-risk accounts out of <strong style="color: #22D3EE;">{total_customers:,}</strong> subscribers, with a targeted outreach list of <strong style="color: #F59E0B;">{flagged_churners:,}</strong> accounts.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with ins2:
        st.markdown(
            f"""
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; font-size: 0.76rem; line-height: 1.35; margin-top: 0.2rem;">
                <div>
                    <strong style="color: #8890A6;">Top Risk Segment:</strong><br>
                    <span style="color: #EF4444; font-weight: 700;">Month-to-Month Fiber Optic</span>
                </div>
                <div>
                    <strong style="color: #8890A6;">Expected Revenue Saved:</strong><br>
                    <span style="color: #10B981; font-weight: 700; font-family: 'JetBrains Mono', monospace;">${roi['revenue_saved']:,.0f}</span>
                </div>
                <div>
                    <strong style="color: #8890A6;">Recommended Campaign:</strong><br>
                    <span style="color: #22D3EE; font-weight: 700;">Contract Migration &amp; Auto-Pay</span>
                </div>
                <div>
                    <strong style="color: #8890A6;">Key Drivers of Churn:</strong><br>
                    <span style="color: #EDEFF5;">Month-to-month, Fiber Optic, E-Check</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Level 2: Pie & Histogram
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    chart1, chart2 = st.columns([1, 1], gap="small")
    with chart1:
        st.markdown("<div class='panel' style='min-height: 176px !important;'>", unsafe_allow_html=True)
        st.markdown("<h5 class='panel-title'>Customer Risk Segmentation</h5>", unsafe_allow_html=True)
        fig = px.pie(
            names=tier_counts.index,
            values=tier_counts.values,
            hole=0.55,
            color=tier_counts.index,
            color_discrete_map={"Low": "#34D399", "Medium": "#FBBF24", "High": "#F87171"},
        )
        apply_plotly_theme(fig)
        fig.update_layout(height=200, legend_title_text="Risk")
        st.plotly_chart(fig, width="stretch", config={'responsive': True})
        st.markdown("</div>", unsafe_allow_html=True)
    with chart2:
        st.markdown("<div class='panel' style='min-height: 176px !important;'>", unsafe_allow_html=True)
        st.markdown("<h5 class='panel-title'>Predicted Churn Score Distribution</h5>", unsafe_allow_html=True)
        hist = px.histogram(
            scored,
            x="churn_probability",
            nbins=30,
            color="risk_tier",
            color_discrete_map={"Low": "#34D399", "Medium": "#FBBF24", "High": "#F87171"},
        )
        hist.add_vline(x=decision_threshold, line_dash="dash", line_color="#EDEFF5")
        apply_plotly_theme(hist)
        hist.update_layout(height=200, xaxis_tickformat=".0%", bargap=0.03, legend_title_text="Risk")
        st.plotly_chart(hist, width="stretch", config={'responsive': True})
        st.markdown("</div>", unsafe_allow_html=True)

    # Level 3: Segment Concentration Charts
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    chart3, chart4 = st.columns([1, 1], gap="small")
    with chart3:
        st.markdown("<div class='panel' style='min-height: 176px !important;'>", unsafe_allow_html=True)
        st.markdown("<h5 class='panel-title'>Customer Risk Segmentation by Contract Type</h5>", unsafe_allow_html=True)
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
            color_discrete_map={"Low": "#34D399", "Medium": "#FBBF24", "High": "#F87171"},
        )
        apply_plotly_theme(fig_segment)
        fig_segment.update_layout(height=200, legend_title_text="Risk")
        st.plotly_chart(fig_segment, width="stretch", config={'responsive': True})
        st.markdown("</div>", unsafe_allow_html=True)
    with chart4:
        st.markdown("<div class='panel' style='min-height: 176px !important;'>", unsafe_allow_html=True)
        st.markdown("<h5 class='panel-title'>Customer Risk Segmentation by Internet Service</h5>", unsafe_allow_html=True)
        segment_internet = (
            scored.groupby(["InternetService", "risk_tier"], observed=False)
            .size()
            .reset_index(name="customers")
        )
        fig_segment_internet = px.bar(
            segment_internet,
            x="InternetService",
            y="customers",
            color="risk_tier",
            barmode="group",
            color_discrete_map={"Low": "#34D399", "Medium": "#FBBF24", "High": "#F87171"},
        )
        apply_plotly_theme(fig_segment_internet)
        fig_segment_internet.update_layout(height=200, legend_title_text="Risk")
        st.plotly_chart(fig_segment_internet, width="stretch", config={'responsive': True})
        st.markdown("</div>", unsafe_allow_html=True)

    # Level 4: Business Summary Model Details
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<h5 class='panel-title'>ROI model</h5>", unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns(4, gap="small")
    with r1:
        st.markdown(
            f"""
            <div class="kpi-card-horizontal" style="box-shadow: none !important; border: 1px solid rgba(255,255,255,0.08) !important; border-left: 3px solid #EF4444 !important; height: 56px !important;">
                <div>
                    <div class="kpi-metric-name">Est. True Churners</div>
                    <div class="kpi-metric-value" style="color: #EF4444; font-size: 1.1rem;">${roi["estimated_true_churners"]:,.0f}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with r2:
        st.markdown(
            f"""
            <div class="kpi-card-horizontal" style="box-shadow: none !important; border: 1px solid rgba(255,255,255,0.08) !important; border-left: 3px solid #10B981 !important; height: 56px !important;">
                <div>
                    <div class="kpi-metric-name">Customers Retained</div>
                    <div class="kpi-metric-value" style="color: #10B981; font-size: 1.1rem;">{roi["customers_retained"]:,.0f}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with r3:
        st.markdown(
            f"""
            <div class="kpi-card-horizontal" style="box-shadow: none !important; border: 1px solid rgba(255,255,255,0.08) !important; border-left: 3px solid #10B981 !important; height: 56px !important;">
                <div>
                    <div class="kpi-metric-name">Revenue Saved</div>
                    <div class="kpi-metric-value" style="color: #10B981; font-size: 1.1rem;">${roi["revenue_saved"]:,.0f}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with r4:
        st.markdown(
            f"""
            <div class="kpi-card-horizontal" style="box-shadow: none !important; border: 1px solid rgba(255,255,255,0.08) !important; border-left: 3px solid #F59E0B !important; height: 56px !important;">
                <div>
                    <div class="kpi-metric-name">Campaign Cost</div>
                    <div class="kpi-metric-value" style="color: #FBBF24; font-size: 1.1rem;">${roi["campaign_cost"]:,.0f}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption(
        f"Assumptions: annual ARPU value ${ANNUAL_VALUE:.0f}, CAC avoided ${CAC:.0f}, "
        f"offer cost ${OFFER_COST:.0f}, campaign success {CAMPAIGN_SUCCESS_RATE:.0%}, "
        f"precision estimate {precision_estimate:.1%}."
    )
    st.markdown("</div>", unsafe_allow_html=True)  # close .panel
    st.markdown("</div>", unsafe_allow_html=True)  # close .bi-report-container


def _apply_profile_to_widgets(profile: dict):
    """
    Copy a customer profile into Streamlit session_state
    so all sidebar widgets display the selected profile.
    """

    if profile is None:
        return

    st.session_state.customer_defaults = profile.copy()

    for key, value in profile.items():
        st.session_state[key] = value

# Initialize defaults in session state if not already set
if "customer_defaults" not in st.session_state:
    st.session_state.customer_defaults = DEFAULT_CUSTOMER.copy()
    _apply_profile_to_widgets(DEFAULT_CUSTOMER)
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "prediction_customer" not in st.session_state:
    st.session_state.prediction_customer = None

# Custom Navigation State Initialization (Predict / Executive selection)
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Predict"

# Load model parameters
try:
    model, preprocessor, feature_names = load_artifacts()
except Exception as exc:  # noqa: BLE001
    logger.critical("Fatal startup error: could not load model artifacts: %s", exc)
    st.error(f"Model artifacts missing: {exc}")
    st.stop()

decision_threshold = get_decision_threshold()

submitted = render_sidebar_drawer(
    {},
    decision_threshold,
    lambda: None,
    lambda: None,
    lambda: None,
)

# Form Submission Processing
if submitted:
    try:
        customer = build_customer_from_form()
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        for index, message in enumerate(
            [
                "Preparing the profile...",
                "Scanning churn signals...",
                "Checking risk patterns...",
                "Drafting the recommendation...",
            ],
            start=1,
        ):
            status_placeholder.markdown(
                f"<div class='ai-processing-card'>🔍 {message}</div>",
                unsafe_allow_html=True,
            )
            progress_bar.progress(index / 4)
            time.sleep(0.12)
        st.session_state.prediction_result = predict_churn(
            customer, model=model, preprocessor=preprocessor, feature_names=feature_names
        )
        st.session_state.prediction_customer = customer
        status_placeholder.markdown(
            "<div class='ai-processing-card'>✅ Done. The summary and next steps are ready.</div>",
            unsafe_allow_html=True,
        )
        progress_bar.progress(1.0)
        time.sleep(0.25)
    except ValueError as exc:
        # Expected input-validation failure (e.g. malformed form data) --
        # logged at INFO since it's a user-input issue, not a system fault.
        logger.info("Prediction rejected due to invalid input: %s", exc)
        st.warning(str(exc))
    except Exception as exc:  # noqa: BLE001
        # Unexpected failure in the inference pipeline -- logged with full
        # traceback since this indicates a real bug or environment problem.
        logger.exception("Unexpected error while scoring a customer: %s", exc)
        st.error(f"Inference failed: {exc}")

# ==============================================================================
# MAIN PAGE RENDER
# ==============================================================================

# Custom Horizontal Header Navbar (Logo, Page Selection Routing, Profile Indicators)
st.markdown("<div class='nav-header-row'>", unsafe_allow_html=True)
nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([0.44, 0.18, 0.18, 0.18, 0.02], gap="small")
with nav_col1:
    st.markdown(
        """
        <div class="nav-brand">
            <span style="display:flex; align-items:center; justify-content:center; width:38px; height:38px; border-radius:10px; background: rgba(37, 99, 235, 0.16); border: 1px solid rgba(37, 99, 235, 0.24); color: #38BDF8; font-weight: 800; font-size: 0.95rem; letter-spacing: -0.02em; line-height: 1;">CT</span>
            <div>
                <h2 style="color: #EDEFF5; font-size: 1.02rem; font-weight: 800; margin: 0; line-height: 1.15; letter-spacing: -0.01em;">ConnectTel Churn Studio</h2>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with nav_col2:
    predict_clicked = st.button("Predictive Analysis", width="stretch", type="primary" if st.session_state.active_tab == "Predict" else "secondary")
with nav_col3:
    explain_clicked = st.button("Explain AI", width="stretch", type="primary" if st.session_state.active_tab == "Explain" else "secondary")
with nav_col4:
    exec_clicked = st.button("Executive Dashboard", width="stretch", type="primary" if st.session_state.active_tab == "Executive Dashboard" else "secondary")
with nav_col5:
    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: flex-end; height: 100%;">
            <div class="nav-avatar" title="User Profile">JS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)

# Handle navigation tab routing click triggers
if predict_clicked:
    st.session_state.active_tab = "Predict"
    st.rerun()
if explain_clicked:
    st.session_state.active_tab = "Explain"
    st.rerun()
if exec_clicked:
    st.session_state.active_tab = "Executive Dashboard"
    st.rerun()

# Linear Gradient Accent Bar below top navbar
st.markdown(
    """
    <div style="height: 3px; background: linear-gradient(90deg, #22D3EE 0%, #10B981 100%); border-radius: 2px; margin-bottom: 0.55rem;"></div>
    """,
    unsafe_allow_html=True
)

# Screen routing based on custom navbar state
if st.session_state.active_tab == "Predict":
    # Render horizontal 100px KPI Cards (displays '-' placeholder fallbacks if no prediction has been run yet)
    render_kpi_cards(st.session_state.prediction_result, st.session_state.prediction_customer)
    
    if st.session_state.prediction_result and st.session_state.prediction_customer:
        render_prediction_result(st.session_state.prediction_result, st.session_state.prediction_customer)
    else:
        # Professional empty state block when no prediction has been triggered yet
        st.markdown(
            """
            <div class="panel" style="text-align: center; padding: 1.8rem 1.2rem !important; margin-top: 0.7rem;">
                <h3 style="color: #EDEFF5; font-weight: 800; font-size: 1.3rem; margin: 0 0 0.4rem 0;">Ready to Predict</h3>
                <p style="color: #8890A6; font-size: 0.82rem; max-width: 500px; margin: 0 auto 1.25rem auto; line-height: 1.45;">
                    Open the left <strong>Filter Drawer</strong> to adjust parameters, select example templates, and click <strong>Run Risk Prediction</strong> to generate:
                </p>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.7rem; max-width: 600px; margin: 0 auto 0.8rem auto; text-align: left;">
                    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-left: 3px solid #22D3EE; padding: 0.65rem 0.75rem; border-radius: 6px;">
                        <strong style="color: #22D3EE; font-size: 0.7rem; letter-spacing: 0.04em; text-transform: uppercase;">Core Metrics</strong><br>
                        <span style="font-size: 0.72rem; color: #8890A6;">Churn Probability, Risk Tier, and Model Confidence</span>
                    </div>
                    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-left: 3px solid #A78BFA; padding: 0.65rem 0.75rem; border-radius: 6px;">
                        <strong style="color: #38BDF8; font-size: 0.7rem; letter-spacing: 0.04em; text-transform: uppercase;">Explainability</strong><br>
                        <span style="font-size: 0.72rem; color: #8890A6;">Top risk drivers and local explainability diagnostics</span>
                    </div>
                    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-left: 3px solid #10B981; padding: 0.65rem 0.75rem; border-radius: 6px;">
                        <strong style="color: #10B981; font-size: 0.7rem; letter-spacing: 0.04em; text-transform: uppercase;">Action &amp; ROI</strong><br>
                        <span style="font-size: 0.72rem; color: #8890A6;">Recommended CRM actions and retention economics ROI</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

elif st.session_state.active_tab == "Explain":
    render_explain_ai(model, preprocessor, feature_names)
else:
    with st.spinner("Scoring full customer base for the executive portfolio view..."):
        try:
            render_executive_dashboard(model, preprocessor, feature_names, decision_threshold)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Executive dashboard batch-scoring failed: %s", exc)
            st.error(f"Could not render executive portfolio: {exc}")
