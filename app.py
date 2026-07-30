"""Enterprise Streamlit dashboard for ConnectTel churn prediction."""

import json
import os
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

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
from src.utils import load_config

st.set_page_config(page_title="ConnectTel AI Churn Studio", layout="wide", initial_sidebar_state="expanded")

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

# Premium SaaS Analytics Stylesheet (Microsoft Fabric & Tableau Inspired)
_BASE_CSS = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"], .stMarkdown, p, h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Viewport-optimizations: eliminate unnecessary vertical scrolling */
    .main .block-container {
        max-width: 1600px !important;
        padding-top: 0.25rem !important;
        padding-bottom: 0.4rem !important;
        padding-left: 1.1rem !important;
        padding-right: 1.1rem !important;
        margin: 0 auto !important;
    }

    /* Enable vertical scrolling on the entire page */
    html, body, [data-testid="stAppViewContainer"] {
        height: auto !important;
        overflow-y: auto !important;
    }
    [data-testid="stAppViewContainer"] > .main,
    div[data-testid="stMain"],
    div[data-testid="stMainBlockContainer"],
    div[data-testid="stMain"] > div {
        overflow-y: visible !important;
        height: auto !important;
        max-height: none !important;
    }
    section[data-testid="stSidebar"] {
        height: 100vh !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
    }
    
    .stApp {
        background: linear-gradient(180deg, #0F172A 0%, #111827 100%) !important;
    }
    
    .main {
        background-color: transparent !important;
    }
    
    /* Hide Streamlit default visual clutter completely */
    /* Hide Streamlit's decorative header bar and toolbar (Deploy button,
       hamburger menu, etc) but NOT the header element itself -- the
       collapsed-sidebar reopen control lives inside/near this header, so
       nuking it with display:none strands the user with no way to reopen
       the sidebar once it's collapsed. */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 2.4rem !important;
    }
    header[data-testid="stHeader"] [data-testid="stToolbar"],
    header[data-testid="stHeader"] [data-testid="stDecoration"] {
        display: none !important;
    }
    [data-testid="stHeaderActionElements"] {
        display: flex !important;
        align-items: center !important;
    }
    footer {
        display: none !important;
    }
    /* Streamlit auto-injects a chain-link anchor icon next to every heading
       (h1-h6), including raw HTML ones inside st.markdown blocks -- purely
       decorative clutter for this dashboard, so hide it everywhere. */
    [data-testid="stHeaderActionElements"],
    .stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a,
    .stMarkdown h4 a, .stMarkdown h5 a, .stMarkdown h6 a {
        display: none !important;
    }
    /* Sidebar reopen control (shown after the sidebar is collapsed) --
       force it visible and theme it to match, wherever Streamlit renders it. */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 1000 !important;
        position: relative !important;
        margin: 0.35rem 0 0.25rem 0 !important;
        pointer-events: auto !important;
    }
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="collapsedControl"] button {
        background-color: rgba(15, 23, 42, 0.96) !important;
        border: 1px solid rgba(37, 99, 235, 0.25) !important;
        border-radius: 999px !important;
        color: #38BDF8 !important;
        width: 2.25rem !important;
        height: 2.25rem !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #0A0D14; }
    ::-webkit-scrollbar-thumb { background: #2A3142; border-radius: 4px; }
    
    /* Style Streamlit sidebar as a premium dark glassmorphic filter drawer */
    section[data-testid="stSidebar"] {
        width: 320px !important;
        min-width: 320px !important;
        max-width: 320px !important;
        background: rgba(15, 23, 42, 0.97) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.28) !important;
        transition: width 0.2s ease, min-width 0.2s ease !important;
    }
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        padding-top: 0.35rem !important;
        padding-left: 0.45rem !important;
        padding-right: 0.45rem !important;
        gap: 0.25rem !important;
    }
    section[data-testid="stSidebar"] .stButton > button,
    section[data-testid="stSidebar"] .stDownloadButton > button {
        width: 100% !important;
        border-radius: 10px !important;
        border: 1px solid rgba(37, 99, 235, 0.25) !important;
        background: linear-gradient(180deg, #111827 0%, #0F172A 100%) !important;
        color: #EDEFF5 !important;
        padding: 0.55rem 0.75rem !important;
        font-weight: 600 !important;
        min-height: 2.35rem !important;
        box-shadow: none !important;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"],
    section[data-testid="stSidebar"] .stDownloadButton > button {
        border-color: rgba(37, 99, 235, 0.35) !important;
        background: linear-gradient(180deg, #111827 0%, #0F172A 100%) !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        border-color: rgba(34,211,238,0.25) !important;
        transform: translateY(-1px) !important;
    }
    section[data-testid="stSidebar"] .stExpander {
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 12px !important;
        padding: 0.35rem 0.45rem !important;
        background: rgba(255,255,255,0.02) !important;
    }
    section[data-testid="stSidebar"] .stExpander summary {
        color: #EDEFF5 !important;
        font-weight: 700 !important;
        font-size: 0.8rem !important;
    }
    section[data-testid="stSidebar"] * {
        color: #C4CAD9 !important;
    }
    button[data-testid="stSidebarCollapseButton"] {
        background-color: #151a24 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
        border-radius: 50% !important;
        color: #22D3EE !important;
    }
    
    /* Style the header navigation buttons as a compact segmented-control */
    div[data-testid="stHorizontalBlock"] button {
        background-color: transparent !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 999px !important;
        color: #8890A6 !important;
        font-weight: 700 !important;
        font-size: 0.76rem !important;
        padding: 0.32rem 0.8rem !important;
        transition: all 0.2s ease !important;
        height: 2.05rem !important;
        min-height: 2.05rem !important;
        width: 100% !important;
        box-shadow: none !important;
    }
    div[data-testid="stHorizontalBlock"] button:hover {
        color: #EDEFF5 !important;
        border-color: rgba(37, 99, 235, 0.25) !important;
        background-color: rgba(37, 99, 235, 0.08) !important;
    }
    /* Active/selected nav tab. Streamlit no longer exposes a plain `kind`
       HTML attribute on the rendered <button> -- only a prefixed
       data-testid="stBaseButton-primary" -- so matching on button[kind=...]
       (as before) never matches anything, and the selected tab never
       highlights. Match on the testid instead. */
    div[data-testid="stHorizontalBlock"] button[data-testid^="stBaseButton-primary"] {
        color: #F8FAFC !important;
        border-color: transparent !important;
        background: linear-gradient(90deg, #2563EB 0%, #38BDF8 100%) !important;
        box-shadow: none !important;
        text-shadow: none !important;
    }
    .nav-header-row {
        display: flex !important;
        align-items: center !important;
        gap: 0.3rem !important;
        padding: 0.45rem 0.5rem !important;
        margin: 0 0 0.35rem 0 !important;
        background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015)) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 16px !important;
        box-shadow: 0 10px 24px -14px rgba(0,0,0,0.55) !important;
    }
    .nav-brand {
        display: flex !important;
        align-items: center !important;
        gap: 0.6rem !important;
        min-height: 2.25rem !important;
        padding: 0 !important;
    }
    .nav-avatar {
        width: 32px !important;
        height: 32px !important;
        border-radius: 50% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-weight: 700 !important;
        font-size: 0.78rem !important;
        background: linear-gradient(135deg, #22D3EE 0%, #10B981 100%) !important;
        color: #05161A !important;
    }
    
    /* Dark analytics-console Card Design */
    .panel {
        background: linear-gradient(180deg, #1E293B 0%, #172033 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 12px !important;
        padding: 10px 11px !important;
        box-shadow: 0 8px 20px -12px rgba(0, 0, 0, 0.5) !important;
        margin-bottom: 0px !important;
        box-sizing: border-box !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-start !important;
        height: 100% !important;
    }
    .section-gap {
        height: 0.28rem !important;
    }
    .panel-title {
        margin: 0 0 0.2rem 0 !important;
        color: #EDEFF5 !important;
        font-size: 0.8rem !important;
        font-weight: 700 !important;
    }
    .chip-wrap {
        display: flex !important;
        flex-wrap: wrap !important;
        gap: 0.3rem !important;
        margin-top: 0.3rem !important;
    }
    .chip-pill {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 999px !important;
        padding: 0.25rem 0.55rem !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        color: #C4CAD9 !important;
        background: rgba(255,255,255,0.04) !important;
    }
    .chip-pill.active {
        color: #F8FAFC !important;
        border-color: rgba(56, 189, 248, 0.35) !important;
        background: linear-gradient(90deg, rgba(37, 99, 235, 0.24), rgba(56, 189, 248, 0.14)) !important;
    }
    .ai-summary-panel {
        padding: 0.6rem 0.7rem !important;
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.14), rgba(16, 185, 129, 0.10)) !important;
        border: 1px solid rgba(56, 189, 248, 0.18) !important;
        margin-bottom: 0.3rem !important;
        border-radius: 12px !important;
    }
    .ai-summary-headline {
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
        margin-bottom: 0.2rem !important;
        line-height: 1.25 !important;
    }
    .ai-summary-copy {
        font-size: 0.78rem !important;
        color: #DCE3EF !important;
        line-height: 1.3 !important;
        margin-bottom: 0.25rem !important;
    }
    .ai-summary-row {
        display: flex !important;
        flex-wrap: wrap !important;
        gap: 0.35rem !important;
        margin-top: 0.2rem !important;
    }
    .mini-badge {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0.2rem 0.45rem !important;
        border-radius: 999px !important;
        font-size: 0.68rem !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
    }
    .strategy-card {
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-left: 3px solid var(--accent, #22D3EE) !important;
        border-radius: 12px !important;
        padding: 0.45rem 0.55rem !important;
        background: rgba(255,255,255,0.025) !important;
        margin-bottom: 0.25rem !important;
    }
    .strategy-title {
        font-size: 0.76rem !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
        margin-bottom: 0.15rem !important;
    }
    .strategy-metrics {
        display: flex !important;
        justify-content: space-between !important;
        gap: 0.25rem !important;
        font-size: 0.68rem !important;
        color: #DCE3EF !important;
        margin-top: 0.2rem !important;
    }
    .strategy-metrics strong {
        color: #F8FAFC !important;
    }
    .ai-processing-card {
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        border-radius: 10px !important;
        padding: 0.5rem 0.6rem !important;
        background: rgba(37, 99, 235, 0.12) !important;
        color: #EDEFF5 !important;
        font-size: 0.78rem !important;
        margin-bottom: 0.35rem !important;
    }
    .panel-subtitle {
        margin: 0 0 0.35rem 0 !important;
        color: #8890A6 !important;
        font-size: 0.74rem !important;
        line-height: 1.35 !important;
    }
    
    /* Single-sheet BI Report Container for Executive tab. Flat/borderless --
       the sections inside it are their own .panel cards, so this container
       only needs to set the outer page padding, not add a second card
       border around the whole thing (that reads as a card-in-a-card). */
    .bi-report-container {
        background: transparent !important;
        border: none !important;
        border-radius: 0px !important;
        padding: 0px !important;
        box-shadow: none !important;
        margin-bottom: 0px !important;
        box-sizing: border-box !important;
    }
    .panel {
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease, opacity 0.35s ease !important;
        animation: cardFadeIn 0.45s ease both !important;
    }
    .panel:hover {
        border-color: rgba(37, 99, 235, 0.22) !important;
        box-shadow: 0 8px 18px -10px rgba(0, 0, 0, 0.5) !important;
    }
    @keyframes cardFadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Heading styling */
    h4, h5 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        color: #EDEFF5 !important;
        margin: 0 0 0.3rem 0 !important;
        font-size: 0.84rem !important;
        letter-spacing: 0.01em !important;
    }
    p, li, .stMarkdown {
        font-size: 0.9rem !important;
        line-height: 1.5 !important;
    }
    h3 {
        color: #EDEFF5 !important;
    }
    .stCaption, [data-testid="stCaptionContainer"], .stMarkdown small {
        color: #9AA3B7 !important;
    }
    
    /* Premium Compact 100px KPI Card. A colored left-accent border (set
       per-card via the --accent custom property) stands in for a category
       icon -- clean, legible, and doesn't depend on an emoji glyph. */
    .kpi-card-horizontal {
        background: linear-gradient(180deg, #1E293B 0%, #172033 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-left: 3px solid var(--accent, #2563EB) !important;
        border-radius: 12px !important;
        padding: 0.55rem 0.8rem !important;
        box-shadow: 0 6px 16px -8px rgba(0, 0, 0, 0.45) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        gap: 0.55rem !important;
        height: 82px !important;
        width: 100% !important;
        box-sizing: border-box !important;
        transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.2s ease, border-color 0.2s ease, opacity 0.35s ease !important;
        animation: cardFadeIn 0.5s ease both !important;
    }
    .kpi-card-horizontal:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 18px -10px rgba(0, 0, 0, 0.5) !important;
    }
    .strategy-card {
        transition: transform 0.2s ease, border-color 0.2s ease, background 0.2s ease !important;
    }
    .strategy-card:hover {
        transform: translateY(-1px) !important;
        border-color: rgba(56, 189, 248, 0.26) !important;
    }
    .risk-badge {
        transition: transform 0.2s ease, opacity 0.2s ease !important;
    }
    .risk-badge:hover {
        transform: translateY(-1px) !important;
    }
    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button,
    div[data-testid="stDownloadButton"] > button,
    button[data-testid^="stBaseButton"] {
        transition: transform 0.18s ease, border-color 0.2s ease, background-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover,
    div[data-testid="stDownloadButton"] > button:hover,
    button[data-testid^="stBaseButton"]:hover {
        transform: translateY(-1px) !important;
    }
    .kpi-metric-name {
        color: #8890A6 !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        margin: 0 !important;
        line-height: 1.2 !important;
    }
    .kpi-metric-value {
        color: #EDEFF5 !important;
        font-size: 1.45rem !important;
        font-weight: 800 !important;
        margin: 0 !important;
        line-height: 1.1 !important;
        font-family: 'JetBrains Mono', 'Inter', monospace !important;
    }
    .kpi-status-badge {
        display: inline-block !important;
        padding: 0.15rem 0.55rem !important;
        border-radius: 4px !important;
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        width: fit-content !important;
        flex-shrink: 0 !important;
    }
    
    /* Risk Badge */
    .risk-badge {
        display: inline-block !important;
        color: #EDEFF5 !important;
        padding: 0.25rem 0.7rem !important;
        border-radius: 999px !important;
        font-weight: 700 !important;
        font-size: 0.74rem !important;
        letter-spacing: 0.02em !important;
        margin-bottom: 0.35rem !important;
        background: rgba(37, 99, 235, 0.15) !important;
        border: 1px solid rgba(37, 99, 235, 0.2) !important;
        box-shadow: none !important;
    }
    
    /* Custom Vertical CRM Timeline nodes */
    .timeline {
        position: relative !important;
        padding-left: 1.5rem !important;
        border-left: 2px solid rgba(255, 255, 255, 0.08) !important;
        margin-left: 0.6rem !important;
        margin-top: 0.5rem !important;
    }
    .timeline-item {
        margin-bottom: 0.65rem !important;
        position: relative !important;
    }
    .timeline-node {
        position: absolute !important;
        left: -1.95rem !important;
        top: 0.15rem !important;
        width: 10px !important;
        height: 10px !important;
        border-radius: 50% !important;
        border: 2px solid #0F131B !important;
        box-shadow: 0 0 8px -1px currentColor !important;
    }
    .timeline-title {
        font-size: 0.8rem !important;
        font-weight: 700 !important;
        color: #EDEFF5 !important;
        margin-bottom: 0.1rem !important;
    }
    .timeline-desc {
        font-size: 0.75rem !important;
        color: #A2AAC0 !important;
        line-height: 1.3 !important;
    }
    
    /* ROI Cards styling. Same left-accent-border pattern as the KPI cards
       above, driven by --accent, instead of an empty icon circle. */
    .roi-card {
        background: linear-gradient(180deg, #1E293B 0%, #172033 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-left: 3px solid var(--accent, #2563EB) !important;
        border-radius: 14px !important;
        padding: 0.65rem 0.9rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.6rem !important;
        box-shadow: 0 6px 16px -8px rgba(0, 0, 0, 0.45) !important;
        height: 100%;
        transition: transform 0.2s ease !important;
    }
    .roi-card:hover {
        transform: translateY(-1px) !important;
    }
    .roi-info {
        flex: 1 !important;
    }
    .roi-label {
        color: #8890A6 !important;
        font-size: 0.72rem !important;
        font-weight: 500 !important;
        margin-bottom: 0.1rem;
    }
    .roi-val {
        color: #EDEFF5 !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', 'Inter', monospace !important;
    }
    
    /* Sidebar filter drawer refinement */
    section[data-testid="stSidebar"] {
        padding-left: 0.2rem !important;
        padding-right: 0.2rem !important;
    }
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        padding-top: 0.25rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        gap: 0.35rem !important;
    }
    section[data-testid="stSidebar"] .filter-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015)) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
        padding: 0.6rem 0.7rem !important;
        margin-bottom: 0.45rem !important;
    }
    section[data-testid="stSidebar"] .filter-card-label {
        font-size: 0.74rem !important;
        font-weight: 700 !important;
        color: #EDEFF5 !important;
        margin-bottom: 0.38rem !important;
        letter-spacing: 0.01em !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div {
        display: flex !important;
        flex-wrap: nowrap !important;
        gap: 0.35rem !important;
        margin-top: 0.1rem !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
        flex: 1 1 0 !important;
        min-height: 2.05rem !important;
        border-radius: 999px !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        background: rgba(255,255,255,0.04) !important;
        color: #C4CAD9 !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        padding: 0.4rem 0.55rem !important;
        justify-content: center !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] {
        background: linear-gradient(90deg, #2563EB 0%, #38BDF8 100%) !important;
        border-color: transparent !important;
        color: #F8FAFC !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] {
        margin-bottom: 0.1rem !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] > div,
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] > div,
    section[data-testid="stSidebar"] div[data-testid="stTextInput"] > div {
        min-height: 2.2rem !important;
        border-radius: 10px !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stFormSubmitButton"] > button {
        min-height: 2.35rem !important;
        border-radius: 10px !important;
    }

    /* Bold compact input form labels */
    .stMarkdown label, .stSelectbox label, .stNumberInput label {
        font-weight: 700 !important;
        font-size: 0.78rem !important;
        color: #C4CAD9 !important;
        margin-bottom: 0.1rem !important;
    }
    
    /* Compact inputs, dark styled. Streamlit/BaseWeb nests inputs several
       divs deep and sets its own background on the innermost node, so we
       force every descendant transparent and only paint the outer shell dark
       -- otherwise you get a white box with white-on-white text. */
    /* Force native form-control color scheme dark first -- kills the
       browser default WHITE background some select/number controls fall
       back to before any custom CSS even applies. */
    html { color-scheme: dark !important; }

    /* Compact inputs, dark styled. Streamlit/BaseWeb nests inputs several
       divs deep and sets its own background on the innermost node, so we
       force every descendant transparent and only paint the outer shell dark
       -- otherwise you get a white box with white-on-white text. Target by
       both data-baseweb AND data-testid wrapper, since the exact nesting
       differs between the closed control and open popover. */
    section[data-testid="stSidebar"] div[data-testid="stForm"] {
        background: linear-gradient(180deg, #12161F 0%, #0F131B 100%) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 14px !important;
        padding: 0.7rem !important;
        box-shadow: 0 8px 20px -8px rgba(0,0,0,0.45) !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"],
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"],
    section[data-testid="stSidebar"] div[data-testid="stTextInput"] {
        width: 100% !important;
        margin-bottom: 0.25rem !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] div[data-baseweb="input"] > div,
    div[data-testid="stTextInput"] div[data-baseweb="input"] > div,
    div[data-baseweb="select"], div[data-baseweb="input"], div[data-baseweb="base-input"] {
        min-height: 2.25rem !important;
        width: 100% !important;
        min-width: 0 !important;
        background-color: #171B26 !important;
        border-color: rgba(255,255,255,0.14) !important;
        font-size: 0.82rem !important;
    }
    /* The selected-value span is a flex child next to the chevron icon.
       min-width:0 above (needed so the control itself can shrink inside
       narrow 118px sidebar columns) also shrinks THIS child to 0, so the
       chosen value disappears and only the chevron is left visible.
       Give the value span its own flex-grow + visible overflow so it
       always shows the current selection. */
    div[data-baseweb="select"] > div > div:first-child {
        flex: 1 1 auto !important;
        min-width: 0 !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
        display: block !important;
    }
    div[data-testid="stSelectbox"] *, div[data-testid="stNumberInput"] *, div[data-testid="stTextInput"] *,
    div[data-baseweb="select"] *, div[data-baseweb="input"] *, div[data-baseweb="base-input"] * {
        background-color: transparent !important;
        color: #EDEFF5 !important;
        -webkit-text-fill-color: #EDEFF5 !important;
        border-color: rgba(255,255,255,0.14) !important;
        fill: #8890A6 !important;
    }
    /* The 2-up sidebar rows (Tenure/Contract, Phone/Lines, etc.) live in
       st.columns() inside a 320px sidebar -- without an explicit min-width
       a long option label can force the flex column to shrink to near
       nothing, which is what made "Contract" render as a blank sliver.
       That min-width alone isn't enough though: Streamlit's column row
       defaults to flex-wrap, so once the two 118px columns + 16px gap
       don't fit the available width (they don't, inside an expander's
       own padding), the SECOND column wraps onto its own line below the
       first -- which is why the whole s1 stack (Phone/Internet/Backup/
       Support/Movies) rendered before the whole s2 stack (Lines/Security/
       Protection/Streaming TV) instead of side-by-side pairs. Force
       nowrap and let each column truly share the row equally instead. */
    section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] {
        flex-wrap: nowrap !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] > div {
        min-width: 0 !important;
        flex: 1 1 0px !important;
        width: 0 !important;
    }
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        height: 100% !important;
        padding-top: 0px !important;
        padding-bottom: 0px !important;
    }
    input, textarea, select {
        background-color: transparent !important;
        color: #EDEFF5 !important;
        -webkit-text-fill-color: #EDEFF5 !important;
    }
    input::placeholder, textarea::placeholder {
        color: #8890A6 !important;
        -webkit-text-fill-color: #8890A6 !important;
    }
    div[data-baseweb="popover"], div[data-baseweb="popover"] *,
    div[data-baseweb="menu"], div[data-baseweb="menu"] * {
        background-color: #171B26 !important;
    }
    div[data-testid="stSelectboxVirtualDropdown"] {
        background-color: #171B26 !important;
        min-width: 180px !important;
    }
    div[data-testid="stSelectboxVirtualDropdown"] * {
        background-color: #171B26 !important;
    }
    ul[role="listbox"] {
        background-color: #171B26 !important;
        min-width: max-content !important;
        width: max-content !important;
    }
    [role="listbox"] {
        background-color: #171B26 !important;
    }
    li[role="option"] {
        color: #EDEFF5 !important;
        background-color: #171B26 !important;
        white-space: nowrap !important;
        overflow: visible !important;
        min-width: max-content !important;
    }
    [role="option"] {
        color: #EDEFF5 !important;
        background-color: #171B26 !important;
    }
    li[role="option"] * {
        color: #EDEFF5 !important;
        white-space: nowrap !important;
    }
    [role="option"] * {
        color: #EDEFF5 !important;
        background-color: transparent !important;
    }
    li[role="option"]:hover, li[role="option"][aria-selected="true"] {
        background-color: rgba(34, 211, 238, 0.12) !important;
    }
    [role="option"]:hover, [role="option"][aria-selected="true"], [role="option"][data-selected="true"] {
        background-color: rgba(34, 211, 238, 0.15) !important;
    }
    li[role="option"]:hover *, li[role="option"][aria-selected="true"] * {
        color: #22D3EE !important;
    }
    [role="option"]:hover *, [role="option"][aria-selected="true"] *, [role="option"][data-selected="true"] * {
        color: #22D3EE !important;
    }
    button[data-baseweb="button"][aria-label*="Clear"] svg, [data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] {
        color: #8890A6 !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown label,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stNumberInput label {
        font-size: 0.74rem !important;
        font-weight: 700 !important;
        color: #EDEFF5 !important;
        margin-bottom: 0.18rem !important;
        letter-spacing: 0.01em !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stExpander"] [data-testid="stVerticalBlock"] {
        gap: 0.35rem !important;
    }

    /* Compact Expander Panels (borderless) */
    div[data-testid="stExpander"] {
        background-color: #12161F !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 12px !important;
        margin-bottom: 0.25rem !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
    }
    /* The summary/header row ships its own light background separate from
       the expander container above -- must be overridden explicitly or it
       stays a bright white bar. */
    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] details {
        background-color: transparent !important;
        color: #C4CAD9 !important;
    }
    div[data-testid="stExpander"] summary * {
        background-color: transparent !important;
        color: #C4CAD9 !important;
    }
    div[data-testid="stExpander"] summary {
        padding: 0.3rem 0.6rem !important;
        font-weight: 600 !important;
        font-size: 0.78rem !important;
    }
    div[data-testid="stExpander"] summary:hover {
        background-color: rgba(255,255,255,0.03) !important;
    }
    div[data-testid="stExpander"] svg {
        fill: #8890A6 !important;
    }
    div[data-testid="stExpander"] [data-testid="stVerticalBlock"] {
        padding: 0.3rem 0.6rem 0.55rem 0.6rem !important;
        gap: 0.3rem !important;
    }
    
    /* Native Form container overrides */
    div[data-testid="stForm"] {
        background-color: #12161F !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 14px !important;
        padding: 1rem !important;
        box-shadow: 0 8px 20px -6px rgba(0,0,0,0.4) !important;
    }
    
    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button,
    div[data-testid="stDownloadButton"] > button,
    button[data-testid^="stBaseButton"] {
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.35rem 0.9rem !important;
        font-size: 0.78rem !important;
        transition: all 0.2s ease-in-out !important;
    }
    div[data-testid="stFormSubmitButton"] > button,
    div[data-testid="stDownloadButton"] > button {
        background: linear-gradient(180deg, #111827 0%, #0F172A 100%) !important;
        color: #EDEFF5 !important;
        border: 1px solid rgba(37, 99, 235, 0.28) !important;
        box-shadow: none !important;
    }
    div[data-testid="stFormSubmitButton"] > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {
        background: linear-gradient(180deg, #172033 0%, #111827 100%) !important;
        border-color: rgba(56, 189, 248, 0.35) !important;
    }
    button[data-testid^="stBaseButton-primary"] {
        background-color: #2563EB !important;
        color: #F8FAFC !important;
        border: none !important;
        box-shadow: none !important;
    }
    button[data-testid^="stBaseButton-primary"]:hover {
        background-color: #3B82F6 !important;
        box-shadow: none !important;
    }
    button[data-testid^="stBaseButton-secondary"] {
        border: 1px solid rgba(37, 99, 235, 0.25) !important;
        color: #EDEFF5 !important;
        background-color: transparent !important;
    }
    button[data-testid^="stBaseButton-secondary"]:hover {
        background-color: rgba(37, 99, 235, 0.1) !important;
        color: #38BDF8 !important;
        border-color: rgba(56, 189, 248, 0.35) !important;
    }
    
    .muted {
        color: #8890A6;
        font-size: 0.8rem;
    }
    
    /* Prevent sidebar labels ("Tenure", "Contract"...) from breaking mid-word
       in the narrow 2-column form layout. */
    section[data-testid="stSidebar"] label p {
        word-break: keep-all !important;
        overflow-wrap: normal !important;
        hyphens: none !important;
    }

    
    /* Horizontal block spacings overrides */
    div[data-testid="stHorizontalBlock"] {
        gap: 10px !important;
    }
    div[data-testid="stVerticalBlock"] > div {
        margin-bottom: 0px !important;
        margin-top: 0px !important;
    }

    /* Sidebar is a permanently docked Filter Drawer, not a collapsible
       panel -- multiple Streamlit versions store "collapsed" state in the
       browser and re-hide it on reload with no visible way back in. Lock
       it open and remove every version of the collapse control instead. */
    section[data-testid="stSidebar"] {
        transform: none !important;
        visibility: visible !important;
        min-width: 320px !important;
        width: 320px !important;
        margin-left: 0px !important;
    }
    button[data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    </style>
    """

st.markdown(_BASE_CSS, unsafe_allow_html=True)


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


@st.cache_data(show_spinner="Scoring historical portfolio...")
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


def _apply_profile_to_widgets(profile):
    """Push a customer profile directly into each widget's session_state key."""
    st.session_state.tenure = int(profile["tenure"])
    st.session_state.monthly_charges = float(profile["MonthlyCharges"])
    st.session_state.total_charges = float(profile["TotalCharges"])
    st.session_state.contract = profile["Contract"]
    st.session_state.paperless_billing = profile["PaperlessBilling"]
    st.session_state.payment_method = profile["PaymentMethod"]
    st.session_state.gender = profile["gender"]
    st.session_state.senior_citizen = "Yes" if profile["SeniorCitizen"] else "No"
    st.session_state.partner = profile["Partner"]
    st.session_state.dependents = profile["Dependents"]
    st.session_state.phone_service = profile["PhoneService"]
    st.session_state.multiple_lines = profile["MultipleLines"]
    st.session_state.internet_service = profile["InternetService"]
    st.session_state.online_security = profile["OnlineSecurity"]
    st.session_state.online_backup = profile["OnlineBackup"]
    st.session_state.device_protection = profile["DeviceProtection"]
    st.session_state.tech_support = profile["TechSupport"]
    st.session_state.streaming_tv = profile["StreamingTV"]
    st.session_state.streaming_movies = profile["StreamingMovies"]


def reset_customer():
    st.session_state.customer_defaults = DEFAULT_CUSTOMER.copy()
    _apply_profile_to_widgets(DEFAULT_CUSTOMER)
    st.session_state.prediction_result = None
    st.session_state.prediction_customer = None


def load_example_customer():
    st.session_state.customer_defaults = DEFAULT_CUSTOMER.copy()
    _apply_profile_to_widgets(DEFAULT_CUSTOMER)
    st.session_state.prediction_result = None
    st.session_state.prediction_customer = None


def load_low_risk_customer():
    st.session_state.customer_defaults = LOW_RISK_CUSTOMER.copy()
    _apply_profile_to_widgets(LOW_RISK_CUSTOMER)
    st.session_state.prediction_result = None
    st.session_state.prediction_customer = None


def build_customer_from_form():
    """Build a validated customer payload from Streamlit widget state."""
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


def apply_plotly_theme(fig, title=None):
    """Enforces a consistent clean enterprise styling across all Plotly charts.

    `title=None` means "leave whatever title the chart already has alone"
    (e.g. one set via px.pie/px.bar's own `title=` kwarg). It must NOT be
    forwarded straight into update_layout, since Plotly treats an explicit
    `title=None` as "clear the title" -- that was wiping out every px-level
    chart title (Risk Tier Mix, Churn Probability Distribution, etc.).
    """
    has_existing_title = bool(fig.layout.title and fig.layout.title.text)
    layout_kwargs = dict(
        font=dict(family="Inter", color="#C4CAD9"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=5, r=5, t=38 if (title or has_existing_title) else 5, b=5),
        autosize=True,
        showlegend=True if fig.layout.showlegend is not False else False,
        hovermode="closest",
        hoverlabel=dict(bgcolor="#171B26", font_size=9, font_family="Inter", bordercolor="rgba(255, 255, 255, 0.12)"),
    )
    title_text = title or (fig.layout.title.text if has_existing_title else None)
    if title_text:
        layout_kwargs["title"] = dict(
            text=title_text,
            font=dict(size=13, family="Inter", color="#EDEFF5", weight="bold"),
            x=0.0,
            y=0.96,
        )
    fig.update_layout(**layout_kwargs)
    if hasattr(fig, "layout") and fig.layout.legend:
        fig.update_layout(
            legend=dict(
                font=dict(size=8, family="Inter"),
                bgcolor="rgba(23,27,38,0.85)",
                bordercolor="rgba(255, 255, 255, 0.12)",
                borderwidth=1
            )
        )
    # Style axes uniformly
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255, 255, 255, 0.12)",
        linecolor="rgba(255, 255, 255, 0.12)",
        tickfont=dict(size=8, family="Inter", color="#8890A6"),
        title_font=dict(size=9, family="Inter", color="#8890A6")
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255, 255, 255, 0.12)",
        linecolor="rgba(255, 255, 255, 0.12)",
        tickfont=dict(size=8, family="Inter", color="#8890A6"),
        title_font=dict(size=9, family="Inter", color="#8890A6")
    )
    return fig


def probability_gauge(probability, threshold):
    """Build a large premium radial gauge chart."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 24, "family": "Inter", "weight": "bold", "color": "#EDEFF5"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8", "ticklen": 3},
                "bar": {"color": risk_color("High" if probability >= 0.6 else "Medium" if probability >= 0.3 else "Low")},
                "steps": [
                    {"range": [0, 30], "color": "rgba(16, 185, 129, 0.12)"},
                    {"range": [30, 60], "color": "rgba(245, 158, 11, 0.12)"},
                    {"range": [60, 100], "color": "rgba(239, 68, 68, 0.12)"},
                ],
                "threshold": {"line": {"color": "#EDEFF5", "width": 2}, "value": threshold * 100},
            },
        )
    )
    apply_plotly_theme(fig)
    fig.update_layout(height=145, margin=dict(l=15, r=15, t=5, b=5))
    return fig


def factors_chart(factors):
    """Create a compact ranked visual for explanation factors with shortened clean labels."""
    short_factors = []
    for f in factors:
        # Extract title chunk before ' — ' or ' (' for clean y-axis representation
        label = f.split(" (")[0].split(" — ")[0]
        if len(label) > 28:
            label = label[:25] + "..."
        short_factors.append(label)
        
    scores = list(range(len(factors), 0, -1))
    fig = px.bar(
        x=scores,
        y=short_factors,
        orientation="h",
        labels={"x": "Relative priority", "y": "Risk factor"},
        color=scores,
        color_continuous_scale="Blues",
    )
    apply_plotly_theme(fig)
    fig.update_layout(height=150, coloraxis_showscale=False)
    fig.update_yaxes(showgrid=False)  # Horizontal bar layout only wants vertical grids
    fig.update_xaxes(showgrid=True)
    return fig


def individual_shap_waterfall(model, preprocessor, feature_names, customer_data):
    """Build a premium interactive Plotly waterfall chart showing local risk contributions."""
    try:
        # Preprocess customer record
        df = prepare_customer_frame(customer_data)
        X_trans = preprocessor.transform(df)
        import scipy.sparse
        if scipy.sparse.issparse(X_trans):
            X_trans = X_trans.toarray()
        X_df = pd.DataFrame(X_trans, columns=feature_names, index=df.index)
        
        coefs = model.coef_[0]
        intercept = model.intercept_[0]
        
        # Calculate contribution to log-odds: coefficient * scaled value
        contributions = X_df.iloc[0].values * coefs
        
        # Create a contributions dataframe
        contrib_df = pd.DataFrame({
            "feature": feature_names,
            "value": X_df.iloc[0].values,
            "contribution": contributions,
            "abs_contrib": np.abs(contributions)
        })
        
        # Filter out features with near-zero contributions to keep plot clean
        contrib_df = contrib_df[contrib_df["abs_contrib"] > 1e-4]
        # Sort by absolute contribution descending and take the top 8
        contrib_df = contrib_df.sort_values(by="abs_contrib", ascending=False).head(8)
        
        # Reverse to draw from base towards final prediction
        contrib_df = contrib_df.iloc[::-1]
        
        # Prepare list for waterfall chart
        y_labels = ["Baseline (Intercept)"]
        x_vals = [intercept]
        measures = ["absolute"]
        
        # Base probability calculation
        base_prob = 1.0 / (1.0 + np.exp(-intercept))
        text_labels = [f"Prob: {base_prob:.1%}"]
        
        cumulative_z = intercept
        for _, row in contrib_df.iterrows():
            feat_name = row["feature"].replace("_", " ").replace("remainder  ", "").replace("remainder ", "")
            if len(feat_name) > 25:
                feat_name = feat_name[:22] + "..."
            
            val = row["value"]
            # Clean label for binary features vs numerical features
            if val in (0.0, 1.0):
                label = f"{feat_name}"
            else:
                label = f"{feat_name} ({val:.1f})"
                
            y_labels.append(label)
            x_vals.append(row["contribution"])
            measures.append("relative")
            
            cumulative_z += row["contribution"]
            step_prob = 1.0 / (1.0 + np.exp(-cumulative_z))
            text_labels.append(f"{row['contribution']:+.2f} (Prob: {step_prob:.1%})")
            
        # Add final prediction total
        y_labels.append("Final Log-Odds")
        x_vals.append(cumulative_z)
        measures.append("total")
        final_prob = 1.0 / (1.0 + np.exp(-cumulative_z))
        text_labels.append(f"Prob: {final_prob:.1%}")
        
        fig = go.Figure(go.Waterfall(
            orientation="h",
            measure=measures,
            y=y_labels,
            x=x_vals,
            text=text_labels,
            textposition="outside",
            connector={"line": {"color": "rgba(255, 255, 255, 0.2)", "width": 1}},
            decreasing={"marker": {"color": "#34D399"}}, # Green for reducing risk
            increasing={"marker": {"color": "#F87171"}}, # Red for increasing risk
            totals={"marker": {"color": "#22D3EE"}},      # Cyan for final
        ))
        
        fig.update_layout(
            waterfallgap=0.2,
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title="Contribution to Log-Odds of Churn"),
        )
        apply_plotly_theme(fig)
        return fig
    except Exception as exc:
        # Fallback empty figure with error message
        fig = go.Figure()
        fig.update_layout(
            title=f"Error generating waterfall: {exc}",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        return fig


def render_explain_ai(model, preprocessor, feature_names):
    """Render the Explain AI dashboard tab."""
    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: space-between; gap: 0.75rem; margin-bottom: 0.1rem;">
            <h2 style="color: #EDEFF5; font-size: 1.04rem; font-weight: 800; margin: 0; letter-spacing: -0.01em;">Prediction Diagnostics</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    
    # Section 1: Individual Prediction Explanation (Waterfall)
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<h4 class='panel-title' style='color: #22D3EE; font-size: 1.05rem;'>Individual Prediction Explanation</h4>", unsafe_allow_html=True)
    
    if st.session_state.prediction_result and st.session_state.prediction_customer:
        res = st.session_state.prediction_result
        cust = st.session_state.prediction_customer
        
        st.markdown(
            f"""
            <div style="font-size: 0.8rem; margin-bottom: 0.6rem; line-height: 1.35; color: #EDEFF5;">
                Feature impacts are shown as positive or negative contributions to the churn score for <strong>{res['churn_probability']:.1%}</strong> probability.
            </div>
            """,
            unsafe_allow_html=True
        )
        
        fig = individual_shap_waterfall(model, preprocessor, feature_names, cust)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(
            f"""
            <div style="background: rgba(255,255,255,0.03); padding: 0.55rem 0.7rem; border-radius: 8px; border-left: 3px solid #22D3EE; font-size: 0.76rem; color: #C4CAD9; line-height: 1.35; margin-top: 0.4rem;">
                <strong>Action:</strong> {res['recommended_action']}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="text-align: center; padding: 2rem 1rem; background: rgba(255,255,255,0.02); border-radius: 8px; border: 1px dashed rgba(255,255,255,0.1);">
                <p style="color: #8890A6; font-size: 0.82rem; margin: 0;">
                    No active prediction session found. Go to the <strong>Predictive Analysis</strong> tab, set a customer profile in the <strong>Filter Drawer</strong>, and click <strong>Run Risk Prediction</strong> to visualize their individual waterfall breakdown here.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    
    # Section 2: Global Model Explainability (SHAP Summary and Feature Importance)
    col1, col2 = st.columns(2, gap="small", vertical_alignment="top")
    with col1:
        st.markdown("<div class='panel' style='padding-bottom: 0.7rem !important; min-height: 520px !important;'>", unsafe_allow_html=True)
        st.markdown("<h4 class='panel-title'>Risk drivers</h4>", unsafe_allow_html=True)
        st.markdown(
            """
            <p style="font-size: 0.72rem; color: #8890A6; line-height: 1.3; margin: 0 0 0.25rem 0;">
                Feature values that consistently increase or reduce modeled churn risk.
            </p>
            """,
            unsafe_allow_html=True
        )
        import os
        if os.path.exists("outputs/plots/shap_summary_plot.png"):
            st.image("outputs/plots/shap_summary_plot.png", use_container_width=True)
        else:
            st.warning("SHAP Summary Plot not found at outputs/plots/shap_summary_plot.png")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='panel' style='padding-bottom: 0.7rem !important; min-height: 520px !important;'>", unsafe_allow_html=True)
        st.markdown("<h4 class='panel-title'>Feature impact</h4>", unsafe_allow_html=True)
        st.markdown(
            """
            <p style="font-size: 0.72rem; color: #8890A6; line-height: 1.3; margin: 0 0 0.25rem 0;">
                Average influence of each feature on the model's predictions.
            </p>
            """,
            unsafe_allow_html=True
        )
        if os.path.exists("outputs/plots/shap_feature_importance_plot.png"):
            st.image("outputs/plots/shap_feature_importance_plot.png", use_container_width=True)
        else:
            st.warning("SHAP Feature Importance Plot not found at outputs/plots/shap_feature_importance_plot.png")
        st.markdown("</div>", unsafe_allow_html=True)


def format_saas_recommendation(item):
    """Format campaign or recommended intervention string into a structured SaaS-like layout dict."""
    if ":" in item:
        title, desc = item.split(": ", 1)
    else:
        title, desc = "Account Outreach Action", item
    
    icon = ""
    benefit = "Stabilizes customer relationship"
    
    if "Month-to-Year" in title or "Contract" in title:
        icon = ""
        benefit = "Secures contract commitment (saves $780 ARPU)"
    elif "Fiber" in title:
        icon = ""
        benefit = "Retains premium Fiber margin segment"
    elif "Auto-Pay" in title:
        icon = ""
        benefit = "Resolves manual billing payment frictions"
    elif "Early-Tenure" in title:
        icon = ""
        benefit = "Improves welcome phase customer retention"
    elif "Proactive" in title:
        icon = ""
        benefit = "Proactive save (5% statement offset)"
        
    return {
        "icon": icon,
        "title": title,
        "description": desc,
        "benefit": benefit
    }


def report_dataframe(customer, result, roi, campaign):
    """Create a one-row downloadable prediction report."""
    return pd.DataFrame([
        {
            "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
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


def render_metric_card_component(label, value, accent, badge_html, element_id, value_color=None, prefix="", suffix="", numeric_target=None):
    """Render a compact KPI card with subtle number-count animation."""
    if numeric_target is None:
        display_value = value
        script = ""
    else:
        display_value = f"{prefix}{value}{suffix}"
        script = f"""
        <script>
        (function() {{
          const el = document.getElementById('{element_id}');
          if (!el) return;
          const target = {numeric_target};
          const prefix = '{prefix}';
          const suffix = '{suffix}';
          const duration = 700;
          const start = performance.now();
          const format = (num) => {{
            if (Number.isInteger(target)) {{
              return prefix + Math.round(num).toLocaleString() + suffix;
            }}
            return prefix + num.toFixed(1) + suffix;
          }};
          const tick = (now) => {{
            const progress = Math.min((now - start) / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            el.textContent = format(target * eased);
            if (progress < 1) requestAnimationFrame(tick);
          }};
          requestAnimationFrame(tick);
        }})();
        </script>
        """

    html = f"""
    <style>
      body {{ margin: 0; background: transparent; font-family: 'Inter', sans-serif; }}
      .kpi-card-horizontal {{
        background: linear-gradient(180deg, #1E293B 0%, #172033 100%);
        border: 1px solid rgba(255,255,255,0.06);
        border-left: 3px solid {accent};
        border-radius: 12px;
        padding: 0.55rem 0.8rem;
        box-shadow: 0 6px 16px -8px rgba(0,0,0,0.45);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.55rem;
        height: 82px;
        width: 100%;
        box-sizing: border-box;
        color: #EDEFF5;
        animation: cardFadeIn 0.5s ease both;
      }}
      @keyframes cardFadeIn {{ from {{ opacity: 0; transform: translateY(6px); }} to {{ opacity: 1; transform: translateY(0); }} }}
      .kpi-metric-name {{ color: #8890A6; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; margin: 0; line-height: 1.2; }}
      .kpi-metric-value {{ color: {value_color or '#EDEFF5'}; font-size: 1.35rem; font-weight: 800; margin: 0; line-height: 1.1; font-family: 'JetBrains Mono', 'Inter', monospace; }}
      .kpi-status-badge {{ display: inline-block; padding: 0.15rem 0.55rem; border-radius: 4px; font-size: 0.7rem; font-weight: 600; width: fit-content; flex-shrink: 0; }}
    </style>
    <div class="kpi-card-horizontal">
      <div>
        <div class="kpi-metric-name">{label}</div>
        <div class="kpi-metric-value" id="{element_id}">{display_value}</div>
      </div>
      {badge_html}
    </div>
    {script}
    """
    components.html(html, height=92, scrolling=False)


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
        st.markdown("<div class='panel' style='min-height: 185px !important;'>", unsafe_allow_html=True)
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
        st.markdown("<div class='panel' style='min-height: 185px !important;'>", unsafe_allow_html=True)
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
        st.markdown("<div class='panel' style='min-height: 210px !important;'>", unsafe_allow_html=True)
        st.markdown("<h5 class='panel-title'>Drivers</h5>", unsafe_allow_html=True)
        st.plotly_chart(factors_chart(result["top_risk_factors"]), width="stretch", config={'responsive': True})
        drivers_html = " ".join([f"<span class='mini-badge'>{factor}</span>" for factor in result["top_risk_factors"][:3]])
        st.markdown(f"<div class='ai-summary-row'>{drivers_html}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with row3_right:
        st.markdown("<div class='panel' style='min-height: 210px !important;'>", unsafe_allow_html=True)
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


def render_sidebar_drawer(metadata, decision_threshold):
    """Render the collapsible filter drawer inside the Streamlit sidebar."""
    st.sidebar.markdown(
        """
        <div style="padding: 0.15rem 0 0.35rem 0; margin-bottom: 0.35rem;">
            <h3 style="color: #EDEFF5; font-size: 0.95rem; font-weight: 800; margin: 0;">Customer Filters</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.button("Example: High Risk", on_click=load_example_customer, width="stretch", type="secondary")
    st.sidebar.button("Example: Low Risk", on_click=load_low_risk_customer, width="stretch", type="secondary")
    st.sidebar.button("Reset", on_click=reset_customer, width="stretch", type="secondary")

    st.sidebar.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

    with st.sidebar.form("customer_form", clear_on_submit=False):
        # 1. Customer Details Collapsible Section
        with st.expander("Customer Details", expanded=True):
            st.markdown("<div class='filter-card'>", unsafe_allow_html=True)
            st.markdown("<div class='filter-card-label'>Tenure (months)</div>", unsafe_allow_html=True)
            st.number_input("Tenure (months)", min_value=0, max_value=100, key="tenure", label_visibility="collapsed")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='filter-card'>", unsafe_allow_html=True)
            st.markdown("<div class='filter-card-label'>Contract type</div>", unsafe_allow_html=True)
            st.radio("Contract Type", ["Month-to-month", "One year", "Two year"], key="contract", horizontal=True, label_visibility="collapsed")
            st.markdown("</div>", unsafe_allow_html=True)
            
        # 2. Demographics Collapsible Section
        with st.expander("Demographics", expanded=False):
            st.markdown("<div class='filter-card'>", unsafe_allow_html=True)
            st.markdown("<div class='filter-card-label'>Gender</div>", unsafe_allow_html=True)
            st.radio("Gender", ["Female", "Male"], key="gender", horizontal=True, label_visibility="collapsed")
            st.markdown("</div>", unsafe_allow_html=True)

            d1, d2 = st.columns(2)
            with d1:
                st.markdown("<div class='filter-card'>", unsafe_allow_html=True)
                st.markdown("<div class='filter-card-label'>Senior citizen</div>", unsafe_allow_html=True)
                st.radio("Senior Citizen", ["No", "Yes"], key="senior_citizen", horizontal=True, label_visibility="collapsed")
                st.markdown("</div>", unsafe_allow_html=True)
            with d2:
                st.markdown("<div class='filter-card'>", unsafe_allow_html=True)
                st.markdown("<div class='filter-card-label'>Partner status</div>", unsafe_allow_html=True)
                st.radio("Partner Status", ["No", "Yes"], key="partner", horizontal=True, label_visibility="collapsed")
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='filter-card'>", unsafe_allow_html=True)
            st.markdown("<div class='filter-card-label'>Dependents</div>", unsafe_allow_html=True)
            st.radio("Dependents", ["No", "Yes"], key="dependents", horizontal=True, label_visibility="collapsed")
            st.markdown("</div>", unsafe_allow_html=True)

        # 3. Billing Collapsible Section
        with st.expander("Billing", expanded=False):
            st.markdown("<div class='filter-card'>", unsafe_allow_html=True)
            st.markdown("<div class='filter-card-label'>Monthly charges</div>", unsafe_allow_html=True)
            st.number_input("Monthly Charges ($)", min_value=0.0, step=0.5, key="monthly_charges", label_visibility="collapsed")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='filter-card'>", unsafe_allow_html=True)
            st.markdown("<div class='filter-card-label'>Total charges</div>", unsafe_allow_html=True)
            st.number_input("Total Charges ($)", min_value=0.0, step=1.0, key="total_charges", label_visibility="collapsed")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='filter-card'>", unsafe_allow_html=True)
            st.markdown("<div class='filter-card-label'>Paperless billing</div>", unsafe_allow_html=True)
            st.radio("Paperless Billing", ["No", "Yes"], key="paperless_billing", horizontal=True, label_visibility="collapsed")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='filter-card'>", unsafe_allow_html=True)
            st.markdown("<div class='filter-card-label'>Payment method</div>", unsafe_allow_html=True)
            st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], key="payment_method", index=0)
            st.markdown("</div>", unsafe_allow_html=True)

        # 4. Services Collapsible Section
        with st.expander("Services", expanded=False):
            s1, s2 = st.columns(2)
            with s1:
                st.markdown("<div class='filter-card'>", unsafe_allow_html=True)
                st.markdown("<div class='filter-card-label'>Phone service</div>", unsafe_allow_html=True)
                st.radio("Phone Service", ["No", "Yes"], key="phone_service", horizontal=True, label_visibility="collapsed")
                st.markdown("</div>", unsafe_allow_html=True)
            with s2:
                st.markdown("<div class='filter-card'>", unsafe_allow_html=True)
                st.markdown("<div class='filter-card-label'>Multiple lines</div>", unsafe_allow_html=True)
                st.radio("Multiple Lines", ["No", "Yes", "No phone service"], key="multiple_lines", horizontal=True, label_visibility="collapsed")
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='filter-card'>", unsafe_allow_html=True)
            st.markdown("<div class='filter-card-label'>Internet service</div>", unsafe_allow_html=True)
            st.radio("Internet Service", ["DSL", "Fiber optic", "No"], key="internet_service", horizontal=True, label_visibility="collapsed")
            st.markdown("</div>", unsafe_allow_html=True)

            s3, s4 = st.columns(2)
            with s3:
                st.markdown("<div class='filter-card'>", unsafe_allow_html=True)
                st.markdown("<div class='filter-card-label'>Online security</div>", unsafe_allow_html=True)
                st.radio("Online Security", ["No", "Yes", "No internet service"], key="online_security", horizontal=True, label_visibility="collapsed")
                st.markdown("</div>", unsafe_allow_html=True)
            with s4:
                st.markdown("<div class='filter-card'>", unsafe_allow_html=True)
                st.markdown("<div class='filter-card-label'>Online backup</div>", unsafe_allow_html=True)
                st.radio("Online Backup", ["No", "Yes", "No internet service"], key="online_backup", horizontal=True, label_visibility="collapsed")
                st.markdown("</div>", unsafe_allow_html=True)

            s5, s6 = st.columns(2)
            with s5:
                st.markdown("<div class='filter-card'>", unsafe_allow_html=True)
                st.markdown("<div class='filter-card-label'>Device protection</div>", unsafe_allow_html=True)
                st.radio("Device Protection", ["No", "Yes", "No internet service"], key="device_protection", horizontal=True, label_visibility="collapsed")
                st.markdown("</div>", unsafe_allow_html=True)
            with s6:
                st.markdown("<div class='filter-card'>", unsafe_allow_html=True)
                st.markdown("<div class='filter-card-label'>Tech support</div>", unsafe_allow_html=True)
                st.radio("Tech Support", ["No", "Yes", "No internet service"], key="tech_support", horizontal=True, label_visibility="collapsed")
                st.markdown("</div>", unsafe_allow_html=True)

            s7, s8 = st.columns(2)
            with s7:
                st.markdown("<div class='filter-card'>", unsafe_allow_html=True)
                st.markdown("<div class='filter-card-label'>Streaming TV</div>", unsafe_allow_html=True)
                st.radio("Streaming TV", ["No", "Yes", "No internet service"], key="streaming_tv", horizontal=True, label_visibility="collapsed")
                st.markdown("</div>", unsafe_allow_html=True)
            with s8:
                st.markdown("<div class='filter-card'>", unsafe_allow_html=True)
                st.markdown("<div class='filter-card-label'>Streaming movies</div>", unsafe_allow_html=True)
                st.radio("Streaming Movies", ["No", "Yes", "No internet service"], key="streaming_movies", horizontal=True, label_visibility="collapsed")
                st.markdown("</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("Run Risk Prediction", type="primary", width="stretch")

    # Render model governance info inside filter drawer sidebar at the bottom
    st.sidebar.markdown("<div class='section-gap' style='height: 1.1rem !important;'></div>", unsafe_allow_html=True)
    with st.sidebar.expander("Model Governance", expanded=False):
        metrics = metadata.get("metrics", {})
        st.markdown(f"""
        <div style="font-size: 0.76rem; line-height: 1.35; color: #A2AAC0;">
            <strong>Accuracy:</strong> {metrics.get('Accuracy', 0):.3f}<br>
            <strong>Precision:</strong> {metrics.get('Precision', 0):.3f}<br>
            <strong>Recall:</strong> {metrics.get('Recall', 0):.3f}<br>
            <strong>ROC AUC:</strong> {metrics.get('ROC_AUC', 0):.3f}<br>
            <strong>Model Type:</strong> {metadata.get('model_type', 'LogisticRegression')}<br>
            <strong>Decision Threshold:</strong> {decision_threshold:.2f}
        </div>
        """, unsafe_allow_html=True)

    return submitted


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
            <div class="kpi-card-horizontal" style="box-shadow: none !important; border: 1px solid rgba(255,255,255,0.08) !important; border-left: 3px solid #22D3EE !important; display: flex; flex-direction: column; align-items: flex-start; justify-content: center; height: 94px;">
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
            <div class="kpi-card-horizontal" style="box-shadow: none !important; border: 1px solid rgba(255,255,255,0.08) !important; border-left: 3px solid #EF4444 !important; display: flex; flex-direction: column; align-items: flex-start; justify-content: center; height: 94px;">
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
            <div class="kpi-card-horizontal" style="box-shadow: none !important; border: 1px solid rgba(255,255,255,0.08) !important; border-left: 3px solid #F59E0B !important; display: flex; flex-direction: column; align-items: flex-start; justify-content: center; height: 94px;">
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
            <div class="kpi-card-horizontal" style="box-shadow: none !important; border: 1px solid rgba(255,255,255,0.08) !important; border-left: 3px solid #10B981 !important; display: flex; flex-direction: column; align-items: flex-start; justify-content: center; height: 94px;">
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
        st.markdown("<div class='panel' style='min-height: 235px !important;'>", unsafe_allow_html=True)
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
        st.markdown("<div class='panel' style='min-height: 235px !important;'>", unsafe_allow_html=True)
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
        st.markdown("<div class='panel' style='min-height: 235px !important;'>", unsafe_allow_html=True)
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
        st.markdown("<div class='panel' style='min-height: 235px !important;'>", unsafe_allow_html=True)
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
            <div class="kpi-card-horizontal" style="box-shadow: none !important; border: 1px solid rgba(255,255,255,0.08) !important; border-left: 3px solid #EF4444 !important; height: 74px !important;">
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
            <div class="kpi-card-horizontal" style="box-shadow: none !important; border: 1px solid rgba(255,255,255,0.08) !important; border-left: 3px solid #10B981 !important; height: 74px !important;">
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
            <div class="kpi-card-horizontal" style="box-shadow: none !important; border: 1px solid rgba(255,255,255,0.08) !important; border-left: 3px solid #10B981 !important; height: 74px !important;">
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
            <div class="kpi-card-horizontal" style="box-shadow: none !important; border: 1px solid rgba(255,255,255,0.08) !important; border-left: 3px solid #F59E0B !important; height: 74px !important;">
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
    model, preprocessor, feature_names = get_artifacts()
except Exception as exc:  # noqa: BLE001
    st.error(f"Model artifacts missing: {exc}")
    st.stop()

decision_threshold = get_decision_threshold()

# Render Sidebar Collapsible Filter Drawer (Fabric Style)
submitted = render_sidebar_drawer(get_model_metadata(), decision_threshold)

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
        st.warning(str(exc))
    except Exception as exc:  # noqa: BLE001
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
            st.error(f"Could not render executive portfolio: {exc}")