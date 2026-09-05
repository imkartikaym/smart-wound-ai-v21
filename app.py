"""
INFECTION DETECTOR AI - Premium Biomedical AI SaaS Interface
DO NOT CHANGE ML PIPELINE - UI/UX REDESIGN ONLY
Multimodal Temporal Sensor & AI-Based Early-Warning System for Wound Infection Detection
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from utils import (
    WoundFeatureEngine, RiskAssessment, SensorSimulator,
    TemporalAnalyzer, get_model_metadata
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Infection Detector AI - Wound Infection Early-Warning System",
    page_icon="🩹",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/smartwoundai',
        'Report a bug': 'https://github.com/smartwoundai/issues',
        'About': "Infection Detector AI - Multimodal Temporal Sensor & AI-Based Early-Warning System"
    }
)

# ============================================================
# PREMIUM BIOMEDICAL AI SAAS CSS
# ============================================================

premium_css = """
<style>
:root {
    --bg-primary: #080D1F;
    --bg-secondary: #111A2C;
    --bg-tertiary: #151F33;
    --accent-cyan: #00D9FF;
    --accent-purple: #7C5CFF;
    --text-primary: #F8FAFC;
    --text-secondary: #A8B4C5;
    --text-muted: #718096;
    --risk-low: #35D07F;
    --risk-monitor: #F6C76D;
    --risk-high: #FF5C67;
    --border-subtle: rgba(168, 180, 197, 0.15);
    --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.3);
    --shadow-md: 0 8px 24px rgba(0, 0, 0, 0.4);
}

* {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", sans-serif;
}

body, .stApp, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-primary) !important;
}

.main {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-subtle) !important;
}

/* Text Visibility - CRITICAL */
h1, h2, h3, h4, h5, h6 {
    color: var(--text-primary) !important;
}

p, span, label, div, li {
    color: var(--text-primary) !important;
}

[data-testid="stMarkdownContainer"] {
    color: var(--text-primary) !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] span {
    color: var(--text-primary) !important;
}

/* Header Section */
.header-premium {
    padding: 2.5rem 0 1.5rem 0;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 2.5rem;
}

.header-premium h1 {
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.5px;
}

.header-premium p {
    font-size: 0.95rem;
    color: var(--text-secondary);
    margin: 0;
    font-weight: 400;
}

/* Badges */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background-color: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    font-size: 0.85rem;
    color: var(--text-secondary);
    font-weight: 500;
    margin-top: 1rem;
    margin-right: 0.5rem;
}

.badge-indicator {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background-color: var(--risk-low);
}

/* Card Base */
.card {
    background-color: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 1.5rem;
    box-shadow: var(--shadow-sm);
    transition: all 0.2s ease;
}

/* Primary Risk Card - MOST IMPORTANT */
.risk-result-card {
    background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    box-shadow: var(--shadow-md);
    margin: 2rem 0;
}

.risk-probability-value {
    font-size: 4.5rem;
    font-weight: 700;
    line-height: 1;
    margin: 1rem 0;
    font-variant-numeric: tabular-nums;
}

.risk-level-text {
    font-size: 1.25rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

.risk-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
}

.risk-high {
    color: var(--risk-high);
}

.risk-monitor {
    color: var(--risk-monitor);
}

.risk-low {
    color: var(--risk-low);
}

/* Biomarker Grid */
.biomarker-card {
    background-color: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}

.biomarker-name {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
}

.biomarker-value {
    font-size: 2rem;
    font-weight: 700;
    font-family: 'Courier New', monospace;
    color: var(--accent-cyan);
    margin-bottom: 0.5rem;
}

.biomarker-unit {
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-bottom: 0.75rem;
}

.biomarker-velocity {
    font-size: 0.85rem;
    color: var(--text-secondary);
    font-weight: 500;
}

/* Section Headers */
.section-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 2.5rem 0 1.5rem 0;
    letter-spacing: -0.3px;
}

.section-subtitle {
    font-size: 0.9rem;
    color: var(--text-secondary);
    margin-bottom: 1.5rem;
    font-weight: 400;
}

/* Inputs */
input, select, textarea, [data-testid="stTextInput"], [data-testid="stSelectbox"] {
    background-color: var(--bg-tertiary) !important;
    border: 1px solid var(--border-subtle) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}

input:focus, select:focus, textarea:focus {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 0 3px rgba(0, 217, 255, 0.1) !important;
}

/* Buttons */
.stButton > button {
    background-color: var(--accent-cyan) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.5rem !important;
    font-size: 0.95rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(0, 217, 255, 0.25) !important;
}

.stButton > button:hover {
    box-shadow: 0 6px 20px rgba(0, 217, 255, 0.4) !important;
    transform: translateY(-1px) !important;
}

/* Radio & Checkbox */
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label {
    color: var(--text-primary) !important;
    font-weight: 500 !important;
}

/* Tabs */
[data-baseweb="tab-list"] {
    border-bottom-color: var(--border-subtle) !important;
}

[data-baseweb="tab"] {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
}

[aria-selected="true"] {
    color: var(--accent-cyan) !important;
    border-bottom-color: var(--accent-cyan) !important;
}

/* Alert Boxes */
.stInfo, .stWarning, .stSuccess, .stError {
    background-color: transparent !important;
    border-radius: 10px !important;
    border-left: 4px solid !important;
    padding: 1.25rem !important;
    color: var(--text-primary) !important;
}

.stInfo {
    border-left-color: var(--accent-cyan) !important;
    background-color: rgba(0, 217, 255, 0.08) !important;
}

.stWarning {
    border-left-color: var(--risk-monitor) !important;
    background-color: rgba(246, 199, 109, 0.08) !important;
}

.stSuccess {
    border-left-color: var(--risk-low) !important;
    background-color: rgba(53, 208, 127, 0.08) !important;
}

.stError {
    border-left-color: var(--risk-high) !important;
    background-color: rgba(255, 92, 103, 0.08) !important;
}

/* Expander */
.streamlit-expanderHeader {
    background-color: var(--bg-tertiary) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
}

.streamlit-expanderHeader:hover {
    background-color: var(--bg-tertiary) !important;
    border-color: var(--accent-cyan) !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-secondary);
}

::-webkit-scrollbar-thumb {
    background: var(--text-muted);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--accent-cyan);
}

/* Divider */
.divider {
    border-top: 1px solid var(--border-subtle);
    margin: 2.5rem 0;
}

/* Footer */
.footer-premium {
    border-top: 1px solid var(--border-subtle);
    padding-top: 2rem;
    margin-top: 3rem;
    text-align: center;
    color: var(--text-secondary);
    font-size: 0.85rem;
}

</style>
"""

st.markdown(premium_css, unsafe_allow_html=True)
st.markdown("""
<style>

/* ---------- GLOBAL LAYOUT ---------- */

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

/* ---------- SIDEBAR ---------- */

section[data-testid="stSidebar"] {
    padding-top: 1rem;
}

section[data-testid="stSidebar"] > div {
    padding-left: 1.2rem;
    padding-right: 1.2rem;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    margin-top: 0.4rem;
    margin-bottom: 0.8rem;
}

/* ---------- MAIN TITLE ---------- */

.main-title {
    margin-top: 0.2rem;
    margin-bottom: 0.2rem;
    line-height: 1.05;
}

.main-subtitle {
    margin-top: 0.25rem;
    margin-bottom: 1.2rem;
    opacity: 0.85;
}

/* ---------- STATUS BADGES ---------- */

.status-row {
    display: flex;
    gap: 0.7rem;
    margin: 0.8rem 0 1.4rem 0;
    flex-wrap: wrap;
}

.status-badge {
    padding: 0.45rem 0.85rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.03em;
}

/* ---------- SECTION SPACING ---------- */

.section-title {
    margin-top: 1.2rem !important;
    margin-bottom: 0.35rem !important;
}

.section-subtitle {
    margin-top: 0 !important;
    margin-bottom: 1rem !important;
    opacity: 0.82;
}

.divider {
    margin: 1.2rem 0 !important;
}

/* ---------- PATIENT SESSION ---------- */

.patient-session {
    margin: 0.2rem 0 1.2rem 0;
}

/* ---------- BIOMARKER GRID ---------- */

.sensor-grid {
    margin-top: 0.8rem;
    margin-bottom: 1rem;
}

.sensor-card {
    min-height: 125px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
}

/* ---------- ACTION BUTTONS ---------- */

div.stButton > button {
    min-height: 52px;
    font-weight: 700;
    border-radius: 12px;
}

/* ---------- EXPANDERS ---------- */

div[data-testid="stExpander"] {
    margin-top: 0.5rem;
    margin-bottom: 0.8rem;
}

/* ---------- PLOTS ---------- */

.js-plotly-plot {
    margin-top: 0.4rem;
    margin-bottom: 1rem;
}

/* ---------- DISCLAIMER ---------- */

.disclaimer {
    margin-top: 1.5rem;
    padding-top: 1rem;
}

</style>
""", unsafe_allow_html=True)
# ============================================================
# LOAD MODEL (ML PIPELINE - UNCHANGED)
# ============================================================

@st.cache_resource
def load_model_package():
    """Load the frozen V2.1 Random Forest Pipeline and metadata."""

    model_file = "smart_wound_v21_random_forest.joblib"
    feature_file = "feature_order.json"
    biomarker_file = "biomarkers.json"
    metadata_file = "model_metadata.json"

    if not os.path.exists(model_file):
        st.error(f"Frozen V2.1 model not found: {model_file}")
        st.stop()

    # Load frozen sklearn Pipeline
    model = joblib.load(model_file)

    # Load frozen 24-feature order
    with open(feature_file, "r", encoding="utf-8") as f:
        features = json.load(f)

    # Load biomarker list
    with open(biomarker_file, "r", encoding="utf-8") as f:
        biomarkers = json.load(f)

    # Load model metadata
    metadata = {}
    if os.path.exists(metadata_file):
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return {
        "model": model,
        "features": features,
        "biomarkers": biomarkers,
        "metadata": metadata
    }

@st.cache_resource
def initialize_components():
    """Initialize reusable ML components."""

    package = load_model_package()

    model = package["model"]

    features = package["features"]

    biomarkers = package["biomarkers"]

    feature_engine = WoundFeatureEngine(
        biomarkers=biomarkers,
        window_size=5
    )

    simulator = SensorSimulator(
        seed=42
    )

    return (
        model,
        features,
        biomarkers,
        feature_engine,
        simulator,
        package
    )


try:

    (
        model,
        features,
        biomarkers,
        feature_engine,
        simulator,
        model_package
    ) = initialize_components()

except Exception as e:

    st.error(
        f"Failed to initialize: {e}"
    )

    st.stop()


# ------------------------------------------------------------
# Model evaluation metadata
# ------------------------------------------------------------

TEST_METRICS = {
    "roc_auc": model_package.get("metadata", {}).get(
        "independent_test_roc_auc", 0.0
    ),
    "accuracy": model_package.get("metadata", {}).get(
        "independent_test_accuracy", 0.0
    ),
    "precision": model_package.get("metadata", {}).get(
        "independent_test_precision", 0.0
    ),
    "recall": model_package.get("metadata", {}).get(
        "independent_test_recall", 0.0
    ),
    "f1": model_package.get("metadata", {}).get(
        "independent_test_f1", 0.0
    ),
    "specificity": model_package.get("metadata", {}).get(
        "independent_test_specificity", 0.0
    ),
    "sensitivity": model_package.get("metadata", {}).get(
        "independent_test_sensitivity", 0.0
    ),
    "pr_auc": model_package.get("metadata", {}).get(
        "independent_test_pr_auc", 0.0
    ),
    "brier": model_package.get("metadata", {}).get(
        "brier_score", 0.0
    )
}
# ============================================================
# SESSION STATE
# ============================================================

if "readings_history" not in st.session_state:
    st.session_state.readings_history = []

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "patient_id" not in st.session_state:
    st.session_state.patient_id = "PATIENT-001"

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def predict_risk(X_features):
    """Get infection risk prediction from model"""
    if X_features is None:
        return None
    
    try:
        probability = float(model.predict_proba(X_features)[0, 1])
        return probability
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# ============================================================
# PREMIUM HEADER - COMPACT & CLEAN
# ============================================================

st.markdown("""
<div class="header-premium">
    <h1>INFECTION DETECTOR AI</h1>
    <p>Multimodal Wound Infection Early-Warning System</p>
</div>
""", unsafe_allow_html=True)

# Status badges
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""<span class="badge"><span class="badge-indicator"></span>SYSTEM ONLINE</span>""", unsafe_allow_html=True)
with col2:
    st.markdown("""<span class="badge">RANDOM FOREST</span>""", unsafe_allow_html=True)
with col3:
    st.markdown("""<span class="badge">24 FEATURES</span>""", unsafe_allow_html=True)
with col4:
    st.markdown("""<span class="badge">WINDOW 0–4</span>""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR - SIMPLIFIED CONTROL PANEL
# ============================================================

with st.sidebar:
    st.markdown("""
    <h3 style="color: var(--text-primary); margin-top: 0;">CONTROL PANEL</h3>
    """, unsafe_allow_html=True)
    
    # Patient ID
    st.markdown("<p style='color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; margin: 2rem 0 0.75rem 0;'>PATIENT</p>", unsafe_allow_html=True)
    patient_id = st.text_input(
        "Patient ID",
        value=st.session_state.patient_id,
        key="patient_input",
        placeholder="e.g., PATIENT-001",
        label_visibility="collapsed"
    )
    st.session_state.patient_id = patient_id
    
# Input Mode
st.markdown(
    "<p style='color: var(--text-muted); font-size: 0.8rem; "
    "text-transform: uppercase; letter-spacing: 1px; font-weight: 600; "
    "margin: 1.5rem 0 0.75rem 0;'>INPUT MODE</p>",
    unsafe_allow_html=True
)

input_mode = st.radio(
    "Select input mode:",
    [
        "Manual Sensor Input",

        "Simulated Healthy",
        "Simulated Risk Progression"
    ],
    key="input_mode",
    label_visibility="collapsed"
)

# System Status
st.markdown(
    "<p style='color: var(--text-muted); font-size: 0.8rem; "
    "text-transform: uppercase; letter-spacing: 1px; font-weight: 600; "
    "margin: 1.5rem 0 0.75rem 0;'>SYSTEM STATUS</p>",
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        '<span style="color: var(--risk-low);">●</span> ONLINE',
        unsafe_allow_html=True
    )

with col2:
    meta = get_model_metadata()
    st.markdown(
        f'<span style="color: var(--text-secondary);">'
        f'{meta["model_type"]}</span>',
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f'<span style="color: var(--text-secondary);">'
        f'{meta["features_count"]} FEATURES</span>',
        unsafe_allow_html=True
    )

st.markdown(
    '<div class="divider"></div>',
    unsafe_allow_html=True
)
  # Model Details
with st.expander("📋 Model Details", expanded=False):
    st.markdown(
        f"""
        **Biomarkers:** {', '.join(biomarkers)}

        **Features:** 24 Temporal (Latest, Mean, Velocity, Variability)

        **Algorithm:** Random Forest (500 trees)

        **Independent Test ROC-AUC:** {TEST_METRICS.get("roc_auc", 0.0):.4f}
        """
    )
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Actions
    st.markdown("<p style='color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; margin: 1.5rem 0 0.75rem 0;'>ACTIONS</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 EXPORT", use_container_width=True, key="export_data"):
            if st.session_state.readings_history:
                csv = pd.DataFrame(st.session_state.readings_history).to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"readings_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_csv"
                )
            else:
                st.info("No data yet")
    
    with col2:
        if st.button("🗑️ CLEAR", use_container_width=True, key="clear_all"):
            st.session_state.readings_history = []
            st.session_state.prediction_history = []
            st.rerun()

# ============================================================
# PATIENT / SESSION SECTION
# ============================================================

st.markdown(f"""
<div class="section-title">PATIENT & SESSION</div>
<div class="section-subtitle">Currently monitoring: <strong>{patient_id}</strong> | Readings: {len(st.session_state.readings_history)}</div>
""", unsafe_allow_html=True)

# ============================================================
# PRIMARY ACTION SECTION
# ============================================================

st.markdown("""
<div class="section-title">ANALYZE INFECTION RISK</div>
<div class="section-subtitle">Configure biomarker inputs and generate risk assessment</div>
""", unsafe_allow_html=True)

if input_mode == "Manual Sensor Input":
    st.markdown("**Enter wound biomarker measurements:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ph = st.slider("pH Level", value=6.9, min_value=5.0, max_value=8.0, step=0.1)
        lactate = st.slider("Lactate (mmol/L)", value=2.2, min_value=0.0, max_value=10.0, step=0.1)
    
    with col2:
        temp = st.slider("Temperature (°C)", value=36.8, min_value=35.0, max_value=40.0, step=0.1)
        protease = st.slider("Protease (U/mL)", value=40.0, min_value=0.0, max_value=200.0, step=1.0)
    
    with col3:
        glucose = st.slider("Glucose (mg/dL)", value=110.0, min_value=50.0, max_value=300.0, step=5.0)
        oxygen = st.slider("Oxygen (%)", value=75.0, min_value=0.0, max_value=100.0, step=1.0)
    
    current_reading = {
        "pH": ph,
        "temperature": temp,
        "lactate": lactate,
        "protease": protease,
        "glucose": glucose,
        "oxygen": oxygen,
        "timestamp": datetime.now()
    }

else:
    if input_mode == "Simulated Healthy":
        st.success("🟢 Generating healthy wound biomarkers...")
        current_reading = simulator.generate_normal_reading()
        current_reading["timestamp"] = datetime.now()
    else:
        st.warning("🔴 Simulating infection risk progression...")
        current_reading = simulator.generate_elevated_risk_reading()
        current_reading["timestamp"] = datetime.now()
    
    # Display generated readings
    cols = st.columns(6)
    biomarker_colors = {
        "pH": "var(--accent-cyan)",
        "temperature": "var(--risk-high)",
        "lactate": "var(--accent-purple)",
        "protease": "var(--risk-high)",
        "glucose": "var(--accent-purple)",
        "oxygen": "var(--risk-low)"
    }
    
    for col, biomarker in zip(cols, biomarkers):
        with col:
            st.markdown(f"""
            <div class="biomarker-card">
                <div class="biomarker-name">{biomarker.upper()}</div>
                <div class="biomarker-value" style="color: {biomarker_colors.get(biomarker, 'var(--accent-cyan)')}">{current_reading[biomarker]:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

# Action buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("➕ ADD READING", use_container_width=True, key="add_reading"):
        st.session_state.readings_history.append(current_reading)
        if len(st.session_state.readings_history) > 10:
            st.session_state.readings_history = st.session_state.readings_history[-10:]
        st.success(f"✅ Reading #{len(st.session_state.readings_history)} Added")

with col2:
    if input_mode == "Simulated Risk Progression" and st.button("📈 ADD TREND (5)", use_container_width=True):
        trend = simulator.generate_deteriorating_trend(current_reading, steps=5)
        for i, reading in enumerate(trend):
            reading["timestamp"] = datetime.now() - timedelta(minutes=(5-i)*10)
            st.session_state.readings_history.append(reading)
        if len(st.session_state.readings_history) > 10:
            st.session_state.readings_history = st.session_state.readings_history[-10:]
        st.success(f"✅ Trend Added! Total: {len(st.session_state.readings_history)}")

with col3:
    if st.button("🔄 RESET ALL", use_container_width=True, key="reset_readings"):
        st.session_state.readings_history = []
        st.session_state.prediction_history = []
        st.rerun()

# ============================================================
# PRIMARY RISK RESULT - MOST VISUALLY IMPORTANT
# ============================================================

if len(st.session_state.readings_history) < 2:
    st.info("📊 Minimum 2 readings required for infection risk prediction")
else:
    X_features = feature_engine.create_features_from_readings(st.session_state.readings_history)
    
    if X_features is not None:
        probability = predict_risk(X_features)
        
        if probability is not None:
            prediction_record = {
                "timestamp": datetime.now(),
                "probability": probability,
                "n_readings": len(st.session_state.readings_history)
            }
            st.session_state.prediction_history.append(prediction_record)
            
            risk_data = RiskAssessment.get_risk_details(probability)
            risk_level_key = risk_data["risk_level"].split()[0]
            
            # Risk color mapping
            risk_color_map = {
                "HIGH": "var(--risk-high)",
                "MONITOR": "var(--risk-monitor)",
                "LOW": "var(--risk-low)"
            }
            risk_color = risk_color_map.get(risk_level_key, "var(--accent-cyan)")
            
            # Primary Risk Result Display
            st.markdown(f"""
            <div class="risk-result-card">
                <div class="risk-label">FUTURE INFECTION RISK</div>
                <div class="risk-probability-value" style="color: {risk_color};">{risk_data['probability_pct']}</div>
                <div class="risk-level-text" style="color: {risk_color};">{risk_data['risk_level']}</div>
                <p style="color: var(--text-secondary); margin-top: 1.5rem;">{risk_data['trend']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Recommended Action
            st.markdown(f"""
            <div style="background-color: rgba(0, 217, 255, 0.08); border-left: 4px solid var(--accent-cyan); padding: 1.5rem; border-radius: 10px;">
                <div style="color: var(--accent-cyan); font-weight: 600; margin-bottom: 0.5rem;">⚡ RECOMMENDED ACTION</div>
                <div style="color: var(--text-primary);">{risk_data['action']}</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# BIOMARKER STATUS - 2x3 GRID
# ============================================================

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="section-title">BIOMARKER STATUS</div>
<div class="section-subtitle">Six-channel temporal wound sensor monitoring</div>
""", unsafe_allow_html=True)

if st.session_state.readings_history:
    # Get latest reading
    latest = st.session_state.readings_history[-1]
    
    # Calculate velocities
    if len(st.session_state.readings_history) >= 2:
        prev = st.session_state.readings_history[-2]
        velocities = {bm: (latest[bm] - prev[bm]) for bm in biomarkers}
    else:
        velocities = {bm: 0 for bm in biomarkers}
    
    # 2x3 grid
    cols = st.columns(3)
    for idx, biomarker in enumerate(biomarkers):
        col_idx = idx % 3
        if col_idx == 0:
            cols = st.columns(3)
        
        with cols[col_idx]:
            vel = velocities[biomarker]
            vel_emoji = "▲" if vel > 0 else "▼" if vel < 0 else "→"
            vel_color = "var(--risk-high)" if vel > 0.1 else "var(--risk-low)" if vel < -0.1 else "var(--text-secondary)"
            
            st.markdown(f"""
            <div class="biomarker-card">
                <div class="biomarker-name">{biomarker.upper()}</div>
                <div class="biomarker-value">{latest[biomarker]:.2f}</div>
                <div class="biomarker-unit">per unit</div>
                <div class="biomarker-velocity" style="color: {vel_color};">{vel_emoji} {vel:.3f} / step</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# TEMPORAL ANALYSIS
# ============================================================

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="section-title">TEMPORAL TRAJECTORY</div>
<div class="section-subtitle">Biomarker trends over time</div>
""", unsafe_allow_html=True)

if st.session_state.readings_history:
    readings_df = pd.DataFrame(st.session_state.readings_history)
    
    fig_timeline = px.line(
        readings_df,
        y=biomarkers,
        title=None,
        labels={value: value.upper() for value in biomarkers}
    )
    fig_timeline.update_layout(
        hovermode='x unified',
        height=400,
        paper_bgcolor='rgba(8, 13, 31, 0)',
        plot_bgcolor='rgba(21, 31, 51, 0.3)',
        font={'family': 'Arial', 'color': '#A8B4C5'},
        xaxis={'gridcolor': 'rgba(168, 180, 197, 0.1)'},
        yaxis={'gridcolor': 'rgba(168, 180, 197, 0.1)'},
        legend={'x': 1.05, 'y': 1}
    )
    
    st.plotly_chart(fig_timeline, use_container_width=True)
else:
    st.info("No readings recorded yet")

# ============================================================
# AI EXPLANATION
# ============================================================

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="section-title">AI EXPLANATION</div>
<div class="section-subtitle">Why the model generated this risk estimate</div>
""", unsafe_allow_html=True)

if st.session_state.prediction_history:
    # Feature importance proxy (using actual features)
    feature_names = [
        f.replace("_", " ").title()
        for f in features
    ]

    importance_scores = model[-1].feature_importances_

    top_features = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance_scores
    }).sort_values("Importance", ascending=False).head(8)
    
    fig_importance = px.bar(
        top_features,
        y="Feature",
        x="Importance",
        orientation="h",
        color="Importance",
        color_continuous_scale=["#151F33", "#00D9FF", "#7C5CFF"],
        labels={"Importance": "Relative Importance"}
    )
    
    fig_importance.update_layout(
        paper_bgcolor='rgba(8, 13, 31, 0)',
        plot_bgcolor='rgba(21, 31, 51, 0.3)',
        font={'family': 'Arial', 'color': '#A8B4C5'},
        xaxis={'gridcolor': 'rgba(168, 180, 197, 0.1)'},
        yaxis={'gridcolor': 'rgba(168, 180, 197, 0.1)'},
        height=350,
        showlegend=False,
        margin=dict(l=200, r=20, t=20, b=50)
    )
    
    st.plotly_chart(fig_importance, use_container_width=True)
else:
    st.info("Generate predictions to see feature analysis")

# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="section-title">MODEL SPECIFICATIONS</div>
""", unsafe_allow_html=True)

spec_cols = st.columns(3)

with spec_cols[0]:
    st.markdown(f"""
    <div class="card">
        <div class="section-subtitle">Algorithm</div>
        <div style="font-size: 1.1rem; font-weight: 600; color: var(--accent-cyan);">Random Forest</div>
    </div>
    """, unsafe_allow_html=True)

with spec_cols[1]:
    st.markdown(f"""
    <div class="card">
        <div class="section-subtitle">Input Features</div>
        <div style="font-size: 1.1rem; font-weight: 600; color: var(--accent-cyan);">24</div>
    </div>
    """, unsafe_allow_html=True)

with spec_cols[2]:
    st.markdown(f"""
    <div class="card">
        <div class="section-subtitle">Test ROC-AUC</div>
        <div style="font-size: 1.1rem; font-weight: 600; color: var(--accent-cyan);">{TEST_METRICS.get("roc_auc", 0.0):.4f}</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# DISCLAIMER
# ============================================================

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="footer-premium">
    <strong style="color: var(--risk-high);">⚠️ IMPORTANT DISCLAIMER</strong><br>
    This is a simulation-based research prototype. NOT clinically validated. For research & development use only. Requires external validation and regulatory approval before clinical deployment.
    <div style="margin-top: 1rem; font-size: 0.8rem;">
        Infection Detector AI V2.1 — Frozen Random Forest | 24 Temporal Features | Last Updated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """
    </div>
</div>
""", unsafe_allow_html=True)
