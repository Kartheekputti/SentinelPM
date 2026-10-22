"""SentinelPM — Industrial Predictive Maintenance & Explainable ML Dashboard.

Main entry point and overview.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure SentinelPM project root is always at the head of sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DASHBOARD_DIR = Path(__file__).resolve().parent
while str(DASHBOARD_DIR) in sys.path:
    sys.path.remove(str(DASHBOARD_DIR))

import streamlit as st

from app.core.config import get_settings
from app.dashboard.client import ApiClient
from app.dashboard.components import (
    render_api_status_banner,
    render_header,
    render_simulated_data_disclaimer,
)

# Page configuration
st.set_page_config(
    page_title="SentinelPM — Industrial Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize settings and API client
settings = get_settings()
api_url = os.getenv("SENTINELPM_API_URL", settings.api_url)
client = ApiClient(base_url=api_url)
is_connected = client.is_available()

# Sidebar branding & connectivity
st.sidebar.title("⚙️ SentinelPM")
st.sidebar.caption("Industrial Predictive Maintenance Platform")
st.sidebar.markdown("---")
render_api_status_banner(is_connected, api_url)
st.sidebar.markdown("---")
st.sidebar.info(
    "Use the left navigation sidebar to navigate through pages:\n\n"
    "1. **Overview**: Operational health & KPIs\n"
    "2. **Single Prediction**: Real-time sensor scoring\n"
    "3. **Batch Prediction**: Multi-machine CSV upload\n"
    "4. **Explainability**: SHAP attribution insights\n"
    "5. **Monitoring**: Telemetry, drift & alerts\n"
    "6. **Model Info**: Version & evaluation metrics"
)

# Main content
render_header(
    "Industrial Predictive Maintenance Platform",
    "Real-time sensor failure risk scoring, SHAP explainability, and telemetry drift monitoring",
)

render_simulated_data_disclaimer()

# Service health check section
st.subheader("System Health & Operational Status")
col1, col2, col3, col4 = st.columns(4)

if is_connected:
    try:
        health_data = client.check_health()
        ready_data = client.check_readiness()
        model_info = client.get_model_info()

        with col1:
            st.metric(
                "Service Status", health_data.get("status", "healthy").upper(), delta="Active"
            )
        with col2:
            is_ready = ready_data.get("status") == "ready"
            st.metric("Model Readiness", "READY" if is_ready else "NOT READY", delta="Serving")
        with col3:
            st.metric("Active Model", model_info.get("model_name", "Unknown"))
        with col4:
            st.metric("Model Version", f"v{model_info.get('model_version', '1.0.0')}")

    except Exception as exc:
        st.warning(f"Connected to API, but encountered query warning: {exc}")
else:
    with col1:
        st.metric("Service Status", "OFFLINE", delta="-")
    with col2:
        st.metric("Model Readiness", "UNAVAILABLE", delta="-")
    with col3:
        st.metric("Active Model", "N/A")
    with col4:
        st.metric("Model Version", "N/A")

st.markdown("---")

# Quick Navigation Cards
st.subheader("Platform Capabilities")
q_col1, q_col2, q_col3 = st.columns(3)

with q_col1:
    st.markdown(
        """
        ### 🔍 Single Prediction
        Submit real-time sensor measurements (Type, temperatures, speed, torque, wear)
        to compute instantaneous failure risk and inspect primary physical risk drivers.
        """
    )

with q_col2:
    st.markdown(
        """
        ### 📁 Batch Prediction
        Upload machine telemetry batches in CSV format. Validate schemas, generate
        risk distribution reports, and export scored datasets.
        """
    )

with q_col3:
    st.markdown(
        """
        ### 📊 SHAP Explainability
        Examine global feature importance across operating regimes and review
        individual feature attribution waterfalls with causal caveats.
        """
    )

q_col4, q_col5, q_col6 = st.columns(3)

with q_col4:
    st.markdown(
        """
        ### 📈 Live Telemetry Monitoring
        Observe request latency percentiles (p50/p95/p99), API error rates,
        and Kolmogorov-Smirnov drift alerts on continuous sensor channels.
        """
    )

with q_col5:
    st.markdown(
        """
        ### 📋 Model Governance
        Audit active model hyperparameters, cross-validation metrics,
        decision thresholds, and MLflow experiment lineage.
        """
    )

with q_col6:
    st.markdown(
        """
        ### 🛡️ Operational Safety
        Review system limitations, false alarm vs. missed failure trade-offs,
        and human-in-the-loop engineering inspection guidelines.
        """
    )
