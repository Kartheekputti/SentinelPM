"""Single Machine Prediction Page — SentinelPM Industrial Predictive Maintenance Dashboard."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure SentinelPM project root is always at the head of sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DASHBOARD_DIR = Path(__file__).resolve().parents[1]
while str(DASHBOARD_DIR) in sys.path:
    sys.path.remove(str(DASHBOARD_DIR))

import streamlit as st

from app.core.config import get_settings
from app.dashboard.client import ApiClient
from app.dashboard.components import (
    render_api_status_banner,
    render_causality_disclaimer,
    render_contributions_chart,
    render_header,
    render_prediction_disclaimer,
    render_risk_badge,
    render_simulated_data_disclaimer,
)

st.set_page_config(page_title="Single Prediction — SentinelPM", page_icon="⚙️", layout="wide")

settings = get_settings()
api_url = os.getenv("SENTINELPM_API_URL", settings.api_url)
client = ApiClient(base_url=api_url)
is_connected = client.is_available()

render_header(
    "Single Machine Failure Prediction",
    "Interactive sensor telemetry risk scoring and real-time SHAP attribution",
)
render_simulated_data_disclaimer()
render_api_status_banner(is_connected, api_url)

if not is_connected:
    st.stop()

# Input parameter forms
st.subheader("Machine Operating Telemetry Input")

with st.form("single_prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        machine_type = st.selectbox(
            "Machine Quality Type",
            options=["L", "M", "H"],
            index=0,
            help="L: Low (50% of fleet), M: Medium (30%), H: High variant (20%)",
        )
        air_temp = st.number_input(
            "Air Temperature [K]",
            min_value=290.0,
            max_value=310.0,
            value=298.2,
            step=0.1,
            format="%.1f",
            help="Surrounding ambient temperature in Kelvin (standard range: 295-305 K)",
        )

    with col2:
        process_temp = st.number_input(
            "Process Temperature [K]",
            min_value=300.0,
            max_value=320.0,
            value=308.7,
            step=0.1,
            format="%.1f",
            help="Operating process temperature in Kelvin (typically 7-12 K above ambient)",
        )
        rot_speed = st.number_input(
            "Rotational Speed [rpm]",
            min_value=1000,
            max_value=3000,
            value=1408,
            step=10,
            help="Spindle rotational speed (standard range: 1200-2800 rpm)",
        )

    with col3:
        torque = st.number_input(
            "Torque [Nm]",
            min_value=0.0,
            max_value=100.0,
            value=46.3,
            step=0.5,
            format="%.1f",
            help="Spindle torque in Newton-meters (standard range: 10-80 Nm)",
        )
        tool_wear = st.number_input(
            "Tool Wear [min]",
            min_value=0,
            max_value=300,
            value=50,
            step=5,
            help="Accumulated cutting tool operational wear in minutes (tool failure often > 200 min)",
        )

    submit_button = st.form_submit_button(
        "Predict Failure Risk & Explain", use_container_width=True
    )

if submit_button:
    # 1. Client-side input validation
    if process_temp <= air_temp:
        st.warning(
            "⚠️ Note: Process temperature is normally higher than ambient air temperature under operation."
        )

    payload = {
        "Type": machine_type,
        "Air_temperature_K": float(air_temp),
        "Process_temperature_K": float(process_temp),
        "Rotational_speed_rpm": int(rot_speed),
        "Torque_Nm": float(torque),
        "Tool_wear_min": int(tool_wear),
    }

    with st.spinner("Executing model scoring and computing SHAP attributions..."):
        try:
            prediction_result = client.predict(payload)
            explanation_result = client.explain(payload, top_n=5)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

    st.markdown("---")
    st.subheader("Prediction Results & Risk Assessment")

    pred_failure = prediction_result.get("predicted_failure", False)
    prob = prediction_result.get("failure_probability", 0.0)
    risk_level = prediction_result.get("risk_level", "low")
    model_version = prediction_result.get("model_version", "1.0.0")

    res_col1, res_col2, res_col3, res_col4 = st.columns(4)

    with res_col1:
        st.markdown("**Status**")
        if pred_failure:
            st.error("🚨 FAILURE PREDICTED")
        else:
            st.success("✅ NORMAL OPERATION")

    with res_col2:
        st.markdown("**Failure Probability**")
        st.markdown(f"### {prob * 100:.1f}%")

    with res_col3:
        st.markdown("**Risk Tier**")
        st.markdown(render_risk_badge(risk_level), unsafe_allow_html=True)

    with res_col4:
        st.markdown("**Model Serving**")
        st.markdown(f"`sentinelpm:v{model_version}`")

    render_prediction_disclaimer()

    st.markdown("---")
    st.subheader("Feature Attributions (SHAP)")

    narrative = explanation_result.get("summary_text")
    if narrative:
        st.info(f"**Diagnostic Summary:** {narrative}", icon="🔍")

    contributions = explanation_result.get("contributions", [])
    render_contributions_chart(contributions)

    render_causality_disclaimer()
