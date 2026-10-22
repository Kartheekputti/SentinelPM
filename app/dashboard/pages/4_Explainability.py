"""Explainability Page — SentinelPM Industrial Predictive Maintenance Dashboard."""

from __future__ import annotations

import json
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

import pandas as pd
import streamlit as st

from app.core.config import get_settings
from app.dashboard.client import ApiClient
from app.dashboard.components import (
    render_api_status_banner,
    render_causality_disclaimer,
    render_contributions_chart,
    render_header,
    render_simulated_data_disclaimer,
)

st.set_page_config(page_title="Explainability — SentinelPM", page_icon="⚙️", layout="wide")

settings = get_settings()
api_url = os.getenv("SENTINELPM_API_URL", settings.api_url)
client = ApiClient(base_url=api_url)
is_connected = client.is_available()

render_header(
    "Explainable AI & SHAP Attribution",
    "Deconstruct model failure predictions using cooperative game theory (SHAP)",
)
render_simulated_data_disclaimer()
render_api_status_banner(is_connected, api_url)

# Educational Overview Tab
tab1, tab2, tab3 = st.tabs(
    [
        "📊 Global Feature Importance",
        "🔍 Single Prediction Attribution",
        "📖 Methodology & Interpretation",
    ]
)

with tab1:
    st.subheader("Fleet-Wide Feature Importance (Mean Absolute SHAP)")
    st.markdown(
        "Global feature importance measures the average magnitude of impact each engineered feature "
        "exerts on the predicted failure probability across the evaluated dataset."
    )

    report_path = Path("artifacts/reports/shap_explanation_report.json")
    if report_path.exists():
        try:
            with open(report_path, encoding="utf-8") as f:
                report_data = json.load(f)

            ranking = report_data.get("ranking", [])
            importance_map = report_data.get("feature_importance", {})

            if ranking:
                importance_df = pd.DataFrame(
                    [
                        {"Feature": feat, "Mean |SHAP|": importance_map.get(feat, 0.0)}
                        for feat in ranking
                    ]
                )
                st.bar_chart(importance_df.set_index("Feature")["Mean |SHAP|"], horizontal=True)

                st.markdown(
                    f"**Evaluated on:** {report_data.get('n_samples', 'N/A')} holdout test samples."
                )
                if report_data.get("summary_text"):
                    st.info(f"**Key Finding:** {report_data['summary_text']}")
        except Exception as exc:
            st.warning(f"Could not load offline SHAP report: {exc}")
    else:
        st.info(
            "Offline SHAP evaluation report not found. Run `python -m scripts.run_explainability` to generate global rankings."
        )

with tab2:
    st.subheader("Interactive Individual Attribution")
    st.markdown(
        "Select an operational scenario or enter custom sensor measurements to obtain an on-demand "
        "SHAP attribution breakdown computed by the live FastAPI service."
    )

    if not is_connected:
        st.warning(
            "FastAPI backend is offline. Connect the API service to generate individual explanations."
        )
    else:
        preset = st.selectbox(
            "Select Scenario Preset",
            options=[
                "Custom",
                "High Tool Wear (Imminent Failure)",
                "Thermal Dissipation Stress",
                "Normal Operating Baseline",
            ],
        )

        if preset == "High Tool Wear (Imminent Failure)":
            def_type, def_air, def_proc, def_rpm, def_torque, def_wear = (
                "L",
                298.5,
                309.0,
                1400,
                50.0,
                240,
            )
        elif preset == "Thermal Dissipation Stress":
            def_type, def_air, def_proc, def_rpm, def_torque, def_wear = (
                "M",
                303.5,
                312.0,
                1320,
                62.0,
                110,
            )
        elif preset == "Normal Operating Baseline":
            def_type, def_air, def_proc, def_rpm, def_torque, def_wear = (
                "H",
                298.1,
                308.6,
                1551,
                42.8,
                20,
            )
        else:
            def_type, def_air, def_proc, def_rpm, def_torque, def_wear = (
                "L",
                298.2,
                308.7,
                1408,
                46.3,
                50,
            )

        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            e_type = st.selectbox(
                "Machine Type", ["L", "M", "H"], index=["L", "M", "H"].index(def_type)
            )
            e_air = st.number_input("Air Temperature [K]", 290.0, 310.0, float(def_air), 0.1)
            e_proc = st.number_input("Process Temperature [K]", 300.0, 320.0, float(def_proc), 0.1)
        with exp_col2:
            e_rpm = st.number_input("Rotational Speed [rpm]", 1000, 3000, int(def_rpm), 10)
            e_torque = st.number_input("Torque [Nm]", 0.0, 100.0, float(def_torque), 0.5)
            e_wear = st.number_input("Tool Wear [min]", 0, 300, int(def_wear), 5)

        if st.button("Generate SHAP Attribution", type="primary"):
            payload = {
                "Type": e_type,
                "Air_temperature_K": float(e_air),
                "Process_temperature_K": float(e_proc),
                "Rotational_speed_rpm": int(e_rpm),
                "Torque_Nm": float(e_torque),
                "Tool_wear_min": int(e_wear),
            }

            with st.spinner("Computing TreeExplainer attributions..."):
                try:
                    exp_data = client.explain(payload, top_n=5)
                except Exception as exc:
                    st.error(f"Failed to generate explanation: {exc}")
                    st.stop()

            st.markdown("---")
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric(
                    "Predicted Failure Probability",
                    f"{exp_data.get('predicted_probability', 0.0) * 100:.1f}%",
                )
            with m_col2:
                st.metric(
                    "Base Rate (Fleet Average)", f"{exp_data.get('base_value', 0.0) * 100:.1f}%"
                )
            with m_col3:
                diff = (
                    exp_data.get("predicted_probability", 0.0) - exp_data.get("base_value", 0.0)
                ) * 100
                st.metric("Attribution Shift", f"{diff:+.1f}%")

            if exp_data.get("summary_text"):
                st.info(f"**Narrative:** {exp_data['summary_text']}")

            render_contributions_chart(exp_data.get("contributions", []))

with tab3:
    st.subheader("How to Interpret SHAP Attributions")
    st.markdown(
        """
        ### What is SHAP?
        SHAP (*SHapley Additive exPlanations*) is based on cooperative game theory. It decomposes
        a machine learning model's output into additive contributions from each input feature:

        $$\\text{Prediction} = \\text{Base Value} + \\sum_{i=1}^{M} \\phi_i$$

        - **Base Value (Expected Value):** The average predicted failure rate across the background training data (~3.4%).
        - **Feature Contribution ($\\phi_i$):** The directional push that a specific sensor value imparts:
          - **Positive ($\\phi_i > 0$):** Increases the failure probability (risk driver).
          - **Negative ($\\phi_i < 0$):** Decreases the failure probability (stabilizing factor / risk mitigator).

        ### Engineered Interaction Features
        SentinelPM engineers physically meaningful interaction features:
        - **`temp_diff` ($T_{\\text{process}} - T_{\\text{air}}$):** Quantifies convective thermal dissipation. Low difference indicates heat accumulation.
        - **`mechanical_power` ($\\text{Torque} \\times \\text{RPM}$):** Proportional to mechanical power load transferred across the spindle.
        - **`temp_ratio` ($T_{\\text{process}} / T_{\\text{air}}$):** Relative thermal gradient.
        - **`strain` ($\\text{Torque} \\times \\text{Tool Wear}$):** Cumulative mechanical stress on the cutting insert.
        """
    )
    render_causality_disclaimer()
