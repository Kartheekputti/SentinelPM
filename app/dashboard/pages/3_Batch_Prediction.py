"""Batch Prediction Page — SentinelPM Industrial Predictive Maintenance Dashboard."""

from __future__ import annotations

import io
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
    render_header,
    render_prediction_disclaimer,
    render_simulated_data_disclaimer,
)

st.set_page_config(page_title="Batch Prediction — SentinelPM", page_icon="⚙️", layout="wide")

settings = get_settings()
api_url = os.getenv("SENTINELPM_API_URL", settings.api_url)
client = ApiClient(base_url=api_url)
is_connected = client.is_available()

render_header(
    "Batch Machine Telemetry Prediction",
    "Upload multi-machine CSV telemetry for batch scoring and risk report generation",
)
render_simulated_data_disclaimer()
render_api_status_banner(is_connected, api_url)

if not is_connected:
    st.stop()

# Instructions & download template
st.markdown(
    """
    Upload a CSV file containing machine operating observations.
    Required columns: `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`.
    *(Target labels such as `Machine failure` and failure indicators `TWF`, `HDF`, etc. are strictly forbidden and will be rejected).*
    """
)

# Provide sample template download
sample_df = pd.DataFrame(
    [
        {
            "Type": "L",
            "Air temperature [K]": 298.1,
            "Process temperature [K]": 308.6,
            "Rotational speed [rpm]": 1551,
            "Torque [Nm]": 42.8,
            "Tool wear [min]": 0,
        },
        {
            "Type": "M",
            "Air temperature [K]": 298.2,
            "Process temperature [K]": 308.7,
            "Rotational speed [rpm]": 1408,
            "Torque [Nm]": 46.3,
            "Tool wear [min]": 3,
        },
        {
            "Type": "H",
            "Air temperature [K]": 302.5,
            "Process temperature [K]": 311.2,
            "Rotational speed [rpm]": 1310,
            "Torque [Nm]": 64.2,
            "Tool wear [min]": 215,
        },
    ]
)
template_csv = sample_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Download Sample CSV Template",
    data=template_csv,
    file_name="sentinelpm_sample_telemetry.csv",
    mime="text/csv",
)

st.markdown("---")

uploaded_file = st.file_uploader(
    "Choose machine telemetry CSV file",
    type=["csv"],
    help=f"Maximum file size: {settings.max_upload_size_mb} MB. Maximum batch size: {settings.max_batch_size} records.",
)

if uploaded_file is not None:
    # Check file size
    if uploaded_file.size > settings.max_upload_size_mb * 1024 * 1024:
        st.error(
            f"Uploaded file exceeds maximum permitted limit ({settings.max_upload_size_mb} MB)."
        )
        st.stop()

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Failed to parse CSV file: {exc}")
        st.stop()

    st.write(f"**Loaded {len(df)} records.** Preview:")
    st.dataframe(df.head(5), use_container_width=True)

    # Standardize column naming if original UCI headers are present
    col_mapping = {
        "Air temperature [K]": "Air_temperature_K",
        "Process temperature [K]": "Process_temperature_K",
        "Rotational speed [rpm]": "Rotational_speed_rpm",
        "Torque [Nm]": "Torque_Nm",
        "Tool wear [min]": "Tool_wear_min",
    }
    df_renamed = df.rename(columns=col_mapping)

    required_cols = [
        "Type",
        "Air_temperature_K",
        "Process_temperature_K",
        "Rotational_speed_rpm",
        "Torque_Nm",
        "Tool_wear_min",
    ]

    missing = [c for c in required_cols if c not in df_renamed.columns]
    if missing:
        st.error(f"CSV is missing required sensor columns: {missing}")
        st.stop()

    # Reject target leakage if present
    forbidden_cols = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
    leakage = [c for c in forbidden_cols if c in df.columns]
    if leakage:
        st.error(
            f"🚨 Data Leakage Rejected: Input contains target label or failure indicators: {leakage}. "
            "Remove target columns before submitting."
        )
        st.stop()

    if len(df_renamed) > settings.max_batch_size:
        st.warning(
            f"Batch size ({len(df_renamed)}) exceeds configured limit ({settings.max_batch_size}). "
            f"Trimming to first {settings.max_batch_size} rows."
        )
        df_renamed = df_renamed.iloc[: settings.max_batch_size]

    if st.button("Run Batch Inference", type="primary"):
        records = df_renamed[required_cols].to_dict(orient="records")

        with st.spinner(f"Scoring {len(records)} machine observations..."):
            try:
                batch_response = client.predict_batch(records)
            except Exception as exc:
                st.error(f"Batch prediction failed: {exc}")
                st.stop()

        preds = batch_response.get("predictions", [])
        if not preds:
            st.warning("No predictions returned.")
            st.stop()

        # Combine predictions with original dataframe
        results_df = df_renamed.copy()
        results_df["predicted_failure"] = [p.get("predicted_failure", False) for p in preds]
        results_df["predicted_class"] = [1 if p.get("predicted_failure") else 0 for p in preds]
        results_df["failure_probability"] = [
            round(p.get("failure_probability", 0.0), 4) for p in preds
        ]
        results_df["risk_level"] = [p.get("risk_level") for p in preds]

        # Summary KPIs
        high_risk_count = (results_df["risk_level"] == "high").sum()
        medium_risk_count = (results_df["risk_level"] == "medium").sum()
        low_risk_count = (results_df["risk_level"] == "low").sum()

        st.success(f"Batch inference complete for {len(results_df)} records.")

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Total Analyzed", len(results_df))
        with kpi2:
            st.metric(
                "High Risk", int(high_risk_count), f"{high_risk_count / len(results_df) * 100:.1f}%"
            )
        with kpi3:
            st.metric("Medium Risk", int(medium_risk_count))
        with kpi4:
            st.metric("Low Risk", int(low_risk_count))

        # Filter and display
        st.markdown("### Scored Fleet Telemetry")
        risk_filter = st.multiselect(
            "Filter by Risk Level",
            options=["high", "medium", "low"],
            default=["high", "medium", "low"],
        )
        filtered_df = results_df[results_df["risk_level"].isin(risk_filter)]
        st.dataframe(filtered_df, use_container_width=True)

        # Download result CSV
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "📥 Download Batch Prediction Report (CSV)",
            data=csv_buffer.getvalue().encode("utf-8"),
            file_name="sentinelpm_batch_predictions.csv",
            mime="text/csv",
            type="primary",
        )

        render_prediction_disclaimer()
