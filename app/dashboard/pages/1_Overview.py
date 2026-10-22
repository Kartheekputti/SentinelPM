"""Overview Page — SentinelPM Industrial Predictive Maintenance Dashboard."""

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

import pandas as pd
import streamlit as st

from app.core.config import get_settings
from app.dashboard.client import ApiClient
from app.dashboard.components import (
    render_api_status_banner,
    render_header,
    render_simulated_data_disclaimer,
)

st.set_page_config(page_title="Overview — SentinelPM", page_icon="⚙️", layout="wide")

settings = get_settings()
api_url = os.getenv("SENTINELPM_API_URL", settings.api_url)
client = ApiClient(base_url=api_url)
is_connected = client.is_available()

render_header(
    "System Overview & Fleet Health",
    "Aggregated predictive maintenance statistics and serving health",
)
render_simulated_data_disclaimer()
render_api_status_banner(is_connected, api_url)

if not is_connected:
    st.stop()

try:
    health = client.check_health()
    ready = client.check_readiness()
    model_info = client.get_model_info()
    monitoring = client.get_monitoring_summary()
except Exception as exc:
    st.error(f"Failed to fetch operational telemetry from API: {exc}")
    st.stop()

# Top KPI metrics
telemetry = monitoring.get("telemetry", {})
drift = monitoring.get("drift", {})
alerts = monitoring.get("alerts", {})

total_requests = telemetry.get("total_requests", 0)
risk_counts = telemetry.get("risk_counts", {"low": 0, "medium": 0, "high": 0})
high_risk_count = risk_counts.get("high", 0)
medium_risk_count = risk_counts.get("medium", 0)
low_risk_count = risk_counts.get("low", 0)

high_risk_pct = (high_risk_count / total_requests * 100) if total_requests > 0 else 0.0

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Records Analyzed", f"{total_requests:,}")
with col2:
    st.metric("High-Risk Alerts", f"{high_risk_count:,}", f"{high_risk_pct:.1f}% of total")
with col3:
    st.metric("Active Model Version", f"v{model_info.get('model_version', '1.0.0')}")
with col4:
    active_alerts_count = alerts.get("active_alerts_count", 0)
    st.metric(
        "Active System Alerts",
        active_alerts_count,
        delta="Normal" if active_alerts_count == 0 else "Action Required",
        delta_color="inverse",
    )

st.markdown("---")

# Fleet Risk Distribution & Latency
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Fleet Risk Breakdown")
    if total_requests > 0:
        risk_df = pd.DataFrame(
            {
                "Risk Tier": ["Low (<0.30)", "Medium (0.30-0.70)", "High (>=0.70)"],
                "Count": [low_risk_count, medium_risk_count, high_risk_count],
            }
        )
        st.bar_chart(risk_df.set_index("Risk Tier")["Count"])
    else:
        st.info(
            "No inference requests recorded yet. Submit single or batch predictions to populate telemetry."
        )

with col_right:
    st.subheader("API Serving Latency & SLA")
    latencies = telemetry.get("latencies_ms", {})
    p50 = latencies.get("p50", 0.0)
    p95 = latencies.get("p95", 0.0)
    p99 = latencies.get("p99", 0.0)
    error_rate = telemetry.get("error_rate", 0.0)

    l_col1, l_col2, l_col3 = st.columns(3)
    with l_col1:
        st.metric("Latency (p50)", f"{p50:.1f} ms")
    with l_col2:
        st.metric("Latency (p95)", f"{p95:.1f} ms")
    with l_col3:
        st.metric("Error Rate", f"{error_rate * 100:.2f}%")

    st.markdown("##### Model Governance & Readiness")
    st.json(
        {
            "service": health.get("service"),
            "status": health.get("status"),
            "model_loaded": ready.get("model_loaded"),
            "algorithm": model_info.get("model_name"),
            "decision_threshold": model_info.get("decision_threshold"),
            "primary_metric": model_info.get("primary_metric"),
            "cross_val_score": model_info.get("cv_score"),
        }
    )

st.markdown("---")

# Latest drift status summary
st.subheader("Sensor Telemetry Drift Status")
drifted_count = drift.get("drifted_features_count", 0)
if drift.get("drift_detected", False):
    st.warning(
        f"⚠️ Telemetry Drift Detected: {drifted_count} sensor channel(s) show distribution shifts "
        f"({', '.join(drift.get('drifted_features', []))}). Inspect Page 5 for details."
    )
else:
    st.success(
        "✅ All continuous and categorical sensor distributions are stable within baseline bounds."
    )
