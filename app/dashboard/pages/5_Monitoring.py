"""Monitoring Page — SentinelPM Industrial Predictive Maintenance Dashboard."""

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
    render_drift_disclaimer,
    render_header,
    render_simulated_data_disclaimer,
)

st.set_page_config(page_title="Monitoring & Drift — SentinelPM", page_icon="⚙️", layout="wide")

settings = get_settings()
api_url = os.getenv("SENTINELPM_API_URL", settings.api_url)
client = ApiClient(base_url=api_url)
is_connected = client.is_available()

render_header(
    "Operational Monitoring & Data Drift",
    "Track real-time request SLA, sensor distribution shift (KS-test / PSI), and active alerts",
)
render_simulated_data_disclaimer()
render_api_status_banner(is_connected, api_url)

if not is_connected:
    st.stop()

try:
    monitoring = client.get_monitoring_summary()
except Exception as exc:
    st.error(f"Failed to query monitoring summary from API: {exc}")
    st.stop()

telemetry = monitoring.get("telemetry", {})
drift = monitoring.get("drift", {})
alerts = monitoring.get("alerts", {})

# Section 1: SLA and Latency Metrics
st.subheader("1. API Performance & Serving SLA")
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

total_reqs = telemetry.get("total_requests", 0)
err_rate = telemetry.get("error_rate", 0.0)
latencies = telemetry.get("latencies_ms", {})
p50 = latencies.get("p50", 0.0)
p95 = latencies.get("p95", 0.0)
p99 = latencies.get("p99", 0.0)

with m_col1:
    st.metric("Total Requests", f"{total_reqs:,}")
with m_col2:
    st.metric(
        "Error Rate",
        f"{err_rate * 100:.2f}%",
        delta="Normal" if err_rate < 0.05 else "Elevated",
        delta_color="inverse",
    )
with m_col3:
    st.metric("Latency p50", f"{p50:.1f} ms")
with m_col4:
    st.metric(
        "Latency p95",
        f"{p95:.1f} ms",
        delta="Within SLA" if p95 < 250 else "SLA Breach",
        delta_color="inverse",
    )

st.markdown("---")

# Section 2: Sensor Data Drift Detection
st.subheader("2. Sensor Telemetry Drift Analysis")
render_drift_disclaimer()

d_col1, d_col2 = st.columns(2)

with d_col1:
    st.markdown("##### Continuous Sensor Drift (Two-Sample Kolmogorov-Smirnov Test)")
    ks_results = drift.get("ks_tests", {})
    if ks_results:
        ks_rows = []
        for sensor, details in ks_results.items():
            ks_rows.append(
                {
                    "Sensor Feature": sensor,
                    "p-value": round(details.get("p_value", 1.0), 4),
                    "Statistic": round(details.get("statistic", 0.0), 4),
                    "Drift Status": "🚨 DRIFTED" if details.get("drift_detected") else "STABLE",
                }
            )
        ks_df = pd.DataFrame(ks_rows)
        st.dataframe(ks_df, use_container_width=True)
    else:
        st.info("No KS drift data recorded yet. Ensure baseline reference data is initialized.")

with d_col2:
    st.markdown("##### Categorical Machinery Drift (Population Stability Index)")
    psi_results = drift.get("psi_tests", {})
    if psi_results:
        psi_rows = []
        for feat, details in psi_results.items():
            psi_val = details.get("psi_value", 0.0)
            psi_rows.append(
                {
                    "Feature": feat,
                    "PSI Value": round(psi_val, 4),
                    "Status": "🚨 Significant Shift"
                    if psi_val > 0.25
                    else ("⚠️ Moderate Shift" if psi_val > 0.10 else "Stable"),
                }
            )
        st.dataframe(pd.DataFrame(psi_rows), use_container_width=True)
    else:
        st.info("No PSI drift data recorded yet.")

    st.markdown(
        """
        **PSI Benchmarks:**
        - $\\text{PSI} < 0.10$: Minimal shift, distribution stable.
        - $0.10 \\le \\text{PSI} < 0.25$: Moderate shift; investigate sensor recalibration.
        - $\\text{PSI} \\ge 0.25$: Significant shift; model retraining recommended.
        """
    )

st.markdown("---")

# Section 3: Alert History & Rule Evaluations
st.subheader("3. Alert Notifications & SLA Governance")
active_alerts = alerts.get("active_alerts", [])
recent_history = alerts.get("recent_history", [])

if active_alerts:
    st.error(f"🚨 {len(active_alerts)} Active Alerts Triggered:")
    for alert in active_alerts:
        st.markdown(
            f"- **[{alert.get('severity', 'WARNING').upper()}]** `{alert.get('name')}`: {alert.get('message')} *(Cooldown active)*"
        )
else:
    st.success("✅ All operational alert checks passing. No active alerts.")

if recent_history:
    st.markdown("##### Recent Alert Event Log")
    hist_df = pd.DataFrame(recent_history)
    st.dataframe(hist_df, use_container_width=True)
