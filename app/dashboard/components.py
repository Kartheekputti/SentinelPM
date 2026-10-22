"""Reusable UI components and charts for the SentinelPM Streamlit dashboard."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def render_header(title: str, subtitle: str | None = None) -> None:
    """Render consistent page header with branding."""
    st.markdown(
        f"""
        <div style="padding-bottom: 1rem; border-bottom: 1px solid #e0e0e0; margin-bottom: 1.5rem;">
            <h1 style="margin: 0; color: #1e293b; font-weight: 700;">SentinelPM</h1>
            <h3 style="margin: 0.25rem 0 0 0; color: #475569; font-weight: 500;">{title}</h3>
            {f'<p style="margin: 0.25rem 0 0 0; color: #64748b; font-size: 0.95rem;">{subtitle}</p>' if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_simulated_data_disclaimer() -> None:
    """Render prominent notice regarding the simulated nature of the AI4I dataset."""
    st.info(
        "**Dataset Notice:** SentinelPM models are trained on the UCI AI4I 2020 Predictive "
        "Maintenance dataset. This is a **simulated synthetic dataset**, not live factory telemetry. "
        "Outputs demonstrate methodology and operational architecture; do not connect directly to machinery control systems."
    )


def render_prediction_disclaimer() -> None:
    """Render disclaimer that predictions do not guarantee physical machine state."""
    st.caption(
        "⚠️ **Operational Notice:** Predictions provide statistical failure probability and risk tiers. "
        "A prediction does not guarantee or rule out machine failure; physical engineering inspection is required before maintenance action."
    )


def render_causality_disclaimer() -> None:
    """Render standard SHAP statistical attribution disclaimer."""
    st.caption(
        "🔍 **Causality Notice:** SHAP values reflect statistical attribution within the model's learned representation; "
        "they do not establish physical causation or diagnostic certainty without engineering inspection."
    )


def render_drift_disclaimer() -> None:
    """Render standard drift interpretation disclaimer."""
    st.caption(
        "📊 **Monitoring Notice:** Data drift reflects statistical shifts between baseline and serving distributions "
        "(Kolmogorov-Smirnov test and PSI). Data drift indicates sensory changes, not proof of declining model accuracy."
    )


def render_risk_badge(risk_level: str) -> str:
    """Return styled HTML badge for risk tier."""
    level = risk_level.lower()
    if level == "high":
        color = "#ef4444"
        bg = "#fee2e2"
        text = "HIGH RISK"
    elif level == "medium":
        color = "#f59e0b"
        bg = "#fef3c7"
        text = "MEDIUM RISK"
    else:
        color = "#10b981"
        bg = "#d1fae5"
        text = "LOW RISK"

    return f'<span style="background-color: {bg}; color: {color}; padding: 0.25rem 0.6rem; border-radius: 9999px; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.05em;">{text}</span>'


def render_api_status_banner(is_connected: bool, base_url: str) -> None:
    """Display connection status to the FastAPI backend."""
    if is_connected:
        st.sidebar.success(f"● Connected to API ({base_url})", icon="✅")
    else:
        st.sidebar.error(f"✕ API Unreachable ({base_url})", icon="🚨")
        st.warning(
            f"**FastAPI backend is currently offline at `{base_url}`.**\n\n"
            "To start the API service, open a terminal and execute:\n"
            "```powershell\n"
            "python -m scripts.run_api\n"
            "```\n"
            "The dashboard will automatically reconnect once the service is live.",
            icon="⚠️",
        )


def render_contributions_chart(contributions: list[dict[str, Any]]) -> None:
    """Render a horizontal bar chart displaying SHAP feature attributions."""
    if not contributions:
        st.info("No feature contributions available to display.")
        return

    df = pd.DataFrame(contributions)
    # Ensure expected columns
    if "display_name" not in df.columns:
        df["display_name"] = df.get("feature", "unknown")
    if "contribution" not in df.columns:
        return

    df["color"] = df["contribution"].apply(lambda x: "#ef4444" if x > 0 else "#10b981")
    df = df.sort_values(by="contribution", ascending=True)

    st.markdown("##### Feature Impact on Failure Risk (SHAP Attribution)")
    st.bar_chart(
        df.set_index("display_name")["contribution"],
        color=None,
        horizontal=True,
    )
    st.caption("Positive values (red) increase failure risk; negative values (green) reduce risk.")
