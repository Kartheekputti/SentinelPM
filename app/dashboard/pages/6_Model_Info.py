"""Model Information Page — SentinelPM Industrial Predictive Maintenance Dashboard."""

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
    render_header,
    render_simulated_data_disclaimer,
)

st.set_page_config(page_title="Model Information — SentinelPM", page_icon="⚙️", layout="wide")

settings = get_settings()
api_url = os.getenv("SENTINELPM_API_URL", settings.api_url)
client = ApiClient(base_url=api_url)
is_connected = client.is_available()

render_header(
    "Model Governance & Specifications",
    "Inspect trained model metadata, cross-validation metrics, and feature schemas",
)
render_simulated_data_disclaimer()
render_api_status_banner(is_connected, api_url)

if not is_connected:
    st.stop()

try:
    model_info = client.get_model_info()
    ready = client.check_readiness()
except Exception as exc:
    st.error(f"Failed to query model specifications: {exc}")
    st.stop()

# Active Serving Model Cards
st.subheader("1. Active Serving Model")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Model Architecture", model_info.get("model_name", "Unknown").replace("_", " ").title()
    )
with col2:
    st.metric("Model Version", f"v{model_info.get('model_version', '1.0.0')}")
with col3:
    st.metric("Primary Evaluation Metric", model_info.get("primary_metric", "PR-AUC").upper())
with col4:
    cv_val = model_info.get("cv_score")
    st.metric("Cross-Validation Score", f"{cv_val:.4f}" if cv_val else "N/A")

st.markdown("---")

# Section 2: Model Comparison and Lineage
st.subheader("2. Candidate Model Evaluation & Benchmark")
st.markdown(
    "Comparison of evaluated baseline and candidate models on 5-fold stratified cross-validation. "
    "Primary selection metric is **PR-AUC (Average Precision)** due to extreme class imbalance (~3.4% failure rate)."
)

meta_path = Path("artifacts/models/metadata.json")
if meta_path.exists():
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta_data = json.load(f)

        models_list = meta_data.get("models", [])
        if models_list:
            sanitized_rows = []
            for m in models_list:
                sanitized_rows.append(
                    {
                        "Model Candidate": m.get("name", "Unknown").replace("_", " ").title(),
                        "PR-AUC (CV Score)": round(m.get("cv_score", 0.0), 4),
                        "Optimal Decision Threshold": round(m.get("threshold", 0.5), 2),
                        "Status": "⭐ Active Champion"
                        if m.get("name") == meta_data.get("best_model")
                        else "Candidate",
                    }
                )
            comp_df = pd.DataFrame(sanitized_rows)
            st.dataframe(comp_df, use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not load metadata summary: {exc}")

st.markdown("---")

# Section 3: Feature Schema & Operational Thresholds
st.subheader("3. Input Feature Specification & Operating Bounds")
f_col1, f_col2 = st.columns(2)

with f_col1:
    st.markdown("##### Post-Engineering Features Expected by Pipeline")
    feats = model_info.get("feature_names", [])
    if feats:
        feat_df = pd.DataFrame(
            {
                "Feature Name": feats,
                "Type": [
                    "Numeric" if "type" not in f.lower() else "One-Hot Categorical" for f in feats
                ],
            }
        )
        st.dataframe(feat_df, use_container_width=True)
    else:
        st.info("No feature schema details returned by API.")

with f_col2:
    st.markdown("##### Decision & Risk Tiers Configuration")
    st.json(
        {
            "active_model_name": model_info.get("model_name"),
            "decision_threshold": model_info.get("decision_threshold"),
            "risk_tier_low": f"< {model_info.get('risk_threshold_low', 0.20)}",
            "risk_tier_medium": f"{model_info.get('risk_threshold_low', 0.20)} - 0.70",
            "risk_tier_high": ">= 0.70",
            "memory_loaded": ready.get("model_loaded"),
            "registered_timestamp": model_info.get("created_at"),
        }
    )

st.markdown(
    """
    > **Security & Governance Guarantee:** Sensitive host credentials, secret tokens, and host filesystem absolute paths 
    > are withheld from all client responses and dashboards.
    """
)
