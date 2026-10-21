"""Explainability module using SHAP for SentinelPM.

Provides individual prediction attributions and global feature importance
for trained model pipelines.
"""

from __future__ import annotations

from app.explainability.explainer import (
    ExplanationResult,
    FeatureContribution,
    ModelExplainer,
)
from app.explainability.report import (
    GlobalExplanation,
    generate_global_explanation,
    save_explanation_report,
)

__all__ = [
    "ExplanationResult",
    "FeatureContribution",
    "GlobalExplanation",
    "ModelExplainer",
    "generate_global_explanation",
    "save_explanation_report",
]
