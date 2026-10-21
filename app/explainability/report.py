"""Global feature importance and explanation reporting for SentinelPM.

Aggregates SHAP attributions across a dataset to evaluate global feature importance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.exceptions import ExplainabilityError
from app.core.logging import get_logger
from app.explainability.explainer import (
    CAUSALITY_DISCLAIMER,
    REVERSE_FEATURE_MAP,
    ModelExplainer,
)
from app.training.artifacts import write_json

logger = get_logger(__name__)


@dataclass
class GlobalExplanation:
    """Global feature attribution summary across an evaluation dataset."""

    model_name: str
    model_version: str
    n_samples: int
    feature_importance: dict[str, float]
    ranking: list[dict[str, Any]]
    summary_text: str
    causality_disclaimer: str = CAUSALITY_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "n_samples": self.n_samples,
            "feature_importance": self.feature_importance,
            "ranking": self.ranking,
            "summary_text": self.summary_text,
            "causality_disclaimer": self.causality_disclaimer,
        }


def generate_global_explanation(
    explainer: ModelExplainer,
    dataset: pd.DataFrame,
    *,
    max_samples: int = 200,
    random_state: int = 42,
) -> GlobalExplanation:
    """Compute global feature importance by averaging absolute SHAP values.

    Args:
        explainer: Initialized :class:`ModelExplainer`.
        dataset: Dataframe with raw features.
        max_samples: Maximum number of rows to evaluate to balance speed and stability.
        random_state: Subsampling seed.

    Returns:
        A :class:`GlobalExplanation` with feature importance ranking.

    Raises:
        ExplainabilityError: If the input dataset is empty or cannot be transformed.
    """
    if len(dataset) == 0:
        msg = "Cannot generate global explanation from an empty dataset"
        raise ExplainabilityError(msg)

    # Subsample if dataset is large
    if len(dataset) > max_samples:
        sample_df = dataset.sample(n=max_samples, random_state=random_state)
    else:
        sample_df = dataset.copy()

    transformed_x = explainer.transform_features(sample_df)
    shap_vals, _ = explainer._extract_shap_values(transformed_x)

    # Compute mean absolute SHAP value per feature
    mean_abs = np.mean(np.abs(shap_vals), axis=0)

    importance_map: dict[str, float] = {}
    ranking: list[dict[str, Any]] = []

    for name in explainer.feature_names:
        idx = explainer.feature_names.index(name)
        val = float(mean_abs[idx])
        importance_map[name] = round(val, 4)
        ranking.append(
            {
                "feature_name": name,
                "display_name": REVERSE_FEATURE_MAP.get(name, name),
                "mean_abs_shap": round(val, 4),
            }
        )

    ranking.sort(key=lambda r: r["mean_abs_shap"], reverse=True)

    top_features = [r["display_name"] for r in ranking[:3]]
    summary_text = (
        f"Global feature importance across {len(sample_df)} observations ranks "
        f"{', '.join(top_features)} as the most influential factors in predicting machine failure. "
        f"{CAUSALITY_DISCLAIMER}"
    )

    return GlobalExplanation(
        model_name=explainer.model_name,
        model_version=explainer.model_version,
        n_samples=len(sample_df),
        feature_importance=importance_map,
        ranking=ranking,
        summary_text=summary_text,
        causality_disclaimer=CAUSALITY_DISCLAIMER,
    )


def save_explanation_report(
    explanation: GlobalExplanation | dict[str, Any],
    path: Path | str,
) -> Path:
    """Save an explanation report to JSON.

    Args:
        explanation: :class:`GlobalExplanation` or dictionary.
        path: Destination path.

    Returns:
        The written Path.
    """
    payload = explanation.to_dict() if isinstance(explanation, GlobalExplanation) else explanation
    return write_json(payload, path)
