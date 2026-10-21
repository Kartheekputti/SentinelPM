"""Reusable SHAP explainability service for SentinelPM.

Supports tree models (Random Forest, XGBoost) and linear models (Logistic Regression).
Explicitly disclaims physical causality and describes attributions as model behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import shap
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.core.exceptions import ExplainabilityError
from app.core.logging import get_logger
from app.features.definitions import FEATURE_NAME_MAP
from app.features.engineering import prepare_feature_frame

logger = get_logger(__name__)

CAUSALITY_DISCLAIMER: str = (
    "SHAP values describe statistical model attribution and feature contributions "
    "to the model's decision; they do not prove physical or mechanical causation."
)

# Reverse map: sanitised model feature names -> human-readable display names with units
REVERSE_FEATURE_MAP: dict[str, str] = {
    sanitised: human for human, sanitised in FEATURE_NAME_MAP.items()
}
REVERSE_FEATURE_MAP.update(
    {
        "type_h": "Product Type: High",
        "type_l": "Product Type: Low",
        "type_m": "Product Type: Medium",
    }
)


@dataclass(frozen=True)
class FeatureContribution:
    """Individual feature contribution to a model prediction."""

    feature_name: str
    display_name: str
    feature_value: float
    contribution: float
    direction: Literal["increases_risk", "decreases_risk", "neutral"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "display_name": self.display_name,
            "feature_value": round(float(self.feature_value), 4),
            "contribution": round(float(self.contribution), 4),
            "direction": self.direction,
        }


@dataclass
class ExplanationResult:
    """Complete prediction explanation outcome."""

    prediction_id: str | None
    model_name: str
    model_version: str
    predicted_probability: float
    base_value: float
    contributions: list[FeatureContribution]
    top_risk_drivers: list[FeatureContribution]
    top_risk_mitigators: list[FeatureContribution]
    summary_text: str
    causality_disclaimer: str = CAUSALITY_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "predicted_probability": round(float(self.predicted_probability), 4),
            "base_value": round(float(self.base_value), 4),
            "contributions": [c.to_dict() for c in self.contributions],
            "top_risk_drivers": [c.to_dict() for c in self.top_risk_drivers],
            "top_risk_mitigators": [c.to_dict() for c in self.top_risk_mitigators],
            "summary_text": self.summary_text,
            "causality_disclaimer": self.causality_disclaimer,
        }


class ModelExplainer:
    """Reusable SHAP explainer for scikit-learn pipelines."""

    def __init__(
        self,
        pipeline: Pipeline,
        *,
        model_name: str = "model",
        model_version: str = "1.0.0",
        background_data: pd.DataFrame | None = None,
    ) -> None:
        """Initialize the explainer for a fitted pipeline.

        Args:
            pipeline: Fitted scikit-learn Pipeline with steps ``features`` and ``model``.
            model_name: Identifier name of the model.
            model_version: Version tag of the model.
            background_data: Optional raw background dataset used for linear/kernel explainers.

        Raises:
            ExplainabilityError: If the pipeline is missing steps or uses an unsupported estimator.
        """
        self.pipeline = pipeline
        self.model_name = model_name
        self.model_version = model_version

        if not hasattr(pipeline, "named_steps"):
            msg = "Pipeline must be a valid scikit-learn Pipeline with named_steps"
            raise ExplainabilityError(msg)

        if "features" not in pipeline.named_steps or "model" not in pipeline.named_steps:
            msg = "Pipeline must contain 'features' and 'model' steps"
            raise ExplainabilityError(msg)

        self.features_step = pipeline.named_steps["features"]
        self.estimator = pipeline.named_steps["model"]

        if isinstance(self.estimator, DummyClassifier):
            msg = "Explanations are not supported for baseline DummyClassifier"
            raise ExplainabilityError(msg)

        self.feature_names: list[str] = [
            str(name) for name in self.features_step.get_feature_names_out()
        ]

        self._explainer = self._init_explainer(background_data)

    def _init_explainer(self, background_data: pd.DataFrame | None) -> Any:
        try:
            # Tree-based estimators
            if isinstance(self.estimator, (RandomForestClassifier,)):
                return shap.TreeExplainer(self.estimator)

            # Check XGBoost
            estimator_module = self.estimator.__class__.__module__
            if "xgboost" in estimator_module:
                return shap.TreeExplainer(self.estimator)

            # Linear models
            if isinstance(self.estimator, (LogisticRegression,)):
                if background_data is not None:
                    transformed_bg = self.transform_features(background_data)
                    masker = shap.maskers.Independent(transformed_bg)
                else:
                    # Synthetic neutral background
                    dummy_bg = pd.DataFrame(
                        np.zeros((10, len(self.feature_names))),
                        columns=self.feature_names,
                    )
                    masker = shap.maskers.Independent(dummy_bg)
                return shap.LinearExplainer(self.estimator, masker=masker)

            # General fallback
            if background_data is not None:
                transformed_bg = self.transform_features(background_data)
                return shap.Explainer(self.estimator, transformed_bg)

            return shap.Explainer(self.estimator)
        except Exception as exc:
            msg = f"Failed to initialize SHAP explainer for {self.estimator.__class__.__name__}: {exc}"
            raise ExplainabilityError(msg) from exc

    def transform_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the preprocessing feature pipeline to raw inputs."""
        valid_raw = prepare_feature_frame(frame)
        transformed = self.features_step.transform(valid_raw)
        if not isinstance(transformed, pd.DataFrame):
            transformed = pd.DataFrame(transformed, columns=self.feature_names)
        return transformed

    def _extract_shap_values(self, transformed_x: pd.DataFrame) -> tuple[np.ndarray, float]:
        """Compute SHAP values and base value for class 1 (failure)."""
        try:
            raw_shap = self._explainer.shap_values(transformed_x)
        except Exception as exc:
            msg = f"Failed to compute SHAP values: {exc}"
            raise ExplainabilityError(msg) from exc

        # Handle different SHAP output formats across versions and models
        if isinstance(raw_shap, list):
            # scikit-learn binary classifiers often return [class_0_vals, class_1_vals]
            values = np.asarray(raw_shap[1])
        elif isinstance(raw_shap, np.ndarray):
            values = raw_shap[:, :, 1] if raw_shap.ndim == 3 else raw_shap
        else:
            values = np.asarray(raw_shap)

        # Expected base value
        expected = getattr(self._explainer, "expected_value", 0.0)
        if isinstance(expected, (list, np.ndarray)):
            base_val = float(expected[1] if len(expected) > 1 else expected[0])
        else:
            base_val = float(expected)

        return values, base_val

    def explain_record(
        self,
        record: pd.DataFrame | dict[str, Any],
        *,
        prediction_id: str | None = None,
        top_n: int = 3,
    ) -> ExplanationResult:
        """Explain a single machine sensor observation.

        Args:
            record: Single record as a DataFrame or dictionary containing raw features.
            prediction_id: Optional correlation/tracing identifier.
            top_n: Number of top driving and mitigating factors to highlight.

        Returns:
            An :class:`ExplanationResult` containing feature contributions and narrative summary.

        Raises:
            ExplainabilityError: If feature transformation or explanation fails.
        """
        if isinstance(record, dict):
            frame = pd.DataFrame([record])
        elif isinstance(record, pd.DataFrame):
            frame = record.copy()
        else:
            msg = f"Record must be a dict or DataFrame, got {type(record).__name__}"
            raise ExplainabilityError(msg)

        if len(frame) == 0:
            msg = "Cannot explain an empty record"
            raise ExplainabilityError(msg)

        # Ensure single row
        single_row = frame.iloc[[0]]
        transformed_x = self.transform_features(single_row)

        shap_values, base_val = self._extract_shap_values(transformed_x)
        row_shap = shap_values[0]
        row_features = transformed_x.iloc[0]

        # Predicted probability of failure (class 1)
        prob = float(self.pipeline.predict_proba(single_row)[0, 1])

        contributions: list[FeatureContribution] = []
        for name in self.feature_names:
            shap_val = float(row_shap[self.feature_names.index(name)])
            feat_val = float(row_features[name])
            display_name = REVERSE_FEATURE_MAP.get(name, name.replace("_", " ").title())

            if shap_val > 1e-4:
                direction: Literal["increases_risk", "decreases_risk", "neutral"] = "increases_risk"
            elif shap_val < -1e-4:
                direction = "decreases_risk"
            else:
                direction = "neutral"

            contributions.append(
                FeatureContribution(
                    feature_name=name,
                    display_name=display_name,
                    feature_value=feat_val,
                    contribution=shap_val,
                    direction=direction,
                )
            )

        # Sort contributions by absolute impact
        contributions.sort(key=lambda c: abs(c.contribution), reverse=True)

        drivers = [c for c in contributions if c.direction == "increases_risk"]
        mitigators = [c for c in contributions if c.direction == "decreases_risk"]

        top_drivers = drivers[:top_n]
        top_mitigators = mitigators[:top_n]

        summary_text = self._build_narrative(prob, top_drivers, top_mitigators)

        return ExplanationResult(
            prediction_id=prediction_id,
            model_name=self.model_name,
            model_version=self.model_version,
            predicted_probability=prob,
            base_value=base_val,
            contributions=contributions,
            top_risk_drivers=top_drivers,
            top_risk_mitigators=top_mitigators,
            summary_text=summary_text,
            causality_disclaimer=CAUSALITY_DISCLAIMER,
        )

    def _build_narrative(
        self,
        prob: float,
        drivers: list[FeatureContribution],
        mitigators: list[FeatureContribution],
    ) -> str:
        """Construct a human-readable attribution summary."""
        lines = [f"Predicted machine failure probability is {prob:.1%}."]

        if drivers:
            driver_str = ", ".join(f"{d.display_name} (+{d.contribution:.3f})" for d in drivers)
            lines.append(f"Primary risk drivers: {driver_str}.")
        else:
            lines.append("No significant risk-increasing factors detected.")

        if mitigators:
            mitigator_str = ", ".join(
                f"{m.display_name} ({m.contribution:.3f})" for m in mitigators
            )
            lines.append(f"Mitigating factors: {mitigator_str}.")

        lines.append(f"Note: {CAUSALITY_DISCLAIMER}")
        return " ".join(lines)
