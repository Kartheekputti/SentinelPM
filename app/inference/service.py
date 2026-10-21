"""Inference service for SentinelPM.

Performs deterministic single and batch predictions with configurable risk-level scoring.
Strictly decoupled from training and HTTP routing.
"""

from __future__ import annotations

import uuid
from typing import Any

from sklearn.pipeline import Pipeline

from app.core.config import Settings, get_settings
from app.core.exceptions import InferenceError
from app.core.logging import get_logger
from app.features.engineering import prepare_feature_frame
from app.inference.schemas import (
    PredictionResult,
    RiskLevel,
    SensorRecord,
    records_to_dataframe,
)
from app.registry.manager import ModelMetadata, ModelRegistryManager

logger = get_logger(__name__)


class InferenceService:
    """Production inference service for industrial predictive maintenance."""

    def __init__(
        self,
        pipeline: Pipeline,
        metadata: ModelMetadata,
        *,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the inference service with a loaded model pipeline.

        Args:
            pipeline: Loaded fitted scikit-learn pipeline.
            metadata: Model metadata including version and threshold.
            settings: Optional configuration settings override.
        """
        self.pipeline = pipeline
        self.metadata = metadata
        self.settings = settings or get_settings()

        # Decision threshold priority: explicit settings override > model metadata
        self.threshold = float(
            self.settings.decision_threshold
            if self.settings.decision_threshold is not None
            else self.metadata.threshold
        )
        self.risk_threshold_low = float(self.settings.risk_threshold_low)
        self.max_batch_size = int(self.settings.max_batch_size)

        # Validate pipeline has predict_proba
        if not hasattr(self.pipeline, "predict_proba"):
            msg = "Loaded pipeline does not support probability estimation (predict_proba)"
            raise InferenceError(msg)

    @classmethod
    def from_registry(
        cls,
        *,
        model_name: str | None = None,
        model_version: str | None = None,
        settings: Settings | None = None,
    ) -> InferenceService:
        """Factory method to construct InferenceService from the approved registry."""
        active_settings = settings or get_settings()
        registry = ModelRegistryManager(settings=active_settings)
        pipeline, metadata = registry.load_approved_model(model_name, model_version)
        return cls(pipeline, metadata, settings=active_settings)

    def _determine_risk_level(self, probability: float) -> RiskLevel:
        """Determine categorical risk tier based on failure probability and configured thresholds.

        - 'high': probability >= decision_threshold (predicted failure = True)
        - 'medium': risk_threshold_low <= probability < decision_threshold (elevated risk)
        - 'low': probability < risk_threshold_low (nominal operation)
        """
        if probability >= self.threshold:
            return "high"
        if probability >= self.risk_threshold_low:
            return "medium"
        return "low"

    def predict_single(
        self,
        record: SensorRecord | dict[str, Any],
        *,
        prediction_id: str | None = None,
    ) -> PredictionResult:
        """Generate prediction and risk score for a single machine observation.

        Args:
            record: Validated :class:`SensorRecord` or dictionary matching the sensor schema.
            prediction_id: Optional correlation UUID.

        Returns:
            A populated :class:`PredictionResult`.

        Raises:
            InferenceError: If feature validation or prediction fails.
        """
        if isinstance(record, dict):
            try:
                record = SensorRecord.model_validate(record)
            except Exception as exc:
                msg = f"Failed to validate sensor input record: {exc}"
                raise InferenceError(msg) from exc

        raw_df = record.to_dataframe()
        valid_df = prepare_feature_frame(raw_df)

        try:
            probabilities = self.pipeline.predict_proba(valid_df)
            prob_failure = float(probabilities[0, 1])
        except Exception as exc:
            msg = f"Model execution failed during prediction: {exc}"
            raise InferenceError(msg) from exc

        predicted_failure = prob_failure >= self.threshold
        risk_level = self._determine_risk_level(prob_failure)

        return PredictionResult(
            prediction_id=prediction_id or str(uuid.uuid4()),
            model_version=self.metadata.model_version,
            model_name=self.metadata.model_name,
            predicted_failure=predicted_failure,
            failure_probability=round(prob_failure, 4),
            risk_level=risk_level,
            threshold=round(self.threshold, 4),
        )

    def predict_batch(
        self,
        records: list[SensorRecord | dict[str, Any]],
    ) -> list[PredictionResult]:
        """Generate predictions for a batch of sensor records.

        Args:
            records: Collection of sensor records.

        Returns:
            List of :class:`PredictionResult` objects, one per valid input record.

        Raises:
            InferenceError: If batch exceeds max_batch_size, is empty, or scoring fails.
        """
        if not records:
            msg = "Batch cannot be empty"
            raise InferenceError(msg)

        if len(records) > self.max_batch_size:
            msg = (
                f"Batch size {len(records)} exceeds maximum allowed limit of "
                f"{self.max_batch_size} records"
            )
            raise InferenceError(msg)

        validated_records: list[SensorRecord] = []
        for idx, item in enumerate(records):
            if isinstance(item, SensorRecord):
                validated_records.append(item)
            elif isinstance(item, dict):
                try:
                    validated_records.append(SensorRecord.model_validate(item))
                except Exception as exc:
                    msg = f"Invalid record at batch index {idx}: {exc}"
                    raise InferenceError(msg) from exc
            else:
                msg = f"Batch record at index {idx} has invalid type: {type(item).__name__}"
                raise InferenceError(msg)

        raw_df = records_to_dataframe(validated_records)
        valid_df = prepare_feature_frame(raw_df)

        try:
            probabilities = self.pipeline.predict_proba(valid_df)
            prob_failures = probabilities[:, 1].astype(float)
        except Exception as exc:
            msg = f"Batch model execution failed: {exc}"
            raise InferenceError(msg) from exc

        results: list[PredictionResult] = []
        for prob in prob_failures:
            p_val = float(prob)
            predicted_failure = p_val >= self.threshold
            risk = self._determine_risk_level(p_val)
            results.append(
                PredictionResult(
                    prediction_id=str(uuid.uuid4()),
                    model_version=self.metadata.model_version,
                    model_name=self.metadata.model_name,
                    predicted_failure=predicted_failure,
                    failure_probability=round(p_val, 4),
                    risk_level=risk,
                    threshold=round(self.threshold, 4),
                )
            )

        return results

    def get_model_info(self) -> dict[str, Any]:
        """Retrieve model metadata, threshold settings, and feature names."""
        return {
            "model_name": self.metadata.model_name,
            "model_version": self.metadata.model_version,
            "primary_metric": self.metadata.primary_metric,
            "cv_score": self.metadata.cv_score,
            "decision_threshold": self.threshold,
            "risk_threshold_low": self.risk_threshold_low,
            "feature_names": self.metadata.feature_names,
            "created_at": self.metadata.created_at,
        }
