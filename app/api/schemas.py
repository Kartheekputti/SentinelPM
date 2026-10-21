"""API request and response schemas for SentinelPM REST backend."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.explainability.explainer import FeatureContribution
from app.inference.schemas import PredictionResult, SensorRecord


class ErrorDetail(BaseModel):
    """Structured error details."""

    code: str
    message: str
    details: Any | None = None
    request_id: str | None = None


class ErrorResponse(BaseModel):
    """Standardized error envelope across all endpoints."""

    error: ErrorDetail


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy"] = "healthy"
    service: str = "sentinelpm-api"
    timestamp: str


class ReadyResponse(BaseModel):
    """Readiness probe response indicating model availability."""

    status: Literal["ready", "not_ready"]
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None


class ModelInfoResponse(BaseModel):
    """Non-sensitive model metadata and threshold configuration."""

    model_name: str
    model_version: str
    primary_metric: str
    cv_score: float | None
    decision_threshold: float
    risk_threshold_low: float
    feature_names: list[str]
    created_at: str


class BatchPredictionRequest(BaseModel):
    """Batch prediction request payload."""

    records: list[SensorRecord] = Field(
        ...,
        min_length=1,
        description="Collection of sensor telemetry observations.",
    )


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    count: int
    model_version: str
    predictions: list[PredictionResult]


class ExplainRequest(BaseModel):
    """Payload for SHAP attribution explanation."""

    record: SensorRecord
    prediction_id: str | None = None
    top_n: int = Field(default=3, ge=1, le=10)


class ExplainResponse(BaseModel):
    """SHAP explanation response with attribution breakdown and narrative summary."""

    prediction_id: str | None
    model_name: str
    model_version: str
    predicted_probability: float
    base_value: float
    contributions: list[FeatureContribution]
    top_risk_drivers: list[FeatureContribution]
    top_risk_mitigators: list[FeatureContribution]
    summary_text: str
    causality_disclaimer: str


class MonitoringSummaryResponse(BaseModel):
    """Unified telemetry, drift detection and alert status response."""

    telemetry: dict[str, Any]
    drift_report: dict[str, Any] | None = None
    active_alerts: list[dict[str, Any]]
