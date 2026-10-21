"""FastAPI route definitions for SentinelPM API."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, status

from app.api.dependencies import (
    get_alert_manager,
    get_drift_detector,
    get_inference_service,
    get_model_explainer,
    get_monitoring_tracker,
)
from app.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    ModelInfoResponse,
    MonitoringSummaryResponse,
    ReadyResponse,
)
from app.core.logging import get_logger
from app.explainability.explainer import ModelExplainer
from app.inference.schemas import PredictionResult, SensorRecord
from app.inference.service import InferenceService
from app.monitoring.alerts import AlertManager
from app.monitoring.drift import DriftDetector
from app.monitoring.tracker import MonitoringTracker

logger = get_logger(__name__)

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Confirm that the application process is running and responding.",
)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        service="sentinelpm-api",
        timestamp=datetime.now(UTC).isoformat(),
    )


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness probe",
    description="Report whether the approved model and dependencies are ready to serve predictions.",
)
def readiness_probe() -> ReadyResponse:
    try:
        service = get_inference_service()
        return ReadyResponse(
            status="ready",
            model_loaded=True,
            model_name=service.metadata.model_name,
            model_version=service.metadata.model_version,
        )
    except Exception as exc:
        logger.warning("Readiness probe check failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not_ready", "model_loaded": False, "message": str(exc)},
        ) from exc


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Model metadata",
    description="Retrieve non-sensitive metadata for the loaded approved model version.",
)
def model_info(
    service: Annotated[InferenceService, Depends(get_inference_service)],
) -> ModelInfoResponse:
    info = service.get_model_info()
    return ModelInfoResponse(
        model_name=info["model_name"],
        model_version=info["model_version"],
        primary_metric=info["primary_metric"],
        cv_score=info["cv_score"],
        decision_threshold=info["decision_threshold"],
        risk_threshold_low=info["risk_threshold_low"],
        feature_names=info["feature_names"],
        created_at=info["created_at"],
    )


@router.post(
    "/predict",
    response_model=PredictionResult,
    summary="Single machine failure prediction",
    description="Predict machine failure and determine risk level from a single sensor record.",
)
def predict_single(
    record: Annotated[SensorRecord, Body()],
    service: Annotated[InferenceService, Depends(get_inference_service)],
    tracker: Annotated[MonitoringTracker, Depends(get_monitoring_tracker)],
) -> PredictionResult:
    result = service.predict_single(record)

    # Record operational telemetry (non-blocking)
    tracker.record_model_usage(result.model_version)
    tracker.record_prediction(result.failure_probability, result.predicted_failure)
    tracker.record_sensor_telemetry(record.to_raw_dict())

    return result


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch machine failure predictions",
    description="Generate predictions for a batch of sensor records under configurable batch limit.",
)
def predict_batch(
    payload: Annotated[BatchPredictionRequest | list[SensorRecord], Body()],
    service: Annotated[InferenceService, Depends(get_inference_service)],
    tracker: Annotated[MonitoringTracker, Depends(get_monitoring_tracker)],
) -> BatchPredictionResponse:
    records = payload.records if isinstance(payload, BatchPredictionRequest) else payload
    results = service.predict_batch(records)

    # Record batch telemetry
    for r, rec in zip(results, records, strict=False):
        tracker.record_model_usage(r.model_version)
        tracker.record_prediction(r.failure_probability, r.predicted_failure)
        tracker.record_sensor_telemetry(rec.to_raw_dict())

    return BatchPredictionResponse(
        count=len(results),
        model_version=service.metadata.model_version,
        predictions=results,
    )


@router.post(
    "/explain",
    response_model=ExplainResponse,
    summary="Prediction attribution explanation",
    description="Compute SHAP feature contributions and narrative explanation for a sensor record.",
)
def explain_prediction(
    payload: Annotated[ExplainRequest, Body()],
    explainer: Annotated[ModelExplainer, Depends(get_model_explainer)],
) -> ExplainResponse:
    result = explainer.explain_record(
        payload.record.to_raw_dict(),
        prediction_id=payload.prediction_id,
        top_n=payload.top_n,
    )
    return ExplainResponse(
        prediction_id=result.prediction_id,
        model_name=result.model_name,
        model_version=result.model_version,
        predicted_probability=result.predicted_probability,
        base_value=result.base_value,
        contributions=result.contributions,
        top_risk_drivers=result.top_risk_drivers,
        top_risk_mitigators=result.top_risk_mitigators,
        summary_text=result.summary_text,
        causality_disclaimer=result.causality_disclaimer,
    )


@router.get(
    "/monitoring/summary",
    response_model=MonitoringSummaryResponse,
    summary="Monitoring and drift summary",
    description="Retrieve operational telemetry, recent latency percentiles, data drift status, and active alerts.",
)
def monitoring_summary(
    tracker: Annotated[MonitoringTracker, Depends(get_monitoring_tracker)],
    drift_detector: Annotated[DriftDetector, Depends(get_drift_detector)],
    alert_manager: Annotated[AlertManager, Depends(get_alert_manager)],
) -> MonitoringSummaryResponse:
    summary = tracker.get_summary()

    # Run drift check if sufficient sensor observations have been buffered
    sensor_df = tracker.get_buffered_sensor_frame()
    drift_report = None
    if len(sensor_df) >= 10:
        drift_report = drift_detector.evaluate_drift(
            sensor_df,
            current_failure_rate=summary.predictions.positive_rate,
        )

    # Evaluate alerts
    alert_manager.evaluate(tracker, drift_report)
    all_recent_alerts = alert_manager.get_recent_alerts()

    return MonitoringSummaryResponse(
        telemetry=summary.to_dict(),
        drift_report=drift_report.to_dict() if drift_report else None,
        active_alerts=[a.to_dict() for a in all_recent_alerts],
    )
