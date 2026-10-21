"""Dependency injection providers for SentinelPM FastAPI backend."""

from __future__ import annotations

from app.core.config import get_settings
from app.explainability.explainer import ModelExplainer
from app.inference.service import InferenceService
from app.monitoring.alerts import AlertManager
from app.monitoring.drift import DriftDetector
from app.monitoring.reference import get_default_reference_data
from app.monitoring.tracker import MonitoringTracker

_inference_service: InferenceService | None = None
_model_explainer: ModelExplainer | None = None
_monitoring_tracker: MonitoringTracker | None = None
_drift_detector: DriftDetector | None = None
_alert_manager: AlertManager | None = None


def get_monitoring_tracker() -> MonitoringTracker:
    """Return the global monitoring tracker singleton."""
    global _monitoring_tracker
    if _monitoring_tracker is None:
        _monitoring_tracker = MonitoringTracker()
    return _monitoring_tracker


def set_monitoring_tracker(tracker: MonitoringTracker | None) -> None:
    """Override or reset the global monitoring tracker."""
    global _monitoring_tracker
    _monitoring_tracker = tracker


def get_inference_service() -> InferenceService:
    """Return the global inference service singleton or load from registry."""
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService.from_registry()
    return _inference_service


def set_inference_service(service: InferenceService | None) -> None:
    """Override or inject the inference service."""
    global _inference_service
    _inference_service = service


def get_model_explainer() -> ModelExplainer:
    """Return the global SHAP model explainer."""
    global _model_explainer
    if _model_explainer is None:
        service = get_inference_service()
        _model_explainer = ModelExplainer(
            service.pipeline,
            model_name=service.metadata.model_name,
            model_version=service.metadata.model_version,
        )
    return _model_explainer


def set_model_explainer(explainer: ModelExplainer | None) -> None:
    """Override or inject the model explainer."""
    global _model_explainer
    _model_explainer = explainer


def get_drift_detector() -> DriftDetector:
    """Return the drift detector singleton initialized with reference data."""
    global _drift_detector
    if _drift_detector is None:
        settings = get_settings()
        ref = get_default_reference_data(settings)
        _drift_detector = DriftDetector(
            ref,
            pvalue_threshold=settings.drift_pvalue_threshold,
        )
    return _drift_detector


def set_drift_detector(detector: DriftDetector | None) -> None:
    """Override or inject the drift detector."""
    global _drift_detector
    _drift_detector = detector


def get_alert_manager() -> AlertManager:
    """Return the alert manager singleton."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


def set_alert_manager(manager: AlertManager | None) -> None:
    """Override or inject the alert manager."""
    global _alert_manager
    _alert_manager = manager


def reset_dependencies() -> None:
    """Reset all dependency singletons to None (for testing isolation)."""
    set_inference_service(None)
    set_model_explainer(None)
    set_monitoring_tracker(None)
    set_drift_detector(None)
    set_alert_manager(None)
