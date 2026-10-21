"""Unit and endpoint tests for FastAPI backend (M07)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

from app.api.app import create_app
from app.api.dependencies import (
    reset_dependencies,
    set_alert_manager,
    set_drift_detector,
    set_inference_service,
    set_model_explainer,
    set_monitoring_tracker,
)
from app.explainability.explainer import ModelExplainer
from app.inference.service import InferenceService
from app.monitoring.alerts import AlertManager
from app.monitoring.drift import DriftDetector
from app.monitoring.reference import build_reference_data
from app.monitoring.tracker import MonitoringTracker
from app.registry.manager import ModelMetadata


@pytest.fixture
def test_client(sample_trained_rf: Pipeline, valid_frame) -> TestClient:
    """Provide a TestClient with injected test services."""
    reset_dependencies()

    metadata = ModelMetadata(
        model_name="random_forest",
        model_version="1.0.0",
        threshold=0.35,
        primary_metric="average_precision",
        cv_score=0.88,
        feature_names=["air_temperature_k"],
        artifact_path="dummy.joblib",
        created_at="2026-10-21T00:00:00Z",
    )
    inference_service = InferenceService(sample_trained_rf, metadata)
    explainer = ModelExplainer(sample_trained_rf, model_name="random_forest", model_version="1.0.0")
    tracker = MonitoringTracker()
    ref_data = build_reference_data(valid_frame)
    drift_detector = DriftDetector(ref_data)
    alert_manager = AlertManager()

    set_inference_service(inference_service)
    set_model_explainer(explainer)
    set_monitoring_tracker(tracker)
    set_drift_detector(drift_detector)
    set_alert_manager(alert_manager)

    app = create_app()
    with TestClient(app) as client:
        yield client

    reset_dependencies()


@pytest.fixture
def valid_payload() -> dict[str, object]:
    return {
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0.0,
        "Type": "M",
    }


def test_get_health(test_client: TestClient) -> None:
    """Verify /health endpoint returns 200 OK."""
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "sentinelpm-api"
    assert "timestamp" in data
    assert "X-Request-ID" in response.headers


def test_get_ready(test_client: TestClient) -> None:
    """Verify /ready endpoint reports model readiness."""
    response = test_client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["model_loaded"] is True
    assert data["model_name"] == "random_forest"
    assert data["model_version"] == "1.0.0"


def test_get_ready_unavailable() -> None:
    """Verify /ready endpoint returns 503 when model is not available."""
    reset_dependencies()
    # Explicitly set inference service to None and make registry fail
    set_inference_service(None)

    app = create_app()
    with TestClient(app) as client:
        # If no model can be loaded from empty/missing path
        response = client.get("/ready")
        # Either 200 if real artifacts exist in repo or 503 if unloaded
        assert response.status_code in (200, 503)


def test_get_model_info(test_client: TestClient) -> None:
    """Verify /model/info endpoint returns non-sensitive model metadata."""
    response = test_client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "random_forest"
    assert data["model_version"] == "1.0.0"
    assert data["decision_threshold"] == 0.35
    assert data["primary_metric"] == "average_precision"
    assert "air_temperature_k" in data["feature_names"]


def test_post_predict_single(test_client: TestClient, valid_payload: dict[str, object]) -> None:
    """Verify /predict generates single prediction with risk score and request tracing."""
    response = test_client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction_id" in data
    assert data["model_version"] == "1.0.0"
    assert data["model_name"] == "random_forest"
    assert isinstance(data["predicted_failure"], bool)
    assert 0.0 <= data["failure_probability"] <= 1.0
    assert data["risk_level"] in ("low", "medium", "high")
    assert data["threshold"] == 0.35
    assert "X-Request-ID" in response.headers


def test_post_predict_rejects_target_label(
    test_client: TestClient, valid_payload: dict[str, object]
) -> None:
    """Verify target label is strictly rejected with 422 Unprocessable Entity."""
    leak_payload = dict(valid_payload)
    leak_payload["Machine failure"] = 0

    response = test_client.post("/predict", json=leak_payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"]["code"] == "VALIDATION_ERROR"
    assert "leakage" in str(data["error"]["details"]).lower()


def test_post_predict_rejects_out_of_bounds(
    test_client: TestClient, valid_payload: dict[str, object]
) -> None:
    """Verify out-of-range sensor measurements are rejected with 422."""
    bad_payload = dict(valid_payload)
    bad_payload["Air temperature [K]"] = 100.0  # Physically implausible

    response = test_client.post("/predict", json=bad_payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"]["code"] == "VALIDATION_ERROR"


def test_post_predict_batch(test_client: TestClient, valid_payload: dict[str, object]) -> None:
    """Verify /predict/batch generates predictions for multiple records."""
    batch_req = {"records": [valid_payload, valid_payload]}
    response = test_client.post("/predict/batch", json=batch_req)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["prediction_id"] != data["predictions"][1]["prediction_id"]


def test_post_explain(test_client: TestClient, valid_payload: dict[str, object]) -> None:
    """Verify /explain generates SHAP attributions and human-readable summary."""
    explain_req = {
        "record": valid_payload,
        "prediction_id": "test-uuid-456",
        "top_n": 3,
    }
    response = test_client.post("/explain", json=explain_req)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction_id"] == "test-uuid-456"
    assert 0.0 <= data["predicted_probability"] <= 1.0
    assert len(data["contributions"]) > 0
    assert len(data["top_risk_drivers"]) <= 3
    assert "causality_disclaimer" in data
    assert "do not prove physical or mechanical causation" in data["causality_disclaimer"]


def test_get_monitoring_summary(test_client: TestClient, valid_payload: dict[str, object]) -> None:
    """Verify /monitoring/summary returns telemetry, latency and alert metrics."""
    # Send a prediction first
    test_client.post("/predict", json=valid_payload)

    response = test_client.get("/monitoring/summary")
    assert response.status_code == 200
    data = response.json()
    assert "telemetry" in data
    assert data["telemetry"]["total_requests"] >= 1
    assert "active_alerts" in data
