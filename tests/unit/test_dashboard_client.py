"""Unit tests for the SentinelPM Dashboard ApiClient."""

from __future__ import annotations

import httpx
import pytest

from app.core.exceptions import ApiClientError
from app.dashboard.client import ApiClient


class MockTransport(httpx.BaseTransport):
    """Custom mock transport for testing ApiClient."""

    def __init__(self, handler) -> None:
        self.handler = handler

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self.handler(request)


def test_api_client_is_available_true() -> None:
    """Verify is_available returns True when /health responds 200."""

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"status": "healthy", "service": "sentinelpm-api", "timestamp": "now"}
        )

    transport = MockTransport(mock_handler)
    client = ApiClient(base_url="http://mock-api:8000", transport=transport)
    assert client.is_available() is True


def test_api_client_is_available_false() -> None:
    """Verify is_available returns False when connection fails."""

    def mock_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    transport = MockTransport(mock_handler)
    client = ApiClient(base_url="http://mock-api:8000", transport=transport)
    assert client.is_available() is False


def test_api_client_check_health_success() -> None:
    """Verify check_health parses successful 200 response."""

    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(
            200,
            json={
                "status": "healthy",
                "service": "sentinelpm-api",
                "timestamp": "2026-10-22T00:00:00Z",
            },
        )

    transport = MockTransport(mock_handler)
    client = ApiClient(base_url="http://mock-api:8000", transport=transport)
    res = client.check_health()
    assert res["status"] == "healthy"
    assert res["service"] == "sentinelpm-api"


def test_api_client_check_readiness_failure() -> None:
    """Verify check_readiness raises ApiClientError on HTTP 503."""

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"status": "not_ready", "model_loaded": False})

    transport = MockTransport(mock_handler)
    client = ApiClient(base_url="http://mock-api:8000", transport=transport)
    with pytest.raises(ApiClientError, match="API readiness check failed with status 503"):
        client.check_readiness()


def test_api_client_predict_success() -> None:
    """Verify predict returns formatted prediction payload."""

    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/predict"
        return httpx.Response(
            200,
            json={
                "predicted_class": 0,
                "failure_probability": 0.042,
                "risk_level": "low",
                "model_version": "1.0.0",
                "timestamp": "2026-10-22T00:00:00Z",
            },
        )

    transport = MockTransport(mock_handler)
    client = ApiClient(base_url="http://mock-api:8000", transport=transport)
    res = client.predict(
        {
            "Type": "M",
            "Air_temperature_K": 298.2,
            "Process_temperature_K": 308.7,
            "Rotational_speed_rpm": 1408,
            "Torque_Nm": 46.3,
            "Tool_wear_min": 3,
        }
    )
    assert res["predicted_class"] == 0
    assert res["risk_level"] == "low"
    assert res["failure_probability"] == 0.042


def test_api_client_predict_validation_error() -> None:
    """Verify predict raises ApiClientError with custom validation details on 422."""

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Target label 'Machine failure' is strictly forbidden as input.",
                }
            },
        )

    transport = MockTransport(mock_handler)
    client = ApiClient(base_url="http://mock-api:8000", transport=transport)
    with pytest.raises(
        ApiClientError,
        match="Validation error: Target label 'Machine failure' is strictly forbidden",
    ):
        client.predict({"Type": "M", "Machine failure": 1})


def test_api_client_predict_batch_success() -> None:
    """Verify predict_batch sends records list and parses response."""

    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/predict/batch"
        return httpx.Response(
            200,
            json={
                "count": 2,
                "model_version": "1.0.0",
                "predictions": [
                    {
                        "predicted_class": 0,
                        "failure_probability": 0.01,
                        "risk_level": "low",
                        "model_version": "1.0.0",
                    },
                    {
                        "predicted_class": 1,
                        "failure_probability": 0.85,
                        "risk_level": "high",
                        "model_version": "1.0.0",
                    },
                ],
            },
        )

    transport = MockTransport(mock_handler)
    client = ApiClient(base_url="http://mock-api:8000", transport=transport)
    records = [
        {
            "Type": "L",
            "Air_temperature_K": 298.0,
            "Process_temperature_K": 308.0,
            "Rotational_speed_rpm": 1500,
            "Torque_Nm": 40.0,
            "Tool_wear_min": 10,
        },
        {
            "Type": "H",
            "Air_temperature_K": 303.0,
            "Process_temperature_K": 312.0,
            "Rotational_speed_rpm": 1300,
            "Torque_Nm": 65.0,
            "Tool_wear_min": 240,
        },
    ]
    res = client.predict_batch(records)
    assert res["count"] == 2
    assert len(res["predictions"]) == 2
    assert res["predictions"][1]["risk_level"] == "high"


def test_api_client_explain_success() -> None:
    """Verify explain returns SHAP contributions and narrative."""

    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/explain"
        return httpx.Response(
            200,
            json={
                "prediction_id": None,
                "model_name": "random_forest",
                "model_version": "1.0.0",
                "predicted_probability": 0.88,
                "base_value": 0.034,
                "contributions": [
                    {
                        "feature": "Tool_wear_min",
                        "display_name": "Tool Wear",
                        "feature_value": 240.0,
                        "contribution": 0.35,
                    }
                ],
                "top_risk_drivers": [],
                "top_risk_mitigators": [],
                "summary_text": "High tool wear drives failure probability up to 88%.",
                "causality_disclaimer": "SHAP values reflect statistical attribution.",
            },
        )

    transport = MockTransport(mock_handler)
    client = ApiClient(base_url="http://mock-api:8000", transport=transport)
    res = client.explain({"Type": "L", "Tool_wear_min": 240})
    assert res["predicted_probability"] == 0.88
    assert res["base_value"] == 0.034
    assert len(res["contributions"]) == 1


def test_api_client_get_monitoring_summary() -> None:
    """Verify get_monitoring_summary parses operational telemetry."""

    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/monitoring/summary"
        return httpx.Response(
            200,
            json={
                "telemetry": {
                    "total_requests": 150,
                    "error_rate": 0.0,
                    "latencies_ms": {"p50": 1.2, "p95": 3.4, "p99": 5.0},
                },
                "drift": {"drift_detected": False, "drifted_features_count": 0},
                "alerts": {"active_alerts_count": 0, "active_alerts": []},
            },
        )

    transport = MockTransport(mock_handler)
    client = ApiClient(base_url="http://mock-api:8000", transport=transport)
    summary = client.get_monitoring_summary()
    assert summary["telemetry"]["total_requests"] == 150
    assert summary["drift"]["drift_detected"] is False
