"""End-to-end integration test verifying the complete operational chain:
Data Validation -> Pipeline Features -> Model Scoring -> FastAPI REST API -> Dashboard Client -> Monitoring.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.core.exceptions import ApiClientError
from app.dashboard.client import ApiClient


@pytest.fixture
def integrated_client() -> ApiClient:
    """Instantiate a real FastAPI app connected via TestClient to ApiClient."""
    app = create_app()
    test_client = TestClient(app)
    return ApiClient(client=test_client)


@pytest.mark.integration
def test_full_system_e2e_workflow(integrated_client: ApiClient) -> None:
    """Verify complete operational workflow through Dashboard ApiClient against real trained models."""
    # 1. Health probe
    health = integrated_client.check_health()
    assert health["status"] == "healthy"
    assert health["service"] == "sentinelpm-api"

    # 2. Readiness probe
    ready = integrated_client.check_readiness()
    assert ready["status"] == "ready"
    assert ready["model_loaded"] is True
    assert ready["model_name"] in ["random_forest", "xgboost"]

    # 3. Model info specification
    model_info = integrated_client.get_model_info()
    assert model_info["model_name"] in ["random_forest", "xgboost"]
    assert "feature_names" in model_info
    assert len(model_info["feature_names"]) >= 5

    # 4. Single Machine Prediction (normal operation)
    normal_record = {
        "Type": "M",
        "Air_temperature_K": 298.2,
        "Process_temperature_K": 308.7,
        "Rotational_speed_rpm": 1408,
        "Torque_Nm": 46.3,
        "Tool_wear_min": 3,
    }
    pred_res = integrated_client.predict(normal_record)
    assert "predicted_failure" in pred_res
    assert "failure_probability" in pred_res
    assert pred_res["risk_level"] in ["low", "medium", "high"]
    assert pred_res["failure_probability"] >= 0.0

    # 5. On-Demand SHAP Attribution
    explain_res = integrated_client.explain(normal_record, top_n=3)
    assert explain_res["model_name"] in ["random_forest", "xgboost"]
    assert "contributions" in explain_res
    assert len(explain_res["contributions"]) > 0
    assert "summary_text" in explain_res
    assert "causality_disclaimer" in explain_res

    # 6. High-Throughput Batch Prediction
    batch_records = [
        {
            "Type": "L",
            "Air_temperature_K": 298.1,
            "Process_temperature_K": 308.6,
            "Rotational_speed_rpm": 1551,
            "Torque_Nm": 42.8,
            "Tool_wear_min": 0,
        },
        {
            "Type": "H",
            "Air_temperature_K": 303.0,
            "Process_temperature_K": 312.0,
            "Rotational_speed_rpm": 1320,
            "Torque_Nm": 65.0,
            "Tool_wear_min": 240,
        },
    ]
    batch_res = integrated_client.predict_batch(batch_records)
    assert batch_res["count"] == 2
    assert len(batch_res["predictions"]) == 2

    # 7. Monitoring Telemetry Recording
    monitoring = integrated_client.get_monitoring_summary()
    telemetry = monitoring["telemetry"]
    assert telemetry["total_requests"] >= 2  # At least 1 single and 1 batch call
    assert telemetry["error_rate"] == 0.0

    # 8. Controlled Error Handling: Target leakage strictly rejected
    with pytest.raises(ApiClientError, match="Validation error"):
        integrated_client.predict({"Type": "M", "Machine failure": 1, "Tool_wear_min": 50})
