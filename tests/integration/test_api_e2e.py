"""End-to-end integration tests for FastAPI backend against real trained artifacts."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.api.dependencies import reset_dependencies
from app.core.config import get_settings


@pytest.fixture
def real_api_client() -> TestClient:
    """Create a test client backed by real trained model artifacts."""
    reset_dependencies()
    app = create_app()
    with TestClient(app) as client:
        yield client
    reset_dependencies()


@pytest.fixture
def sample_machine_record() -> dict[str, object]:
    return {
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 5.0,
        "Type": "M",
    }


@pytest.mark.integration
def test_real_api_lifecycle(
    real_api_client: TestClient,
    sample_machine_record: dict[str, object],
) -> None:
    """Test full API journey with health, readiness, prediction, SHAP, and telemetry."""
    settings = get_settings()
    real_model_file = settings.models_dir / f"{settings.model_name}.joblib"
    if not real_model_file.exists():
        pytest.skip(f"Real model artifact not found at {real_model_file}")

    # 1. Health check
    res_health = real_api_client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"

    # 2. Readiness probe
    res_ready = real_api_client.get("/ready")
    assert res_ready.status_code == 200
    assert res_ready.json()["model_loaded"] is True

    # 3. Model Info
    res_info = real_api_client.get("/model/info")
    assert res_info.status_code == 200
    info_data = res_info.json()
    assert info_data["model_name"] == settings.model_name
    assert "air_temperature_k" in info_data["feature_names"]

    # 4. Single prediction
    res_pred = real_api_client.post("/predict", json=sample_machine_record)
    assert res_pred.status_code == 200
    pred_data = res_pred.json()
    assert "prediction_id" in pred_data
    assert 0.0 <= pred_data["failure_probability"] <= 1.0
    assert pred_data["risk_level"] in ("low", "medium", "high")

    # 5. Batch prediction
    batch_records = [sample_machine_record, sample_machine_record]
    res_batch = real_api_client.post("/predict/batch", json={"records": batch_records})
    assert res_batch.status_code == 200
    batch_data = res_batch.json()
    assert batch_data["count"] == 2

    # 6. SHAP explanation
    res_explain = real_api_client.post(
        "/explain",
        json={"record": sample_machine_record, "top_n": 3},
    )
    assert res_explain.status_code == 200
    explain_data = res_explain.json()
    assert len(explain_data["contributions"]) > 0
    assert "causality_disclaimer" in explain_data

    # 7. Monitoring summary
    res_mon = real_api_client.get("/monitoring/summary")
    assert res_mon.status_code == 200
    mon_data = res_mon.json()
    assert mon_data["telemetry"]["total_requests"] >= 4
