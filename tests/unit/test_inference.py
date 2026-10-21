"""Unit tests for inference and risk scoring (M06)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from sklearn.pipeline import Pipeline

from app.core.exceptions import InferenceError
from app.inference.schemas import PredictionResult, SensorRecord
from app.inference.service import InferenceService
from app.registry.manager import ModelMetadata


@pytest.fixture
def mock_metadata() -> ModelMetadata:
    return ModelMetadata(
        model_name="random_forest",
        model_version="1.0.0",
        threshold=0.35,
        primary_metric="average_precision",
        cv_score=0.88,
        feature_names=["air_temperature_k"],
        artifact_path="dummy.joblib",
        created_at="2026-10-21T00:00:00Z",
    )


@pytest.fixture
def valid_sensor_dict() -> dict[str, object]:
    return {
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0.0,
        "Type": "M",
    }


def test_sensor_record_validation(valid_sensor_dict: dict[str, object]) -> None:
    """Verify SensorRecord parses valid data with canonical column names."""
    record = SensorRecord.model_validate(valid_sensor_dict)
    assert record.air_temperature_k == 298.1
    assert record.type == "M"

    raw_dict = record.to_raw_dict()
    assert "Air temperature [K]" in raw_dict
    assert "Type" in raw_dict


def test_sensor_record_snake_case_input() -> None:
    """Verify SensorRecord parses snake_case inputs."""
    data = {
        "air_temperature_k": 298.1,
        "process_temperature_k": 308.6,
        "rotational_speed_rpm": 1551.0,
        "torque_nm": 42.8,
        "tool_wear_min": 0.0,
        "type": "L",
    }
    record = SensorRecord.model_validate(data)
    assert record.air_temperature_k == 298.1
    assert record.type == "L"


def test_sensor_record_rejects_out_of_bounds() -> None:
    """Verify physically impossible values are rejected."""
    bad_data = {
        "Air temperature [K]": 100.0,  # Below minimum 250K
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0.0,
        "Type": "M",
    }
    with pytest.raises(ValidationError):
        SensorRecord.model_validate(bad_data)


def test_sensor_record_rejects_target_label() -> None:
    """Verify target label is strictly rejected to prevent leakage."""
    leak_data = {
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0.0,
        "Type": "M",
        "Machine failure": 0,
    }
    with pytest.raises(ValidationError, match="strictly forbidden"):
        SensorRecord.model_validate(leak_data)


def test_sensor_record_rejects_target_derived_indicators() -> None:
    """Verify failure-mode indicators are rejected to prevent leakage."""
    leak_data = {
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0.0,
        "Type": "M",
        "TWF": 0,
    }
    with pytest.raises(ValidationError, match="strictly forbidden"):
        SensorRecord.model_validate(leak_data)


def test_inference_single_prediction(
    sample_trained_rf: Pipeline,
    mock_metadata: ModelMetadata,
    valid_sensor_dict: dict[str, object],
) -> None:
    """Verify single prediction returns structured PredictionResult."""
    service = InferenceService(sample_trained_rf, mock_metadata)
    result = service.predict_single(valid_sensor_dict)

    assert isinstance(result, PredictionResult)
    assert result.model_version == "1.0.0"
    assert result.model_name == "random_forest"
    assert 0.0 <= result.failure_probability <= 1.0
    assert result.risk_level in ("low", "medium", "high")
    assert result.threshold == 0.35
    assert isinstance(result.predicted_failure, bool)


def test_inference_batch_prediction(
    sample_trained_rf: Pipeline,
    mock_metadata: ModelMetadata,
    valid_sensor_dict: dict[str, object],
) -> None:
    """Verify batch prediction returns one result per valid record."""
    service = InferenceService(sample_trained_rf, mock_metadata)
    records = [valid_sensor_dict, valid_sensor_dict]

    results = service.predict_batch(records)
    assert len(results) == 2
    assert results[0].prediction_id != results[1].prediction_id


def test_inference_batch_limit_exceeded(
    sample_trained_rf: Pipeline,
    mock_metadata: ModelMetadata,
    valid_sensor_dict: dict[str, object],
) -> None:
    """Verify batch size limit enforcement."""
    service = InferenceService(sample_trained_rf, mock_metadata)
    service.max_batch_size = 2
    records = [valid_sensor_dict] * 3

    with pytest.raises(InferenceError, match="exceeds maximum allowed limit"):
        service.predict_batch(records)


def test_inference_empty_batch_raises(
    sample_trained_rf: Pipeline,
    mock_metadata: ModelMetadata,
) -> None:
    """Verify empty batch raises InferenceError."""
    service = InferenceService(sample_trained_rf, mock_metadata)
    with pytest.raises(InferenceError, match="cannot be empty"):
        service.predict_batch([])


def test_inference_deterministic_repeat(
    sample_trained_rf: Pipeline,
    mock_metadata: ModelMetadata,
    valid_sensor_dict: dict[str, object],
) -> None:
    """Verify repeated predictions on identical input yield identical probabilities."""
    service = InferenceService(sample_trained_rf, mock_metadata)
    res1 = service.predict_single(valid_sensor_dict)
    res2 = service.predict_single(valid_sensor_dict)

    assert res1.failure_probability == res2.failure_probability
    assert res1.predicted_failure == res2.predicted_failure
    assert res1.risk_level == res2.risk_level
