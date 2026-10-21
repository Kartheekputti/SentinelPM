"""Unit tests for model registry and tracking (M06)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from app.core.exceptions import ModelRegistryError
from app.features.pipeline import build_feature_pipeline
from app.registry.manager import ModelRegistryManager
from app.registry.tracker import MLflowTracker


@pytest.fixture
def mock_models_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with a dummy model and metadata."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)

    # Create dummy trained model
    pipeline = Pipeline(
        [
            ("features", build_feature_pipeline(scale=False)),
            ("model", RandomForestClassifier(n_estimators=5, random_state=42)),
        ]
    )
    import joblib

    joblib.dump(pipeline, models_dir / "random_forest.joblib")

    # Metadata
    meta = {
        "best_model": "random_forest",
        "primary_metric": "average_precision",
        "models": [
            {
                "name": "random_forest",
                "threshold": 0.35,
                "cv_score": 0.88,
                "artifact": str(models_dir / "random_forest.joblib"),
            }
        ],
        "feature_names": ["air_temperature_k"],
    }
    with (models_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f)

    return models_dir


def test_registry_load_approved_model(mock_models_dir: Path) -> None:
    """Ensure registry loads approved model and metadata successfully."""
    registry = ModelRegistryManager(models_dir=mock_models_dir)
    pipeline, metadata = registry.load_approved_model()

    assert metadata.model_name == "random_forest"
    assert metadata.threshold == 0.35
    assert metadata.cv_score == 0.88
    assert hasattr(pipeline, "predict_proba")


def test_registry_missing_model_raises(tmp_path: Path) -> None:
    """Ensure attempting to load a non-existent model raises ModelRegistryError."""
    empty_dir = tmp_path / "empty_models"
    empty_dir.mkdir()
    registry = ModelRegistryManager(models_dir=empty_dir)

    with pytest.raises(ModelRegistryError, match="Model artifact not found"):
        registry.load_approved_model("non_existent_model")


def test_registry_rejects_path_traversal(mock_models_dir: Path) -> None:
    """Verify security sandboxing prevents directory traversal attacks."""
    registry = ModelRegistryManager(models_dir=mock_models_dir)
    with pytest.raises(ModelRegistryError, match="path traversal"):
        # Even if someone attempts to pass a filename that resolves outside
        registry._safe_resolve_artifact("../../../evil.joblib")


def test_mlflow_tracker_logging(tmp_path: Path, sample_trained_rf: Pipeline) -> None:
    """Verify MLflowTracker logs runs, parameters, metrics, and models."""
    mlruns_dir = tmp_path / "mlruns"
    tracker = MLflowTracker(
        tracking_uri=str(mlruns_dir),
        experiment_name="test_experiment",
    )

    run_id = tracker.log_model_run(
        run_name="test_run",
        model_name="random_forest",
        pipeline=sample_trained_rf,
        params={"n_estimators": 10},
        metrics={"pr_auc": 0.95, "f1": 0.85},
    )

    assert run_id is not None
    assert len(run_id) > 0
    assert mlruns_dir.exists()
