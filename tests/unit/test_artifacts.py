"""Unit tests for artifact persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.training.artifacts import (
    ArtifactError,
    load_model,
    model_path,
    save_model,
    write_json,
)
from app.training.config import ModelSpec
from app.training.models import build_model_pipeline


@pytest.fixture
def fitted_pipeline(make_frame, valid_frame):
    target = np.array([1 if index % 3 == 0 else 0 for index in range(len(valid_frame))])
    pipeline = build_model_pipeline(ModelSpec(name="logistic_regression", scale=True))
    pipeline.fit(valid_frame, target)
    return pipeline


class TestSaveLoadModel:
    def test_roundtrip_preserves_predictions(
        self, fitted_pipeline, valid_frame, tmp_path: Path
    ) -> None:
        path = save_model(fitted_pipeline, "logreg", tmp_path)
        assert path.exists()
        reloaded = load_model(path)
        np.testing.assert_allclose(
            reloaded.predict_proba(valid_frame),
            fitted_pipeline.predict_proba(valid_frame),
        )

    def test_model_path_naming(self, tmp_path: Path) -> None:
        assert model_path("rf", tmp_path).name == "rf.joblib"

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ArtifactError, match="not found"):
            load_model(tmp_path / "missing.joblib")

    def test_load_non_estimator_raises(self, tmp_path: Path) -> None:
        import joblib

        path = tmp_path / "not_a_model.joblib"
        joblib.dump({"a": 1}, path)
        with pytest.raises(ArtifactError):
            load_model(path)


class TestWriteJson:
    def test_writes_and_creates_parents(self, tmp_path: Path) -> None:
        path = write_json({"a": 1}, tmp_path / "nested" / "file.json")
        assert path.exists()
        assert "a" in path.read_text(encoding="utf-8")

    def test_unserialisable_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ArtifactError):
            write_json({"bad": object()}, tmp_path / "bad.json")
