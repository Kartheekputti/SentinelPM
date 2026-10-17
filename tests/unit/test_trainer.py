"""Unit tests for the training orchestrator using synthetic data.

These tests verify pipeline mechanics only (artifacts, structure, determinism).
They make no claims about model quality and do not fabricate performance.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from app.core.config import Settings
from app.training.artifacts import load_model
from app.training.config import training_config_from_mapping
from app.training.trainer import TrainingReport, run_training


def _small_config() -> object:
    return training_config_from_mapping(
        {
            "random_seed": 7,
            "test_size": 0.3,
            "cv_folds": 3,
            "n_jobs": 1,
            "search": "grid",
            "models": {
                "dummy": {"enabled": True},
                "logistic_regression": {
                    "enabled": True,
                    "scale": True,
                    "params": {"model__C": [1.0]},
                },
            },
        }
    )


def _labeled_frame(make_frame: Callable[..., pd.DataFrame], n_rows: int = 150) -> pd.DataFrame:
    target = [1 if index % 6 == 0 else 0 for index in range(n_rows)]
    return make_frame(n_rows, **{"Machine failure": target})


@pytest.fixture(scope="module")
def trained_report(
    make_frame: Callable[..., pd.DataFrame], tmp_path_factory: pytest.TempPathFactory
) -> TrainingReport:
    base = tmp_path_factory.mktemp("trainer")
    settings = Settings(data_dir=base, artifacts_dir=base / "artifacts")
    frame = _labeled_frame(make_frame)
    return run_training(
        frame,
        _small_config(),
        settings=settings,
        models_dir=base / "models",
        reports_dir=base / "reports",
    )


class TestRunTraining:
    def test_all_enabled_models_are_trained(self, trained_report: TrainingReport) -> None:
        names = {result.name for result in trained_report.results}
        assert names == {"dummy", "logistic_regression"}

    def test_best_model_is_on_leaderboard(self, trained_report: TrainingReport) -> None:
        assert trained_report.best_model in {result.name for result in trained_report.results}

    def test_leaderboard_sorted_descending(self, trained_report: TrainingReport) -> None:
        scores = [entry["cv_score"] for entry in trained_report.leaderboard()]
        assert scores == sorted(scores, reverse=True)

    def test_threshold_within_range(self, trained_report: TrainingReport) -> None:
        for result in trained_report.results:
            assert 0.0 <= result.threshold <= 1.0

    def test_metrics_are_populated(self, trained_report: TrainingReport) -> None:
        for result in trained_report.results:
            assert 0.0 <= result.test_metrics.pr_auc <= 1.0
            assert result.test_metrics.n_samples > 0

    def test_feature_names_include_engineered_and_encoded(
        self, trained_report: TrainingReport
    ) -> None:
        names = trained_report.feature_names
        assert "temperature_difference_k" in names
        assert "rotational_power_w" in names
        assert any(name.startswith("type_") for name in names)
        assert len(names) == 10

    def test_accuracy_is_not_reported(self, trained_report: TrainingReport) -> None:
        payload = trained_report.results[0].test_metrics.to_dict()
        assert "accuracy" not in payload

    def test_summary_mentions_leaderboard(self, trained_report: TrainingReport) -> None:
        assert "leaderboard" in trained_report.summary()


class TestArtifacts:
    def test_model_files_written(self, trained_report: TrainingReport) -> None:
        for result in trained_report.results:
            assert Path(result.artifact_path).exists()

    def test_report_file_written(self, trained_report: TrainingReport) -> None:
        assert trained_report.report_path is not None
        path = Path(trained_report.report_path)
        assert path.exists()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["best_model"] == trained_report.best_model

    def test_model_reloads_and_predicts(
        self, trained_report: TrainingReport, make_frame: Callable[..., pd.DataFrame]
    ) -> None:
        artifact = Path(trained_report.results[0].artifact_path)
        model = load_model(artifact)
        frame = make_frame(5)
        probabilities = model.predict_proba(frame)
        assert probabilities.shape[0] == 5

    def test_metadata_written(self, trained_report: TrainingReport) -> None:
        models_dir = Path(trained_report.results[0].artifact_path).parent
        metadata = json.loads((models_dir / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["best_model"] == trained_report.best_model
        assert metadata["best_model_threshold"] is not None


class TestReproducibility:
    def test_same_seed_same_scores(
        self, make_frame: Callable[..., pd.DataFrame], tmp_path: Path
    ) -> None:
        settings = Settings(data_dir=tmp_path, artifacts_dir=tmp_path / "artifacts")
        frame = _labeled_frame(make_frame, n_rows=120)
        first = run_training(
            frame,
            _small_config(),
            settings=settings,
            models_dir=tmp_path / "m1",
            reports_dir=tmp_path / "r1",
        )
        second = run_training(
            frame,
            _small_config(),
            settings=settings,
            models_dir=tmp_path / "m2",
            reports_dir=tmp_path / "r2",
        )
        assert first.best_model == second.best_model
        assert [result.cv_score for result in first.results] == [
            result.cv_score for result in second.results
        ]
