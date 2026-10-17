"""Integration test for the training pipeline on the real dataset.

Uses a lightweight configuration (dummy + logistic regression) so the test stays
fast while still running real experiments end to end.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import Settings, get_settings
from app.data.loader import load_raw_dataset
from app.data.schema import DEFAULT_RAW_FILENAME
from app.training.artifacts import load_model
from app.training.config import training_config_from_mapping
from app.training.trainer import run_training

pytestmark = pytest.mark.integration

_SETTINGS = get_settings()
_DATASET = _SETTINGS.raw_data_dir / DEFAULT_RAW_FILENAME

requires_dataset = pytest.mark.skipif(
    not _DATASET.exists(),
    reason=f"AI4I dataset not found at {_DATASET}",
)


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory):
    frame = load_raw_dataset(_DATASET, settings=_SETTINGS)
    config = training_config_from_mapping(
        {
            "random_seed": 42,
            "test_size": 0.2,
            "cv_folds": 3,
            "n_jobs": 1,
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
    base = tmp_path_factory.mktemp("training")
    settings = Settings(data_dir=_SETTINGS.data_dir, artifacts_dir=base)
    return run_training(
        frame,
        config,
        settings=settings,
        models_dir=base / "models",
        reports_dir=base / "reports",
    )


@requires_dataset
class TestTrainingIntegration:
    def test_models_trained_and_compared(self, report) -> None:
        assert {result.name for result in report.results} == {"dummy", "logistic_regression"}
        assert report.best_model in {result.name for result in report.results}

    def test_real_metrics_reported(self, report) -> None:
        for result in report.results:
            assert 0.0 <= result.test_metrics.pr_auc <= 1.0
            assert result.test_metrics.n_samples > 0
            assert result.test_metrics.roc_auc is None or 0.0 <= result.test_metrics.roc_auc <= 1.0

    def test_threshold_analysis_present(self, report) -> None:
        assert all(len(result.threshold_analysis) > 0 for result in report.results)

    def test_best_model_reloads_for_inference(self, report) -> None:
        best = next(result for result in report.results if result.name == report.best_model)
        model = load_model(Path(best.artifact_path))
        frame = load_raw_dataset(_DATASET, settings=_SETTINGS).head(10)
        probabilities = model.predict_proba(frame)
        assert probabilities.shape == (10, 2)

    def test_feature_count_matches_pipeline(self, report) -> None:
        assert len(report.feature_names) == 10
