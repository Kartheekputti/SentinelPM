"""Unit tests for evaluation metrics and threshold selection."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.exceptions import TrainingError
from app.training.evaluation import (
    compare_models,
    compute_metrics,
    select_threshold,
    threshold_analysis,
)


class TestComputeMetrics:
    def test_confusion_counts(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.4, 0.35, 0.8])
        metrics = compute_metrics(y_true, y_score, threshold=0.5)
        assert metrics.true_positives == 1
        assert metrics.false_positives == 0
        assert metrics.true_negatives == 2
        assert metrics.false_negatives == 1

    def test_threshold_dependent_scores(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.4, 0.35, 0.8])
        metrics = compute_metrics(y_true, y_score, threshold=0.5)
        assert metrics.precision == pytest.approx(1.0)
        assert metrics.recall == pytest.approx(0.5)
        assert metrics.f1 == pytest.approx(2 / 3)

    def test_pr_auc_in_range(self) -> None:
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_score = np.array([0.2, 0.9, 0.3, 0.7, 0.1, 0.8])
        metrics = compute_metrics(y_true, y_score)
        assert 0.0 <= metrics.pr_auc <= 1.0
        assert metrics.roc_auc is not None and 0.0 <= metrics.roc_auc <= 1.0

    @pytest.mark.filterwarnings("ignore:No positive class found")
    def test_single_class_roc_auc_is_none(self) -> None:
        metrics = compute_metrics(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))
        assert metrics.roc_auc is None

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(TrainingError):
            compute_metrics(np.array([0, 1]), np.array([0.2]))

    def test_empty_raises(self) -> None:
        with pytest.raises(TrainingError):
            compute_metrics(np.array([]), np.array([]))

    def test_serialises(self) -> None:
        metrics = compute_metrics(np.array([0, 1]), np.array([0.2, 0.9]))
        payload = metrics.to_dict()
        assert payload["confusion_matrix"]["true_positives"] == 1


class TestThresholdAnalysis:
    def test_length_matches_grid(self) -> None:
        rows = threshold_analysis(
            np.array([0, 1, 0, 1]),
            np.array([0.2, 0.8, 0.3, 0.7]),
            thresholds=(0.25, 0.5, 0.75),
        )
        assert len(rows) == 3
        assert {"threshold", "precision", "recall", "f1"} <= set(rows[0])

    def test_recall_decreases_as_threshold_rises(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        y_score = np.array([0.2, 0.8, 0.3, 0.7])
        rows = threshold_analysis(y_true, y_score, thresholds=(0.25, 0.75))
        assert rows[0]["recall"] >= rows[1]["recall"]


class TestSelectThreshold:
    def test_returns_threshold_in_range(self) -> None:
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_score = np.array([0.1, 0.9, 0.2, 0.8, 0.15, 0.85])
        threshold, value = select_threshold(y_true, y_score, metric="f1")
        assert 0.0 <= threshold <= 1.0
        assert 0.0 <= value <= 1.0

    def test_prefers_separating_threshold(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        threshold, value = select_threshold(y_true, y_score, metric="f1")
        assert 0.2 < threshold <= 0.8
        assert value == pytest.approx(1.0)

    def test_unsupported_metric_raises(self) -> None:
        with pytest.raises(TrainingError):
            select_threshold(np.array([0, 1]), np.array([0.2, 0.8]), metric="accuracy")


class TestCompareModels:
    def test_selects_highest_primary_metric(self) -> None:
        leaderboard = [
            {"name": "a", "cv_score": 0.3},
            {"name": "b", "cv_score": 0.7},
            {"name": "c", "cv_score": 0.5},
        ]
        assert compare_models(leaderboard, "cv_score")["name"] == "b"

    def test_empty_leaderboard_raises(self) -> None:
        with pytest.raises(TrainingError):
            compare_models([], "cv_score")

    def test_missing_metric_raises(self) -> None:
        with pytest.raises(TrainingError):
            compare_models([{"name": "a"}], "cv_score")
