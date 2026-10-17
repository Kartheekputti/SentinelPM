"""Evaluation metrics, threshold analysis and model comparison.

Accuracy is deliberately absent from the reported metrics: with ~3.4% failure
rate a trivial model can score ~96.6% accuracy while catching zero failures.
The primary ranking metric is PR-AUC (average precision).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.core.exceptions import TrainingError
from app.core.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_THRESHOLD_METRICS: tuple[str, ...] = ("f1", "precision", "recall")
DEFAULT_THRESHOLD_GRID: tuple[float, ...] = tuple(round(x, 4) for x in np.linspace(0.01, 0.99, 99))


@dataclass(frozen=True)
class BinaryMetrics:
    """Classification metrics for a single threshold.

    Attributes:
        threshold: Probability threshold applied to positive-class scores.
        precision: Positive predictive value.
        recall: Sensitivity / true-positive rate.
        f1: Harmonic mean of precision and recall.
        pr_auc: Area under the precision-recall curve (threshold independent).
        roc_auc: Area under the ROC curve (threshold independent), or ``None``
            when only one class is present.
        true_positives / false_positives / true_negatives / false_negatives:
            Confusion-matrix counts at ``threshold``.
        n_samples: Number of scored samples.
        positive_rate: Observed positive base rate.
    """

    threshold: float
    precision: float
    recall: float
    f1: float
    pr_auc: float
    roc_auc: float | None
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    n_samples: int
    positive_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "confusion_matrix": {
                "true_positives": self.true_positives,
                "false_positives": self.false_positives,
                "true_negatives": self.true_negatives,
                "false_negatives": self.false_negatives,
            },
            "n_samples": self.n_samples,
            "positive_rate": self.positive_rate,
        }


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
) -> BinaryMetrics:
    """Compute threshold-dependent and threshold-independent metrics.

    Args:
        y_true: Binary ground-truth labels.
        y_score: Positive-class probability scores.
        threshold: Decision threshold applied to ``y_score``.

    Returns:
        A :class:`BinaryMetrics` instance.

    Raises:
        TrainingError: If the label and score arrays differ in length or are
            empty.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape[0] != y_score.shape[0]:
        msg = "y_true and y_score must have the same length"
        raise TrainingError(msg)
    if y_true.shape[0] == 0:
        msg = "Cannot compute metrics on empty arrays"
        raise TrainingError(msg)

    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    roc_auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None

    metrics = BinaryMetrics(
        threshold=float(threshold),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        pr_auc=float(average_precision_score(y_true, y_score)),
        roc_auc=roc_auc,
        true_positives=int(tp),
        false_positives=int(fp),
        true_negatives=int(tn),
        false_negatives=int(fn),
        n_samples=int(y_true.shape[0]),
        positive_rate=float(y_true.mean()),
    )
    return metrics


def threshold_analysis(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLD_GRID,
) -> list[dict[str, float]]:
    """Tabulate precision, recall and F1 across a grid of thresholds.

    Args:
        y_true: Binary ground-truth labels.
        y_score: Positive-class probability scores.
        thresholds: Threshold values to evaluate.

    Returns:
        A list of ``{threshold, precision, recall, f1}`` dictionaries.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            }
        )
    return rows


def select_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    metric: str = "f1",
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLD_GRID,
) -> tuple[float, float]:
    """Choose the threshold maximising ``metric``.

    The threshold is selected on out-of-fold training predictions, never on the
    held-out test set, to keep the test set independent.

    Args:
        y_true: Binary ground-truth labels.
        y_score: Positive-class probability scores.
        metric: One of ``f1``, ``precision`` or ``recall``.
        thresholds: Candidate thresholds.

    Returns:
        Tuple of ``(best_threshold, best_metric_value)``.

    Raises:
        TrainingError: If ``metric`` is unsupported.
    """
    if metric not in SUPPORTED_THRESHOLD_METRICS:
        msg = f"Unsupported threshold metric {metric!r}; choose from {SUPPORTED_THRESHOLD_METRICS}"
        raise TrainingError(msg)

    rows = threshold_analysis(y_true, y_score, thresholds=thresholds)
    best = max(rows, key=lambda row: (row[metric], -row["threshold"]))
    return float(best["threshold"]), float(best[metric])


def compare_models(leaderboard: list[dict[str, Any]], primary_metric: str) -> dict[str, Any]:
    """Select the best entry from a leaderboard.

    Args:
        leaderboard: List of dictionaries each containing ``name`` and the
            primary metric value.
        primary_metric: Key used to rank entries.

    Returns:
        The entry with the highest primary-metric value.

    Raises:
        TrainingError: If the leaderboard is empty or lacks the metric key.
    """
    if not leaderboard:
        msg = "Cannot compare an empty leaderboard"
        raise TrainingError(msg)
    for entry in leaderboard:
        if primary_metric not in entry:
            msg = f"Leaderboard entry missing metric {primary_metric!r}"
            raise TrainingError(msg)
    return max(leaderboard, key=lambda entry: entry[primary_metric])
