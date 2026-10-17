"""Candidate estimators and model pipelines.

Each candidate is a full scikit-learn pipeline that couples preprocessing with
the estimator, so the artifact saved after training contains everything needed
for inference.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.core.exceptions import TrainingError
from app.core.logging import get_logger
from app.features.pipeline import build_feature_pipeline
from app.training.config import ModelSpec

logger = get_logger(__name__)

SUPPORTED_MODELS: tuple[str, ...] = (
    "dummy",
    "logistic_regression",
    "random_forest",
    "xgboost",
)


def _build_estimator(
    name: str,
    *,
    seed: int,
    scale_pos_weight: float | None,
    n_jobs: int,
) -> ClassifierMixin:
    if name == "dummy":
        return DummyClassifier(strategy="prior", random_state=seed)
    if name == "logistic_regression":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=n_jobs,
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            scale_pos_weight=scale_pos_weight if scale_pos_weight else 1.0,
            random_state=seed,
            n_jobs=n_jobs,
            eval_metric="logloss",
            tree_method="hist",
        )
    msg = f"Unsupported model: {name!r}. Supported models: {SUPPORTED_MODELS}"
    raise TrainingError(msg)


def build_model_pipeline(
    spec: ModelSpec,
    *,
    seed: int = 42,
    scale_pos_weight: float | None = None,
    n_jobs: int = 1,
) -> Pipeline:
    """Build a full preprocessing + estimator pipeline for ``spec``.

    Args:
        spec: The model specification.
        seed: Random seed for the estimator.
        scale_pos_weight: Positive-class weight for XGBoost (``n_negative /
            n_positive`` computed on the training split only).
        n_jobs: Thread count passed to the estimator.

    Returns:
        A :class:`~sklearn.pipeline.Pipeline` with steps ``features`` and
        ``model``.

    Raises:
        TrainingError: If the model name is unsupported.
    """
    estimator = _build_estimator(
        spec.name,
        seed=seed,
        scale_pos_weight=scale_pos_weight,
        n_jobs=n_jobs,
    )
    pipeline = Pipeline(
        steps=[
            ("features", build_feature_pipeline(scale=spec.scale)),
            ("model", estimator),
        ]
    )
    return pipeline


def positive_class_weight(target: np.ndarray) -> float:
    """Return ``n_negative / n_positive`` for XGBoost class balancing.

    Args:
        target: Binary target array.

    Returns:
        The ratio of negatives to positives, or ``1.0`` when there are no
        positive samples.
    """
    positives = int(np.sum(target == 1))
    negatives = int(np.sum(target == 0))
    if positives == 0:
        return 1.0
    return negatives / positives
