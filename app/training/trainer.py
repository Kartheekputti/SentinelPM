"""End-to-end model training, tuning and evaluation.

Design decisions:
    - The held-out test split is created once, stratified on the target, and is
      never used for hyperparameter or threshold selection.
    - Model selection ranks candidates by cross-validated PR-AUC on the training
      split.
    - Decision thresholds are chosen from out-of-fold training predictions.
    - Class imbalance is handled with class weights (not resampling), so no
      synthetic samples are created before splitting and no leakage occurs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.features.engineering import extract_target, prepare_feature_frame
from app.training.artifacts import save_model, write_json
from app.training.config import TrainingConfig, default_training_config
from app.training.evaluation import (
    BinaryMetrics,
    compare_models,
    compute_metrics,
    select_threshold,
    threshold_analysis,
)
from app.training.models import build_model_pipeline, positive_class_weight

logger = get_logger(__name__)


@dataclass
class ModelResult:
    """Training and evaluation outcome for a single candidate model."""

    name: str
    best_params: dict[str, Any]
    cv_score: float
    threshold: float
    train_metrics: BinaryMetrics
    test_metrics: BinaryMetrics
    test_metrics_at_default: BinaryMetrics
    threshold_analysis: list[dict[str, float]]
    artifact_path: str
    fit_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "best_params": self.best_params,
            "cv_score": self.cv_score,
            "threshold": self.threshold,
            "train_metrics": self.train_metrics.to_dict(),
            "test_metrics": self.test_metrics.to_dict(),
            "test_metrics_at_default_threshold": self.test_metrics_at_default.to_dict(),
            "threshold_analysis": self.threshold_analysis,
            "artifact_path": self.artifact_path,
            "fit_seconds": self.fit_seconds,
        }


@dataclass
class TrainingReport:
    """Aggregate report for a training run."""

    config: dict[str, Any]
    random_seed: int
    primary_metric: str
    n_train: int
    n_test: int
    feature_names: list[str]
    results: list[ModelResult] = field(default_factory=list)
    best_model: str = ""
    report_path: str | None = None

    def leaderboard(self) -> list[dict[str, Any]]:
        """Return a comparison table of all trained models.

        The table is ranked by cross-validated PR-AUC (``cv_score``), the
        configured primary metric.
        """
        entries = [
            {
                "name": result.name,
                "cv_score": result.cv_score,
                "threshold": result.threshold,
                "test_pr_auc": result.test_metrics.pr_auc,
                "test_roc_auc": result.test_metrics.roc_auc,
                "test_precision": result.test_metrics.precision,
                "test_recall": result.test_metrics.recall,
                "test_f1": result.test_metrics.f1,
                "test_false_positives": result.test_metrics.false_positives,
                "test_false_negatives": result.test_metrics.false_negatives,
            }
            for result in self.results
        ]
        entries.sort(key=lambda entry: entry["cv_score"], reverse=True)
        return entries

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "random_seed": self.random_seed,
            "primary_metric": self.primary_metric,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "best_model": self.best_model,
            "leaderboard": self.leaderboard(),
            "results": [result.to_dict() for result in self.results],
            "report_path": self.report_path,
        }

    def summary(self) -> str:
        lines = [
            "Training report",
            f"  train rows: {self.n_train}  test rows: {self.n_test}",
            f"  features: {len(self.feature_names)}",
            f"  primary metric: {self.primary_metric} (cross-validated)",
            "  leaderboard (ranked by cv PR-AUC):",
            "    {:<22}{:>10}{:>10}{:>10}{:>10}{:>8}{:>8}".format(
                "model", "cv_pr_auc", "pr_auc", "recall", "f1", "fp", "fn"
            ),
        ]
        for entry in self.leaderboard():
            lines.append(
                "    {:<22}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.4f}{:>8}{:>8}".format(
                    entry["name"],
                    entry["cv_score"],
                    entry["test_pr_auc"],
                    entry["test_recall"],
                    entry["test_f1"],
                    entry["test_false_positives"],
                    entry["test_false_negatives"],
                )
            )
        lines.append(f"  best model: {self.best_model}")
        return "\n".join(lines)


def run_training(
    frame: pd.DataFrame,
    config: TrainingConfig | None = None,
    *,
    settings: Settings | None = None,
    models_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> TrainingReport:
    """Train, tune and evaluate every enabled candidate model.

    Args:
        frame: Raw dataset including features and the target column.
        config: Training configuration; defaults to the packaged config.
        settings: Settings override; falls back to the cached settings.
        models_dir: Optional model artifact directory override.
        reports_dir: Optional report directory override.

    Returns:
        A populated :class:`TrainingReport`.
    """
    active_settings = settings or get_settings()
    active_config = config or default_training_config()
    seed = active_config.random_seed
    _set_global_seed(seed)

    features = prepare_feature_frame(frame)
    target = extract_target(frame)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=active_config.test_size,
        stratify=target,
        random_state=seed,
    )

    resolved_models_dir = (
        models_dir if models_dir is not None else active_settings.artifacts_dir / "models"
    )
    resolved_reports_dir = (
        reports_dir if reports_dir is not None else active_settings.artifacts_dir / "reports"
    )

    cv = StratifiedKFold(n_splits=active_config.cv_folds, shuffle=True, random_state=seed)
    report = TrainingReport(
        config=active_config.to_dict(),
        random_seed=seed,
        primary_metric=active_config.primary_metric,
        n_train=len(x_train),
        n_test=len(x_test),
        feature_names=list(features.columns),
    )

    y_train_array = y_train.to_numpy()
    y_test_array = y_test.to_numpy()
    pipeline_feature_names: list[str] = []

    for spec in active_config.enabled_models():
        logger.info("Training model: %s", spec.name)
        started = perf_counter()
        scale_pos_weight = positive_class_weight(y_train_array) if spec.name == "xgboost" else None
        pipeline = build_model_pipeline(
            spec,
            seed=seed,
            scale_pos_weight=scale_pos_weight,
            n_jobs=1,
        )
        search = _build_search(pipeline, spec, active_config, cv)

        search.fit(x_train, y_train)
        best_estimator = search.best_estimator_
        cv_score = float(search.best_score_)
        if not pipeline_feature_names:
            feature_step = best_estimator.named_steps["features"]
            pipeline_feature_names = [str(name) for name in feature_step.get_feature_names_out()]

        oof_scores = cross_val_predict(
            clone(best_estimator),
            x_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=active_config.n_jobs,
        )[:, 1]
        threshold, _ = select_threshold(
            y_train_array, oof_scores, metric=active_config.threshold_metric
        )

        test_scores = best_estimator.predict_proba(x_test)[:, 1]
        train_scores = best_estimator.predict_proba(x_train)[:, 1]

        result = ModelResult(
            name=spec.name,
            best_params={key: _json_safe(value) for key, value in search.best_params_.items()},
            cv_score=cv_score,
            threshold=threshold,
            train_metrics=compute_metrics(y_train_array, train_scores, threshold=threshold),
            test_metrics=compute_metrics(y_test_array, test_scores, threshold=threshold),
            test_metrics_at_default=compute_metrics(y_test_array, test_scores, threshold=0.5),
            threshold_analysis=threshold_analysis(y_train_array, oof_scores),
            artifact_path=str(save_model(best_estimator, spec.name, resolved_models_dir)),
            fit_seconds=perf_counter() - started,
        )
        report.results.append(result)
        logger.info(
            "Finished %s: cv_pr_auc=%.4f test_pr_auc=%.4f threshold=%.3f",
            spec.name,
            result.cv_score,
            result.test_metrics.pr_auc,
            result.threshold,
        )

    best_entry = compare_models(report.leaderboard(), "cv_score")
    report.best_model = str(best_entry["name"])
    if pipeline_feature_names:
        report.feature_names = pipeline_feature_names

    report_path = write_json(report.to_dict(), resolved_reports_dir / "training_report.json")
    report.report_path = str(report_path)

    _write_models_metadata(report, active_config, resolved_models_dir)
    logger.info("Best model: %s", report.best_model)
    return report


def _build_search(
    pipeline: Any,
    spec: Any,
    config: TrainingConfig,
    cv: StratifiedKFold,
) -> Any:
    grid = spec.search_space()
    if config.search == "random" and grid:
        return RandomizedSearchCV(
            pipeline,
            param_distributions=grid,
            n_iter=min(config.n_iter, _grid_size(grid)),
            scoring=config.scoring,
            cv=cv,
            n_jobs=config.n_jobs,
            refit=True,
            random_state=config.random_seed,
        )
    return GridSearchCV(
        pipeline,
        param_grid=grid,
        scoring=config.scoring,
        cv=cv,
        n_jobs=config.n_jobs,
        refit=True,
    )


def _grid_size(grid: dict[str, list[Any]]) -> int:
    size = 1
    for values in grid.values():
        size *= max(len(values), 1)
    return size


def _write_models_metadata(
    report: TrainingReport,
    config: TrainingConfig,
    models_dir: Path,
) -> None:
    best = next(
        (result for result in report.results if result.name == report.best_model),
        None,
    )
    payload = {
        "best_model": report.best_model,
        "primary_metric": config.primary_metric,
        "random_seed": config.random_seed,
        "feature_names": report.feature_names,
        "models": [
            {
                "name": result.name,
                "threshold": result.threshold,
                "cv_score": result.cv_score,
                "artifact": result.artifact_path,
            }
            for result in report.results
        ],
        "best_model_artifact": best.artifact_path if best else None,
        "best_model_threshold": best.threshold if best else None,
    }
    write_json(payload, models_dir / "metadata.json")


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value
