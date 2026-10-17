"""Training configuration and experiment specifications.

The configuration is intentionally data-driven: models, hyperparameter search
spaces and the selection metric are declared here (and in
``configs/training.yaml``) rather than hard-coded in the training loop.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from app.core.config import PROJECT_ROOT
from app.core.exceptions import ConfigurationError

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "training.yaml"

#: Metric used to rank models. PR-AUC (average precision) is chosen over
#: accuracy or ROC-AUC because failures are rare (~3.4%) and the cost of a
#: missed failure dominates the cost of a false alarm.
DEFAULT_PRIMARY_METRIC = "average_precision"


@dataclass(frozen=True)
class ModelSpec:
    """Declarative specification for a single candidate model."""

    name: str
    enabled: bool = True
    scale: bool = False
    params: Mapping[str, Sequence[Any]] = field(default_factory=dict)

    def search_space(self) -> dict[str, Sequence[Any]]:
        """Return the estimator search space with ``model__`` prefixed keys."""
        return {key: list(values) for key, values in self.params.items()}


@dataclass(frozen=True)
class TrainingConfig:
    """Top-level training configuration."""

    random_seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    scoring: str = DEFAULT_PRIMARY_METRIC
    primary_metric: str = DEFAULT_PRIMARY_METRIC
    threshold_metric: str = "f1"
    search: str = "grid"
    n_iter: int = 20
    n_jobs: int = -1
    models: tuple[ModelSpec, ...] = ()

    def enabled_models(self) -> tuple[ModelSpec, ...]:
        """Return only the models marked as enabled."""
        return tuple(spec for spec in self.models if spec.enabled)

    def get_model(self, name: str) -> ModelSpec:
        """Return the spec for ``name``.

        Raises:
            ConfigurationError: If no model with that name is configured.
        """
        for spec in self.models:
            if spec.name == name:
                return spec
        msg = f"Unknown model in configuration: {name!r}"
        raise ConfigurationError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the configuration."""
        return {
            "random_seed": self.random_seed,
            "test_size": self.test_size,
            "cv_folds": self.cv_folds,
            "scoring": self.scoring,
            "primary_metric": self.primary_metric,
            "threshold_metric": self.threshold_metric,
            "search": self.search,
            "n_iter": self.n_iter,
            "n_jobs": self.n_jobs,
            "models": [
                {
                    "name": spec.name,
                    "enabled": spec.enabled,
                    "scale": spec.scale,
                    "params": {key: list(values) for key, values in spec.params.items()},
                }
                for spec in self.models
            ],
        }


DEFAULT_MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(name="dummy", enabled=True, scale=False, params={}),
    ModelSpec(
        name="logistic_regression",
        enabled=True,
        scale=True,
        params={"model__C": [0.1, 1.0, 10.0]},
    ),
    ModelSpec(
        name="random_forest",
        enabled=True,
        scale=False,
        params={
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 12],
            "model__min_samples_leaf": [1, 5],
        },
    ),
    ModelSpec(
        name="xgboost",
        enabled=True,
        scale=False,
        params={
            "model__n_estimators": [200, 400],
            "model__max_depth": [3, 6],
            "model__learning_rate": [0.05, 0.1],
        },
    ),
)


def default_training_config() -> TrainingConfig:
    """Return a config populated with the built-in model specifications."""
    return TrainingConfig(models=DEFAULT_MODEL_SPECS)


def _parse_models(raw_models: Any) -> tuple[ModelSpec, ...]:
    if raw_models is None:
        return DEFAULT_MODEL_SPECS
    if not isinstance(raw_models, Mapping):
        msg = "The 'models' section of the training config must be a mapping"
        raise ConfigurationError(msg)

    specs: list[ModelSpec] = []
    for name, settings in raw_models.items():
        if settings is None:
            settings = {}
        if not isinstance(settings, Mapping):
            msg = f"Model config for {name!r} must be a mapping"
            raise ConfigurationError(msg)
        specs.append(
            ModelSpec(
                name=str(name),
                enabled=bool(settings.get("enabled", True)),
                scale=bool(settings.get("scale", False)),
                params=settings.get("params", {}) or {},
            )
        )
    return tuple(specs)


def training_config_from_mapping(mapping: Mapping[str, Any]) -> TrainingConfig:
    """Build a :class:`TrainingConfig` from a mapping, validating key fields.

    Args:
        mapping: Configuration mapping, typically parsed from YAML.

    Returns:
        A validated :class:`TrainingConfig`.

    Raises:
        ConfigurationError: If any field is invalid.
    """
    base = default_training_config()
    config = replace(
        base,
        random_seed=int(mapping.get("random_seed", base.random_seed)),
        test_size=float(mapping.get("test_size", base.test_size)),
        cv_folds=int(mapping.get("cv_folds", base.cv_folds)),
        scoring=str(mapping.get("scoring", base.scoring)),
        primary_metric=str(mapping.get("primary_metric", base.primary_metric)),
        threshold_metric=str(mapping.get("threshold_metric", base.threshold_metric)),
        search=str(mapping.get("search", base.search)),
        n_iter=int(mapping.get("n_iter", base.n_iter)),
        n_jobs=int(mapping.get("n_jobs", base.n_jobs)),
        models=_parse_models(mapping.get("models")),
    )
    _validate(config)
    return config


def _validate(config: TrainingConfig) -> None:
    if not 0.0 < config.test_size < 1.0:
        msg = f"test_size must be in (0, 1), got {config.test_size}"
        raise ConfigurationError(msg)
    if config.cv_folds < 2:
        msg = f"cv_folds must be at least 2, got {config.cv_folds}"
        raise ConfigurationError(msg)
    if config.search not in {"grid", "random"}:
        msg = f"search must be 'grid' or 'random', got {config.search!r}"
        raise ConfigurationError(msg)
    if not config.enabled_models():
        msg = "At least one model must be enabled"
        raise ConfigurationError(msg)


def load_training_config(
    path: Path | str | None = None,
) -> TrainingConfig:
    """Load a training config from YAML, falling back to defaults.

    Args:
        path: Optional YAML path. Defaults to ``configs/training.yaml`` when it
            exists, otherwise the built-in defaults are used.

    Returns:
        A validated :class:`TrainingConfig`.

    Raises:
        ConfigurationError: If the file exists but is malformed.
    """
    resolved = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not resolved.exists():
        if path is None:
            return default_training_config()
        msg = f"Training config file not found: {resolved}"
        raise ConfigurationError(msg)

    with resolved.open(encoding="utf-8") as handle:
        mapping = yaml.safe_load(handle) or {}
    if not isinstance(mapping, Mapping):
        msg = f"Training config must be a mapping, got {type(mapping).__name__}"
        raise ConfigurationError(msg)
    return training_config_from_mapping(mapping)
