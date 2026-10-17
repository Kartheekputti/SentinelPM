"""Model training, hyperparameter tuning and evaluation."""

from app.training.artifacts import (
    ArtifactError,
    load_model,
    model_path,
    save_model,
    write_json,
)
from app.training.config import (
    DEFAULT_PRIMARY_METRIC,
    ModelSpec,
    TrainingConfig,
    default_training_config,
    load_training_config,
    training_config_from_mapping,
)
from app.training.evaluation import (
    BinaryMetrics,
    compare_models,
    compute_metrics,
    select_threshold,
    threshold_analysis,
)
from app.training.models import (
    SUPPORTED_MODELS,
    build_model_pipeline,
    positive_class_weight,
)
from app.training.trainer import ModelResult, TrainingReport, run_training

__all__ = [
    "DEFAULT_PRIMARY_METRIC",
    "SUPPORTED_MODELS",
    "ArtifactError",
    "BinaryMetrics",
    "ModelResult",
    "ModelSpec",
    "TrainingConfig",
    "TrainingReport",
    "build_model_pipeline",
    "compare_models",
    "compute_metrics",
    "default_training_config",
    "load_model",
    "load_training_config",
    "model_path",
    "positive_class_weight",
    "run_training",
    "save_model",
    "select_threshold",
    "threshold_analysis",
    "training_config_from_mapping",
    "write_json",
]
