"""Persistence helpers for trained models and reports.

Models are stored as full scikit-learn pipelines with joblib so that
preprocessing travels with the estimator and can be reloaded for inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from app.core.exceptions import SentinelPMError
from app.core.logging import get_logger

logger = get_logger(__name__)


class ArtifactError(SentinelPMError):
    """Raised when an artifact cannot be written or loaded."""


def model_path(name: str, models_dir: Path | str) -> Path:
    """Return the artifact path for a model name."""
    return Path(models_dir) / f"{name}.joblib"


def save_model(pipeline: Pipeline, name: str, models_dir: Path | str) -> Path:
    """Persist a fitted pipeline to disk.

    Args:
        pipeline: The fitted scikit-learn pipeline.
        name: Model name used for the filename.
        models_dir: Destination directory; created if missing.

    Returns:
        The written artifact path.

    Raises:
        ArtifactError: If the model cannot be serialised.
    """
    directory = Path(models_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = model_path(name, directory)
    try:
        joblib.dump(pipeline, path)
    except (OSError, TypeError, ValueError) as exc:
        msg = f"Failed to save model {name!r} to {path}: {exc}"
        raise ArtifactError(msg) from exc
    logger.info("Saved model %s to %s", name, path)
    return path


def load_model(path: Path | str) -> Pipeline:
    """Load a previously saved pipeline.

    Args:
        path: Path to a ``.joblib`` model artifact.

    Returns:
        The deserialised pipeline.

    Raises:
        ArtifactError: If the file is missing or cannot be deserialised.
    """
    model_file = Path(path)
    if not model_file.exists():
        msg = f"Model artifact not found: {model_file}"
        raise ArtifactError(msg)
    try:
        pipeline = joblib.load(model_file)
    except (OSError, TypeError, ValueError) as exc:
        msg = f"Failed to load model from {model_file}: {exc}"
        raise ArtifactError(msg) from exc
    if not hasattr(pipeline, "predict"):
        msg = f"Loaded object from {model_file} is not a usable estimator"
        raise ArtifactError(msg)
    return pipeline


def write_json(payload: dict[str, Any], path: Path | str) -> Path:
    """Write a JSON-serialisable payload to disk.

    Args:
        payload: The data to write.
        path: Destination file path; parent directories are created.

    Returns:
        The written path.

    Raises:
        ArtifactError: If the payload cannot be serialised or written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        msg = f"Failed to write JSON to {destination}: {exc}"
        raise ArtifactError(msg) from exc
    return destination
