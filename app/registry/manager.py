"""Model registry management and secure artifact resolution.

Loads approved model versions, validates integrity, and retrieves training metadata.
Enforces path sandboxing to protect against directory traversal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sklearn.pipeline import Pipeline

from app.core.config import Settings, get_settings
from app.core.exceptions import ModelRegistryError
from app.core.logging import get_logger
from app.training.artifacts import load_model

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelMetadata:
    """Metadata describing an approved trained model."""

    model_name: str
    model_version: str
    threshold: float
    primary_metric: str
    cv_score: float | None
    feature_names: list[str]
    artifact_path: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "threshold": round(self.threshold, 4),
            "primary_metric": self.primary_metric,
            "cv_score": round(self.cv_score, 4) if self.cv_score is not None else None,
            "feature_names": self.feature_names,
            "artifact_path": self.artifact_path,
            "created_at": self.created_at,
        }


class ModelRegistryManager:
    """Manages approved model artifacts and metadata with security containment."""

    def __init__(
        self,
        *,
        models_dir: Path | str | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.models_dir = Path(models_dir or self.settings.models_dir).resolve()

        if not self.models_dir.exists():
            self.models_dir.mkdir(parents=True, exist_ok=True)

    def _safe_resolve_artifact(self, model_file_name: str | Path) -> Path:
        """Ensure the target artifact resides strictly within the models directory.

        Prevents path traversal and loading arbitrary external code.
        """
        raw_path = Path(model_file_name)
        if ".." in raw_path.parts or raw_path.is_absolute():
            msg = f"Security violation: path traversal detected for {model_file_name!r}"
            raise ModelRegistryError(msg)
        candidate = (self.models_dir / raw_path).resolve()
        try:
            candidate.relative_to(self.models_dir)
        except ValueError as exc:
            msg = f"Security violation: path traversal detected for {model_file_name!r}"
            raise ModelRegistryError(msg) from exc
        return candidate

    def _read_metadata_json(self) -> dict[str, Any]:
        meta_file = self.models_dir / "metadata.json"
        if not meta_file.exists():
            return {}
        try:
            with meta_file.open(encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Could not parse %s: %s", meta_file, exc)
            return {}

    def get_model_metadata(
        self,
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> ModelMetadata:
        """Retrieve metadata for a candidate or approved model.

        Args:
            model_name: Optional name of the model. Defaults to approved best model.
            model_version: Optional version identifier.

        Returns:
            A populated :class:`ModelMetadata` instance.

        Raises:
            ModelRegistryError: If metadata cannot be resolved.
        """
        meta_dict = self._read_metadata_json()
        target_name = model_name or meta_dict.get("best_model") or self.settings.model_name
        target_version = model_version or self.settings.model_version

        artifact_file = self._safe_resolve_artifact(f"{target_name}.joblib")
        if not artifact_file.exists():
            msg = f"Model artifact not found for {target_name!r} at {artifact_file}"
            raise ModelRegistryError(msg)

        threshold = 0.5
        cv_score = None
        for m in meta_dict.get("models", []):
            if m.get("name") == target_name:
                threshold = float(m.get("threshold", 0.5))
                cv_score = float(m.get("cv_score")) if m.get("cv_score") is not None else None
                break

        feature_names = meta_dict.get("feature_names", [])

        # File modification time as creation stamp
        mtime = datetime.fromtimestamp(artifact_file.stat().st_mtime, tz=UTC).isoformat()

        return ModelMetadata(
            model_name=target_name,
            model_version=target_version,
            threshold=threshold,
            primary_metric=meta_dict.get("primary_metric", "average_precision"),
            cv_score=cv_score,
            feature_names=feature_names,
            artifact_path=str(artifact_file),
            created_at=mtime,
        )

    def load_approved_model(
        self,
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> tuple[Pipeline, ModelMetadata]:
        """Load the approved model pipeline and its metadata.

        Args:
            model_name: Name of the model. Defaults to approved best model.
            model_version: Version tag.

        Returns:
            Tuple of (loaded Pipeline, ModelMetadata).

        Raises:
            ModelRegistryError: If artifact cannot be found or loaded.
        """
        metadata = self.get_model_metadata(model_name, model_version)
        artifact_path = Path(metadata.artifact_path)

        try:
            pipeline = load_model(artifact_path)
        except Exception as exc:
            msg = (
                f"Failed to load approved model {metadata.model_name!r} from {artifact_path}: {exc}"
            )
            raise ModelRegistryError(msg) from exc

        logger.info(
            "Loaded approved model %s (version=%s, threshold=%.3f)",
            metadata.model_name,
            metadata.model_version,
            metadata.threshold,
        )
        return pipeline, metadata

    def list_available_models(self) -> list[ModelMetadata]:
        """List all available models in the registry."""
        meta_dict = self._read_metadata_json()
        results: list[ModelMetadata] = []
        for model_entry in meta_dict.get("models", []):
            name = model_entry.get("name")
            if name:
                try:
                    results.append(self.get_model_metadata(name))
                except ModelRegistryError:
                    continue
        return results
