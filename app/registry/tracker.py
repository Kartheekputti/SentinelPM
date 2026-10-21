"""MLflow experiment tracking integration for SentinelPM.

Logs parameters, cross-validation metrics, decision thresholds, and model artifacts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow
from sklearn.pipeline import Pipeline

from app.core.config import Settings, get_settings
from app.core.exceptions import ModelRegistryError
from app.core.logging import get_logger

logger = get_logger(__name__)

# Allow filesystem tracking in modern MLflow versions and silence git warnings when git is absent
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")


class MLflowTracker:
    """Experiment tracker wrapping MLflow client operations."""

    def __init__(
        self,
        *,
        tracking_uri: str | Path | None = None,
        experiment_name: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        active_settings = settings or get_settings()
        self.tracking_uri = str(tracking_uri or active_settings.mlflow_tracking_uri)
        self.experiment_name = experiment_name or active_settings.mlflow_experiment_name

        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            self.experiment = mlflow.set_experiment(self.experiment_name)
        except Exception as exc:
            msg = f"Failed to initialize MLflow experiment {self.experiment_name!r} at {self.tracking_uri!r}: {exc}"
            raise ModelRegistryError(msg) from exc

    def log_model_run(
        self,
        run_name: str,
        *,
        model_name: str,
        pipeline: Pipeline,
        params: dict[str, Any],
        metrics: dict[str, float],
        tags: dict[str, str] | None = None,
        artifacts: list[Path | str] | None = None,
    ) -> str:
        """Log a complete training run with model pipeline and evaluation metrics.

        Args:
            run_name: Identifier for the MLflow run.
            model_name: Model identifier (e.g. 'random_forest').
            pipeline: Trained scikit-learn Pipeline.
            params: Hyperparameters and training options.
            metrics: Evaluation metric dictionary (e.g. pr_auc, f1, precision, recall).
            tags: Optional run tags (e.g. environment, version).
            artifacts: Optional file paths to log as artifacts.

        Returns:
            The MLflow run ID.

        Raises:
            ModelRegistryError: If MLflow logging fails.
        """
        try:
            with mlflow.start_run(run_name=run_name) as run:
                run_id = run.info.run_id

                # Log tags
                all_tags = {"model_name": model_name}
                if tags:
                    all_tags.update(tags)
                mlflow.set_tags(all_tags)

                # Log parameters (flatten nested dictionaries if necessary)
                flat_params = {
                    key: str(val) if not isinstance(val, (int, float, bool, str)) else val
                    for key, val in params.items()
                }
                mlflow.log_params(flat_params)

                # Log metrics
                mlflow.log_metrics(metrics)

                # Log pipeline model artifact with cloudpickle for custom transformer support
                mlflow.sklearn.log_model(
                    sk_model=pipeline,
                    name=model_name,
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                )

                # Log additional artifacts if provided
                if artifacts:
                    for artifact in artifacts:
                        art_path = Path(artifact)
                        if art_path.exists():
                            mlflow.log_artifact(str(art_path))

                logger.info("Successfully logged run %s (run_id=%s) to MLflow", run_name, run_id)
                return run_id
        except Exception as exc:
            msg = f"Failed to log model run to MLflow: {exc}"
            raise ModelRegistryError(msg) from exc
