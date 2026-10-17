"""Environment-based configuration loading for SentinelPM.

Precedence (lowest to highest):
1. Built-in defaults defined on :class:`Settings`.
2. Optional YAML file pointed at by ``SENTINELPM_CONFIG_PATH`` or ``config_path``.
3. A ``.env`` file (if present) parsed with python-dotenv.
4. Process environment variables prefixed with ``SENTINELPM_``.

All relative paths are resolved against the project root so that the
configuration never depends on the caller's working directory.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator

ENV_PREFIX = "SENTINELPM_"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

EnvironmentName = Literal["dev", "test", "prod"]


class Settings(BaseModel):
    """Application settings resolved from defaults, YAML, .env and environment."""

    project_name: str = "sentinelpm"
    environment: EnvironmentName = "dev"
    log_level: str = "INFO"
    random_seed: int = 42
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")

    model_config = {"frozen": True, "validate_default": True}

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        level = value.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if level not in allowed:
            msg = f"log_level must be one of {sorted(allowed)}, got {value!r}"
            raise ValueError(msg)
        return level

    @field_validator("data_dir", "artifacts_dir")
    @classmethod
    def _resolve_path(cls, value: Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    @property
    def raw_data_dir(self) -> Path:
        """Directory holding the untouched source dataset."""
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        """Directory holding derived datasets."""
        return self.data_dir / "processed"


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)
    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        msg = f"Config file must contain a mapping, got {type(payload).__name__}"
        raise TypeError(msg)
    return payload


def _settings_from_env() -> dict[str, Any]:
    """Map ``SENTINELPM_*`` environment variables onto Settings field names."""
    mapping: dict[str, Any] = {}
    for key, raw in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        field = key[len(ENV_PREFIX) :].lower()
        mapping[field] = raw
    return mapping


def load_settings(
    config_path: Path | str | None = None,
    env_file: Path | str | None = None,
) -> Settings:
    """Load settings from defaults, an optional YAML file and the environment.

    Args:
        config_path: Optional YAML config file. Falls back to the
            ``SENTINELPM_CONFIG_PATH`` environment variable when omitted.
        env_file: Optional dotenv file. Falls back to ``.env`` in the project
            root when omitted; missing files are ignored silently.

    Returns:
        A fully resolved, immutable :class:`Settings` instance.
    """
    dotenv_path = Path(env_file) if env_file is not None else PROJECT_ROOT / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path, override=False)

    merged: dict[str, Any] = {}
    resolved_config = config_path or os.environ.get(f"{ENV_PREFIX}CONFIG_PATH")
    if resolved_config:
        path = Path(resolved_config)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        merged.update(_load_yaml_config(path))
    merged.update(_settings_from_env())
    return Settings(**merged)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached settings instance."""
    return load_settings()
