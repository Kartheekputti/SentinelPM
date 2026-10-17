"""Tests for environment-based configuration loading."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import PROJECT_ROOT, Settings, get_settings, load_settings
from app.core.logging import get_logger, setup_logging


class TestSettingsDefaults:
    def test_defaults_without_env_file(self) -> None:
        settings = load_settings(env_file=Path("does-not-exist.env"))
        assert settings.environment == "dev"
        assert settings.random_seed == 42
        assert settings.log_level == "INFO"

    def test_relative_paths_resolve_against_project_root(self) -> None:
        settings = load_settings(env_file=Path("does-not-exist.env"))
        assert settings.data_dir.is_absolute()
        assert settings.data_dir == PROJECT_ROOT / "data"
        assert settings.raw_data_dir == settings.data_dir / "raw"
        assert settings.processed_data_dir == settings.data_dir / "processed"

    def test_absolute_paths_are_preserved(self, tmp_path: Path) -> None:
        settings = Settings(data_dir=tmp_path, artifacts_dir=tmp_path / "art")
        assert settings.data_dir == tmp_path
        assert settings.artifacts_dir == (tmp_path / "art").resolve()

    def test_settings_are_immutable(self) -> None:
        settings = load_settings(env_file=Path("does-not-exist.env"))
        with pytest.raises(ValidationError):
            settings.random_seed = 0  # type: ignore[misc]


class TestEnvironmentOverrides:
    def test_env_vars_override_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SENTINELPM_ENVIRONMENT", "test")
        monkeypatch.setenv("SENTINELPM_LOG_LEVEL", "debug")
        monkeypatch.setenv("SENTINELPM_RANDOM_SEED", "7")
        settings = load_settings(env_file=Path("does-not-exist.env"))
        assert settings.environment == "test"
        assert settings.log_level == "DEBUG"
        assert settings.random_seed == 7

    def test_invalid_environment_name_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SENTINELPM_ENVIRONMENT", "staging")
        with pytest.raises(ValidationError):
            load_settings(env_file=Path("does-not-exist.env"))

    def test_invalid_log_level_rejected(self) -> None:
        with pytest.raises(ValueError, match="log_level"):
            Settings(log_level="LOUD")


class TestYamlConfig:
    def test_yaml_file_is_merged(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config = tmp_path / "config.yaml"
        config.write_text("random_seed: 99\nlog_level: WARNING\n", encoding="utf-8")
        monkeypatch.delenv("SENTINELPM_RANDOM_SEED", raising=False)
        settings = load_settings(config_path=config, env_file=Path("missing.env"))
        assert settings.random_seed == 99
        assert settings.log_level == "WARNING"

    def test_env_still_overrides_yaml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config = tmp_path / "config.yaml"
        config.write_text("random_seed: 99\n", encoding="utf-8")
        monkeypatch.setenv("SENTINELPM_RANDOM_SEED", "5")
        settings = load_settings(config_path=config, env_file=Path("missing.env"))
        assert settings.random_seed == 5

    def test_missing_yaml_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_settings(config_path=tmp_path / "nope.yaml")

    def test_non_mapping_yaml_raises(self, tmp_path: Path) -> None:
        config = tmp_path / "config.yaml"
        config.write_text("- 1\n- 2\n", encoding="utf-8")
        with pytest.raises(TypeError):
            load_settings(config_path=config)


class TestCachedSettings:
    def test_get_settings_is_cached(self) -> None:
        get_settings.cache_clear()
        first = get_settings()
        second = get_settings()
        assert first is second
        get_settings.cache_clear()


class TestLogging:
    def test_setup_logging_initialises(self) -> None:
        setup_logging(level="INFO", force=True)
        logger = get_logger("tests.logging")
        assert logger.name == "tests.logging"
        assert logging.getLogger().level == logging.INFO

    def test_log_file_is_written(self, tmp_path: Path) -> None:
        log_path = tmp_path / "logs" / "test.log"
        setup_logging(level="INFO", log_file=log_path, force=True)
        get_logger("tests.file").info("hello sentinel")
        assert log_path.exists()
        assert "hello sentinel" in log_path.read_text(encoding="utf-8")

    def test_invalid_level_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid log level"):
            setup_logging(level="NOT_A_LEVEL", force=True)
