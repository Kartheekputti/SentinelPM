"""Unit tests for training configuration parsing."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from app.core.exceptions import ConfigurationError
from app.training.config import (
    TrainingConfig,
    default_training_config,
    load_training_config,
    training_config_from_mapping,
)


class TestDefaults:
    def test_default_has_all_models(self) -> None:
        config = default_training_config()
        names = {spec.name for spec in config.models}
        assert names == {"dummy", "logistic_regression", "random_forest", "xgboost"}

    def test_primary_metric_is_pr_auc(self) -> None:
        assert default_training_config().primary_metric == "average_precision"

    def test_enabled_models_filters_disabled(self) -> None:
        config = training_config_from_mapping(
            {"models": {"dummy": {"enabled": True}, "xgboost": {"enabled": False}}}
        )
        enabled = {spec.name for spec in config.enabled_models()}
        assert enabled == {"dummy"}

    def test_get_model_unknown_raises(self) -> None:
        with pytest.raises(ConfigurationError):
            default_training_config().get_model("does_not_exist")


class TestMappingParsing:
    def test_overrides_scalars(self) -> None:
        config = training_config_from_mapping({"random_seed": 7, "cv_folds": 3})
        assert config.random_seed == 7
        assert config.cv_folds == 3

    def test_parses_scaling_flag(self) -> None:
        config = training_config_from_mapping(
            {"models": {"logistic_regression": {"scale": True, "params": {"model__C": [1.0]}}}}
        )
        spec = config.get_model("logistic_regression")
        assert spec.scale is True
        assert spec.search_space() == {"model__C": [1.0]}

    def test_invalid_test_size_raises(self) -> None:
        with pytest.raises(ConfigurationError):
            training_config_from_mapping({"test_size": 1.5})

    def test_invalid_cv_folds_raises(self) -> None:
        with pytest.raises(ConfigurationError):
            training_config_from_mapping({"cv_folds": 1})

    def test_invalid_search_raises(self) -> None:
        with pytest.raises(ConfigurationError):
            training_config_from_mapping({"search": "bayesian"})

    def test_no_enabled_models_raises(self) -> None:
        with pytest.raises(ConfigurationError):
            training_config_from_mapping({"models": {"dummy": {"enabled": False}}})


class TestLoadFromFile:
    def test_missing_explicit_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError):
            load_training_config(tmp_path / "nope.yaml")

    def test_loads_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("cv_folds: 4\nmodels:\n  dummy:\n    enabled: true\n", encoding="utf-8")
        config = load_training_config(path)
        assert config.cv_folds == 4

    def test_non_mapping_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("- 1\n- 2\n", encoding="utf-8")
        with pytest.raises(ConfigurationError):
            load_training_config(path)

    def test_to_dict_is_serialisable(self) -> None:
        import json

        payload = default_training_config().to_dict()
        assert json.loads(json.dumps(payload))["models"]

    def test_config_is_immutable(self) -> None:
        config = default_training_config()
        with pytest.raises(FrozenInstanceError):
            config.cv_folds = 3  # type: ignore[misc]

    def test_training_config_type(self) -> None:
        assert isinstance(default_training_config(), TrainingConfig)
