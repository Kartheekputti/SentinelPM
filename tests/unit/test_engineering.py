"""Unit tests for feature engineering transformers and helpers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from app.core.exceptions import FeatureError
from app.data.schema import RAW_FEATURE_COLUMNS, TARGET_DERIVED_COLUMNS
from app.features.definitions import (
    ENGINEERED_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    ROTATIONAL_POWER,
    TEMPERATURE_DIFFERENCE,
)
from app.features.engineering import (
    SensorFeatureEngineer,
    add_engineered_features,
    extract_target,
    prepare_feature_frame,
)


class TestAddEngineeredFeatures:
    def test_temperature_difference(self, valid_frame: pd.DataFrame) -> None:
        result = add_engineered_features(valid_frame)
        expected = valid_frame["Process temperature [K]"] - valid_frame["Air temperature [K]"]
        pd.testing.assert_series_equal(
            result[TEMPERATURE_DIFFERENCE].rename(None),
            expected.rename(None),
            check_names=False,
        )

    def test_rotational_power(self, valid_frame: pd.DataFrame) -> None:
        result = add_engineered_features(valid_frame)
        expected = (
            valid_frame["Torque [Nm]"] * valid_frame["Rotational speed [rpm]"] * (2 * np.pi / 60)
        )
        np.testing.assert_allclose(result[ROTATIONAL_POWER], expected)

    def test_new_columns_present(self, valid_frame: pd.DataFrame) -> None:
        result = add_engineered_features(valid_frame)
        assert set(ENGINEERED_FEATURE_COLUMNS).issubset(result.columns)

    def test_input_is_not_mutated(self, valid_frame: pd.DataFrame) -> None:
        before = valid_frame.copy(deep=True)
        add_engineered_features(valid_frame)
        pd.testing.assert_frame_equal(valid_frame, before)

    def test_deterministic(self, valid_frame: pd.DataFrame) -> None:
        first = add_engineered_features(valid_frame)
        second = add_engineered_features(valid_frame)
        pd.testing.assert_frame_equal(first, second)

    def test_missing_sensor_column_raises(self, valid_frame: pd.DataFrame) -> None:
        frame = valid_frame.drop(columns=["Torque [Nm]"])
        with pytest.raises(FeatureError, match="sensor"):
            add_engineered_features(frame)

    def test_non_numeric_sensor_raises(self, make_frame: Callable[..., pd.DataFrame]) -> None:
        frame = make_frame(4, **{"Torque [Nm]": ["a", "b", "c", "d"]})
        with pytest.raises(FeatureError, match="numeric"):
            add_engineered_features(frame)

    def test_non_dataframe_raises(self) -> None:
        with pytest.raises(FeatureError):
            add_engineered_features([1, 2, 3])  # type: ignore[arg-type]


class TestPrepareFeatureFrame:
    def test_selects_only_allowed_columns(self, valid_frame: pd.DataFrame) -> None:
        prepared = prepare_feature_frame(valid_frame)
        assert list(prepared.columns) == list(RAW_FEATURE_COLUMNS)

    def test_excludes_target_and_failure_modes(self, valid_frame: pd.DataFrame) -> None:
        prepared = prepare_feature_frame(valid_frame)
        for column in ("Machine failure", *TARGET_DERIVED_COLUMNS, "UDI", "Product ID"):
            assert column not in prepared.columns

    def test_missing_column_raises(self, valid_frame: pd.DataFrame) -> None:
        with pytest.raises(FeatureError, match="Missing"):
            prepare_feature_frame(valid_frame.drop(columns=["Type"]))

    def test_non_dataframe_raises(self) -> None:
        with pytest.raises(FeatureError):
            prepare_feature_frame("not a frame")  # type: ignore[arg-type]


class TestExtractTarget:
    def test_returns_integer_series(self, valid_frame: pd.DataFrame) -> None:
        target = extract_target(valid_frame)
        assert target.dtype.kind in "iu"
        assert target.name == "Machine failure"

    def test_missing_target_raises(self, valid_frame: pd.DataFrame) -> None:
        with pytest.raises(FeatureError, match="Target"):
            extract_target(valid_frame.drop(columns=["Machine failure"]))


class TestSensorFeatureEngineer:
    def test_fit_returns_self(self, valid_frame: pd.DataFrame) -> None:
        transformer = SensorFeatureEngineer()
        assert transformer.fit(valid_frame) is transformer

    def test_transform_output_columns_and_order(self, valid_frame: pd.DataFrame) -> None:
        result = SensorFeatureEngineer().fit_transform(valid_frame)
        assert list(result.columns) == list(NUMERIC_FEATURE_COLUMNS)

    def test_get_feature_names_out(self) -> None:
        names = SensorFeatureEngineer().get_feature_names_out()
        assert list(names) == list(NUMERIC_FEATURE_COLUMNS)

    def test_fit_rejects_missing_columns(self, valid_frame: pd.DataFrame) -> None:
        with pytest.raises(FeatureError):
            SensorFeatureEngineer().fit(valid_frame.drop(columns=["Type", "Torque [Nm]"]))

    def test_transform_is_reproducible(self, valid_frame: pd.DataFrame) -> None:
        transformer = SensorFeatureEngineer().fit(valid_frame)
        pd.testing.assert_frame_equal(
            transformer.transform(valid_frame), transformer.transform(valid_frame)
        )
