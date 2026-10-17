"""Unit tests for dataset validation."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from app.data.validation import validate_dataset


def _categories(report) -> set[str]:
    return {issue.category for issue in report.issues}


class TestValidData:
    def test_valid_frame_is_valid(self, valid_frame: pd.DataFrame) -> None:
        report = validate_dataset(valid_frame)
        assert report.is_valid
        assert not report.errors
        assert report.n_rows == 6
        assert report.n_columns == valid_frame.shape[1]

    def test_validation_does_not_mutate_input(self, valid_frame: pd.DataFrame) -> None:
        before = valid_frame.copy(deep=True)
        validate_dataset(valid_frame)
        pd.testing.assert_frame_equal(valid_frame, before)

    def test_report_serialises(self, valid_frame: pd.DataFrame) -> None:
        payload = validate_dataset(valid_frame).to_dict()
        assert payload["is_valid"] is True
        assert "issues" in payload

    def test_report_summary_mentions_result(self, valid_frame: pd.DataFrame) -> None:
        assert "Result: VALID" in validate_dataset(valid_frame).summary()


class TestSchemaProblems:
    def test_missing_column_is_error(self, make_frame: Callable[..., pd.DataFrame]) -> None:
        frame = make_frame(4).drop(columns=["Torque [Nm]"])
        report = validate_dataset(frame)
        assert not report.is_valid
        assert "missing_columns" in _categories(report)

    def test_missing_target_can_be_allowed(self, make_frame: Callable[..., pd.DataFrame]) -> None:
        frame = make_frame(4).drop(columns=["Machine failure"])
        assert not validate_dataset(frame).is_valid
        assert validate_dataset(frame, require_target=False).is_valid

    def test_unexpected_column_warns_only(self, make_frame: Callable[..., pd.DataFrame]) -> None:
        frame = make_frame(4)
        frame["extra"] = 1
        report = validate_dataset(frame)
        assert report.is_valid
        assert "unexpected_columns" in _categories(report)
        assert report.warnings

    def test_non_dataframe_raises(self) -> None:
        with pytest.raises(TypeError):
            validate_dataset([1, 2, 3])  # type: ignore[arg-type]


class TestDtypes:
    def test_string_in_sensor_column_is_error(
        self, make_frame: Callable[..., pd.DataFrame]
    ) -> None:
        frame = make_frame(4, **{"Torque [Nm]": ["a", "b", "c", "d"]})
        report = validate_dataset(frame)
        assert not report.is_valid
        assert "invalid_dtype" in _categories(report)

    def test_numeric_type_column_is_error(self, make_frame: Callable[..., pd.DataFrame]) -> None:
        frame = make_frame(4, Type=[1, 2, 3, 4])
        report = validate_dataset(frame)
        assert not report.is_valid
        assert "invalid_dtype" in _categories(report)

    def test_float_binary_column_is_error(self, make_frame: Callable[..., pd.DataFrame]) -> None:
        frame = make_frame(4, **{"Machine failure": [0.0, 0.0, 1.0, 0.0]})
        report = validate_dataset(frame)
        assert not report.is_valid
        assert "invalid_dtype" in _categories(report)


class TestValueProblems:
    def test_out_of_range_sensor_is_error(self, make_frame: Callable[..., pd.DataFrame]) -> None:
        frame = make_frame(4, **{"Air temperature [K]": [298.0, 298.0, 999.0, 298.0]})
        report = validate_dataset(frame)
        assert not report.is_valid
        assert "out_of_range" in _categories(report)

    def test_invalid_category_is_error(self, make_frame: Callable[..., pd.DataFrame]) -> None:
        frame = make_frame(4, Type=["L", "X", "H", "M"])
        report = validate_dataset(frame)
        assert not report.is_valid
        assert "invalid_category" in _categories(report)

    def test_invalid_binary_value_is_error(self, make_frame: Callable[..., pd.DataFrame]) -> None:
        frame = make_frame(4, **{"TWF": [0, 2, 0, 0]})
        report = validate_dataset(frame)
        assert not report.is_valid

    def test_value_check_can_be_skipped(self, make_frame: Callable[..., pd.DataFrame]) -> None:
        frame = make_frame(4, **{"Air temperature [K]": [999.0, 999.0, 999.0, 999.0]})
        report = validate_dataset(frame)
        assert not report.is_valid


class TestMissingAndDuplicates:
    def test_missing_value_is_error(self, make_frame: Callable[..., pd.DataFrame]) -> None:
        frame = make_frame(4, **{"Torque [Nm]": [40.0, None, 42.0, 43.0]})
        report = validate_dataset(frame)
        assert not report.is_valid
        assert "missing_values" in _categories(report)

    def test_missing_check_can_be_skipped(self, make_frame: Callable[..., pd.DataFrame]) -> None:
        frame = make_frame(4, **{"Torque [Nm]": [40.0, None, 42.0, 43.0]})
        assert validate_dataset(frame, check_missing=False).is_valid

    def test_duplicate_rows_warn(self, make_frame: Callable[..., pd.DataFrame]) -> None:
        frame = make_frame(4)
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        report = validate_dataset(frame)
        assert report.is_valid
        assert "duplicate_rows" in _categories(report)

    def test_duplicate_check_can_be_skipped(self, make_frame: Callable[..., pd.DataFrame]) -> None:
        frame = make_frame(4)
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        report = validate_dataset(frame, check_duplicates=False)
        assert "duplicate_rows" not in _categories(report)


class TestTargetDerivedColumns:
    def test_failure_mode_columns_are_excluded_from_features(self) -> None:
        from app.data.schema import RAW_FEATURE_COLUMNS, TARGET_DERIVED_COLUMNS

        assert all(column not in RAW_FEATURE_COLUMNS for column in TARGET_DERIVED_COLUMNS)
