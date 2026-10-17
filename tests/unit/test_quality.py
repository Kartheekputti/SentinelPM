"""Unit tests for data-quality reporting."""

from __future__ import annotations

import json
from collections.abc import Callable

import pandas as pd
import pytest

from app.core.exceptions import DataValidationError
from app.data.quality import ensure_known_columns, generate_quality_report


class TestQualityReport:
    def test_dimensions_and_memory(self, valid_frame: pd.DataFrame) -> None:
        report = generate_quality_report(valid_frame)
        assert report.n_rows == 6
        assert report.n_columns == valid_frame.shape[1]
        assert report.memory_usage_mb > 0

    def test_numeric_summary_covers_numeric_columns(self, valid_frame: pd.DataFrame) -> None:
        report = generate_quality_report(valid_frame)
        assert "Torque [Nm]" in report.numeric_summary
        summary = report.numeric_summary["Torque [Nm]"]
        assert summary.count == 6
        assert summary.minimum <= summary.median <= summary.maximum

    def test_missing_values_detected(self, make_frame: Callable[..., pd.DataFrame]) -> None:
        frame = make_frame(4, **{"Torque [Nm]": [40.0, None, None, 43.0]})
        report = generate_quality_report(frame)
        assert report.missing_values["Torque [Nm]"] == 2
        assert report.total_missing == 2

    def test_duplicate_rows_detected(self, make_frame: Callable[..., pd.DataFrame]) -> None:
        frame = make_frame(3)
        frame = pd.concat([frame, frame], ignore_index=True)
        report = generate_quality_report(frame)
        assert report.duplicate_rows == 3

    def test_constant_columns_detected(self, make_frame: Callable[..., pd.DataFrame]) -> None:
        frame = make_frame(4)
        frame["constant"] = 7
        report = generate_quality_report(frame)
        assert "constant" in report.constant_columns

    def test_target_balance(self, make_frame: Callable[..., pd.DataFrame]) -> None:
        frame = make_frame(10)
        report = generate_quality_report(frame)
        assert report.target_balance is not None
        assert report.target_balance["n_positive"] == 1
        assert report.target_balance["n_negative"] == 9
        assert report.target_balance["positive_rate"] == pytest.approx(0.1)

    def test_target_balance_none_when_absent(self, make_frame: Callable[..., pd.DataFrame]) -> None:
        frame = make_frame(4).drop(columns=["Machine failure"])
        assert generate_quality_report(frame).target_balance is None

    def test_report_is_json_serialisable(self, valid_frame: pd.DataFrame) -> None:
        payload = generate_quality_report(valid_frame).to_dict()
        assert json.loads(json.dumps(payload))["n_rows"] == 6

    def test_summary_contains_key_fields(self, valid_frame: pd.DataFrame) -> None:
        summary = generate_quality_report(valid_frame).summary()
        assert "Data quality report" in summary
        assert "target balance" in summary

    def test_empty_frame_raises(self) -> None:
        with pytest.raises(DataValidationError):
            generate_quality_report(pd.DataFrame())


class TestEnsureKnownColumns:
    def test_passes_for_complete_frame(self, valid_frame: pd.DataFrame) -> None:
        ensure_known_columns(valid_frame)

    def test_raises_for_missing_columns(self, make_frame: Callable[..., pd.DataFrame]) -> None:
        frame = make_frame(3).drop(columns=["Type"])
        with pytest.raises(DataValidationError, match="Type"):
            ensure_known_columns(frame)
