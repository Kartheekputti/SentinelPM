"""Unit tests for exploratory data analysis."""

from __future__ import annotations

import json
from collections.abc import Callable

import pandas as pd

from app.features.eda import (
    class_balance,
    detect_outliers_iqr,
    failure_rate_by_type,
    feature_target_correlation,
    generate_eda_report,
    numeric_distributions,
    top_feature_correlations,
)


class TestClassBalance:
    def test_counts(self, make_frame: Callable[..., pd.DataFrame]) -> None:
        balance = class_balance(make_frame(10))
        assert balance["n_positive"] == 1
        assert balance["n_negative"] == 9
        assert balance["positive_rate"] == 0.1
        assert balance["imbalance_ratio"] == 9.0


class TestNumericDistributions:
    def test_includes_known_features(self, valid_frame: pd.DataFrame) -> None:
        summary = numeric_distributions(valid_frame)
        assert "Torque [Nm]" in summary
        assert summary["Torque [Nm]"]["count"] == len(valid_frame)


class TestFeatureTargetCorrelation:
    def test_values_within_range(self, valid_frame: pd.DataFrame) -> None:
        correlations = feature_target_correlation(valid_frame)
        assert correlations
        for value in correlations.values():
            assert value is None or -1.0 <= value <= 1.0


class TestFailureRateByType:
    def test_reports_each_category(self, valid_frame: pd.DataFrame) -> None:
        rates = failure_rate_by_type(valid_frame)
        assert set(rates) == {"L", "M", "H"}
        assert all(0.0 <= stats["positive_rate"] <= 1.0 for stats in rates.values())


class TestOutlierDetection:
    def test_injected_outlier_is_detected(self, make_frame: Callable[..., pd.DataFrame]) -> None:
        frame = make_frame(20)
        frame.loc[0, "Torque [Nm]"] = 100_000.0
        outliers = detect_outliers_iqr(frame, columns=("Torque [Nm]",))
        assert outliers["Torque [Nm]"]["count"] >= 1

    def test_no_outliers_in_linear_data(self, make_frame: Callable[..., pd.DataFrame]) -> None:
        outliers = detect_outliers_iqr(make_frame(30), columns=("Torque [Nm]",))
        assert outliers["Torque [Nm]"]["count"] == 0


class TestTopFeatureCorrelations:
    def test_returns_limited_pairs(self, valid_frame: pd.DataFrame) -> None:
        pairs = top_feature_correlations(valid_frame, top_n=3)
        assert len(pairs) <= 3
        for pair in pairs:
            assert "feature_a" in pair and "feature_b" in pair


class TestEDAReport:
    def test_report_serialises(self, valid_frame: pd.DataFrame) -> None:
        payload = generate_eda_report(valid_frame).to_dict()
        assert json.loads(json.dumps(payload))["n_rows"] == len(valid_frame)

    def test_report_is_reproducible(self, valid_frame: pd.DataFrame) -> None:
        first = generate_eda_report(valid_frame).to_dict()
        second = generate_eda_report(valid_frame).to_dict()
        assert first == second

    def test_summary_contains_sections(self, valid_frame: pd.DataFrame) -> None:
        summary = generate_eda_report(valid_frame).summary()
        assert "class balance" in summary
        assert "failure rate by type" in summary
