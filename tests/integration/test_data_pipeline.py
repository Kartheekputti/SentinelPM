"""Integration tests against the real AI4I dataset.

These tests are skipped automatically when the dataset has not been downloaded.
"""

from __future__ import annotations

import pytest

from app.core.config import get_settings
from app.data.loader import load_raw_dataset
from app.data.quality import generate_quality_report
from app.data.schema import (
    DEFAULT_RAW_FILENAME,
    RAW_FEATURE_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_DERIVED_COLUMNS,
)
from app.data.validation import validate_dataset

pytestmark = pytest.mark.integration

_SETTINGS = get_settings()
_DATASET = _SETTINGS.raw_data_dir / DEFAULT_RAW_FILENAME

requires_dataset = pytest.mark.skipif(
    not _DATASET.exists(),
    reason=f"AI4I dataset not found at {_DATASET}",
)


@pytest.fixture(scope="module")
def raw_frame():
    return load_raw_dataset(_DATASET, settings=_SETTINGS)


@requires_dataset
class TestDatasetPipeline:
    def test_dataset_shape_and_columns(self, raw_frame) -> None:
        assert raw_frame.shape[0] == 10_000
        assert all(column in raw_frame.columns for column in REQUIRED_COLUMNS)

    def test_dataset_is_valid(self, raw_frame) -> None:
        report = validate_dataset(raw_frame)
        assert report.is_valid, report.summary()
        assert not report.errors

    def test_failure_modes_excluded_from_features(self) -> None:
        assert all(column not in RAW_FEATURE_COLUMNS for column in TARGET_DERIVED_COLUMNS)

    def test_class_imbalance_is_reported(self, raw_frame) -> None:
        report = generate_quality_report(raw_frame)
        assert report.target_balance is not None
        assert 0.0 < report.target_balance["positive_rate"] < 0.1

    def test_quality_report_is_reproducible(self, raw_frame) -> None:
        first = generate_quality_report(raw_frame).to_dict()
        second = generate_quality_report(raw_frame).to_dict()
        assert first == second

    def test_source_file_unchanged_by_validation(self, raw_frame) -> None:
        before = _DATASET.read_bytes()
        validate_dataset(raw_frame)
        assert _DATASET.read_bytes() == before
