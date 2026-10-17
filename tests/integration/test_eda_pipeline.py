"""Integration tests for feature engineering on the real dataset."""

from __future__ import annotations

import pytest
from sklearn.model_selection import train_test_split

from app.core.config import get_settings
from app.data.loader import load_raw_dataset
from app.data.schema import DEFAULT_RAW_FILENAME, TARGET_COLUMN, TARGET_DERIVED_COLUMNS
from app.features.eda import generate_eda_report
from app.features.engineering import extract_target, prepare_feature_frame
from app.features.pipeline import build_feature_pipeline

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
class TestFeaturePipelineIntegration:
    def test_fit_on_train_only_and_transform_test(self, raw_frame) -> None:
        features = prepare_feature_frame(raw_frame)
        target = extract_target(raw_frame)
        x_train, x_test, y_train, _ = train_test_split(
            features, target, test_size=0.2, stratify=target, random_state=42
        )
        pipeline = build_feature_pipeline().fit(x_train)
        train_matrix = pipeline.transform(x_train)
        test_matrix = pipeline.transform(x_test)

        assert list(train_matrix.columns) == list(test_matrix.columns)
        assert len(train_matrix) == len(y_train)
        assert len(test_matrix) == len(x_test)

    def test_no_target_derived_columns_in_matrix(self, raw_frame) -> None:
        pipeline = build_feature_pipeline().fit(prepare_feature_frame(raw_frame))
        matrix = pipeline.transform(prepare_feature_frame(raw_frame))
        for column in (TARGET_COLUMN, *TARGET_DERIVED_COLUMNS):
            assert column not in matrix.columns

    def test_eda_report_reproducible_on_real_data(self, raw_frame) -> None:
        first = generate_eda_report(raw_frame).to_dict()
        second = generate_eda_report(raw_frame).to_dict()
        assert first == second
        assert first["class_balance"]["n_positive"] > 0

    def test_engineered_features_present(self, raw_frame) -> None:
        report = generate_eda_report(raw_frame)
        assert "temperature_difference [K]" in report.numeric_summary
        assert "rotational_power [W]" in report.numeric_summary
