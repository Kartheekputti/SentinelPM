"""Unit tests for the assembled feature pipeline."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from app.data.schema import TARGET_DERIVED_COLUMNS
from app.features.definitions import NUMERIC_FEATURE_COLUMNS
from app.features.engineering import extract_target, prepare_feature_frame
from app.features.pipeline import (
    build_feature_pipeline,
    feature_names,
    output_feature_names,
)


class TestFeaturePipeline:
    def test_output_columns_match_expected(self, valid_frame: pd.DataFrame) -> None:
        pipeline = build_feature_pipeline()
        features = prepare_feature_frame(valid_frame)
        transformed = pipeline.fit_transform(features)
        assert list(transformed.columns) == feature_names()

    def test_numeric_block_preserved(self, valid_frame: pd.DataFrame) -> None:
        pipeline = build_feature_pipeline()
        transformed = pipeline.fit_transform(prepare_feature_frame(valid_frame))
        assert list(transformed.columns[: len(NUMERIC_FEATURE_COLUMNS)]) == list(
            NUMERIC_FEATURE_COLUMNS
        )

    def test_transform_after_fit_keeps_columns(self, valid_frame: pd.DataFrame) -> None:
        pipeline = build_feature_pipeline()
        features = prepare_feature_frame(valid_frame)
        pipeline.fit(features)
        transformed = pipeline.transform(features)
        assert list(transformed.columns) == feature_names()
        assert len(transformed) == len(features)

    def test_training_and_inference_are_compatible(
        self, make_frame: Callable[..., pd.DataFrame]
    ) -> None:
        train = prepare_feature_frame(make_frame(8))
        pipeline = build_feature_pipeline().fit(train)
        infer = prepare_feature_frame(make_frame(5))
        transformed = pipeline.transform(infer)
        assert list(transformed.columns) == list(pipeline.get_feature_names_out())
        assert len(transformed) == 5

    def test_unknown_category_is_handled(self, make_frame: Callable[..., pd.DataFrame]) -> None:
        train = prepare_feature_frame(make_frame(6))
        pipeline = build_feature_pipeline().fit(train)
        unseen = prepare_feature_frame(make_frame(3))
        unseen["Type"] = "X"
        transformed = pipeline.transform(unseen)
        assert transformed.shape[0] == 3

    def test_no_target_derived_columns_leak(self, valid_frame: pd.DataFrame) -> None:
        pipeline = build_feature_pipeline()
        transformed = pipeline.fit_transform(prepare_feature_frame(valid_frame))
        for column in TARGET_DERIVED_COLUMNS:
            assert column not in transformed.columns
        assert "Machine failure" not in transformed.columns

    def test_target_is_never_used(self, make_frame: Callable[..., pd.DataFrame]) -> None:
        frame = make_frame(8)
        features = prepare_feature_frame(frame)
        pipeline = build_feature_pipeline()
        pipeline.fit(features)
        first = pipeline.transform(features)

        shuffled = frame.copy()
        shuffled["Machine failure"] = (
            shuffled["Machine failure"].sample(frac=1, random_state=0).to_numpy()
        )
        pipeline2 = build_feature_pipeline()
        pipeline2.fit(prepare_feature_frame(shuffled))
        second = pipeline2.transform(features)
        pd.testing.assert_frame_equal(first, second)

    def test_deterministic(self, valid_frame: pd.DataFrame) -> None:
        features = prepare_feature_frame(valid_frame)
        first = build_feature_pipeline().fit_transform(features)
        second = build_feature_pipeline().fit_transform(features)
        pd.testing.assert_frame_equal(first, second)

    def test_scaling_option_runs(self, valid_frame: pd.DataFrame) -> None:
        features = prepare_feature_frame(valid_frame)
        transformed = build_feature_pipeline(scale=True).fit_transform(features)
        assert list(transformed.columns) == feature_names()

    def test_output_feature_names_helper(self, valid_frame: pd.DataFrame) -> None:
        pipeline = build_feature_pipeline().fit(prepare_feature_frame(valid_frame))
        assert list(output_feature_names(pipeline)) == feature_names()

    def test_output_is_pandas(self, valid_frame: pd.DataFrame) -> None:
        transformed = build_feature_pipeline().fit_transform(prepare_feature_frame(valid_frame))
        assert isinstance(transformed, pd.DataFrame)


class TestExtractTargetWithPipeline:
    def test_target_alignment(self, valid_frame: pd.DataFrame) -> None:
        features = prepare_feature_frame(valid_frame)
        target = extract_target(valid_frame)
        transformed = build_feature_pipeline().fit_transform(features)
        assert len(transformed) == len(target)
