"""Unit tests for model construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from app.core.exceptions import TrainingError
from app.training.config import ModelSpec
from app.training.models import (
    SUPPORTED_MODELS,
    build_model_pipeline,
    positive_class_weight,
)


class TestPositiveClassWeight:
    def test_ratio(self) -> None:
        assert positive_class_weight(np.array([0, 0, 0, 1])) == pytest.approx(3.0)

    def test_no_positives_defaults_to_one(self) -> None:
        assert positive_class_weight(np.array([0, 0, 0])) == 1.0


class TestBuildModelPipeline:
    @pytest.mark.parametrize("name", SUPPORTED_MODELS)
    def test_pipeline_steps(self, name: str) -> None:
        pipeline = build_model_pipeline(ModelSpec(name=name, scale=(name == "logistic_regression")))
        assert isinstance(pipeline, Pipeline)
        assert list(pipeline.named_steps) == ["features", "model"]

    def test_unsupported_model_raises(self) -> None:
        with pytest.raises(TrainingError, match="Unsupported"):
            build_model_pipeline(ModelSpec(name="svm"))

    @pytest.mark.parametrize("name", ["dummy", "logistic_regression"])
    def test_fit_and_predict(self, name: str, make_frame) -> None:
        target = [1 if index % 5 == 0 else 0 for index in range(60)]
        frame = make_frame(60, **{"Machine failure": target})
        pipeline = build_model_pipeline(ModelSpec(name=name, scale=(name == "logistic_regression")))
        pipeline.fit(frame, np.asarray(target))
        proba = pipeline.predict_proba(frame)
        assert proba.shape == (60, 2)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_xgboost_accepts_scale_pos_weight(self, make_frame) -> None:
        target = [1 if index % 5 == 0 else 0 for index in range(60)]
        frame = make_frame(60, **{"Machine failure": target})
        pipeline = build_model_pipeline(ModelSpec(name="xgboost"), scale_pos_weight=4.0)
        pipeline.fit(frame, np.asarray(target))
        assert hasattr(pipeline.named_steps["model"], "predict_proba")

    def test_feature_matrix_columns_are_named(self, make_frame) -> None:
        pipeline = build_model_pipeline(ModelSpec(name="dummy"))
        transformed = pipeline.named_steps["features"].fit_transform(
            frame_columns_only(make_frame(6))
        )
        assert isinstance(transformed, pd.DataFrame)


def frame_columns_only(frame: pd.DataFrame) -> pd.DataFrame:
    from app.features.engineering import prepare_feature_frame

    return prepare_feature_frame(frame)
