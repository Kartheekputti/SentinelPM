"""Unit tests for the SHAP explainability module (M05)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.core.exceptions import ExplainabilityError
from app.explainability.explainer import (
    CAUSALITY_DISCLAIMER,
    ModelExplainer,
)
from app.explainability.report import (
    generate_global_explanation,
    save_explanation_report,
)
from app.features.pipeline import build_feature_pipeline


@pytest.fixture
def sample_trained_rf(valid_frame: pd.DataFrame) -> Pipeline:
    """Return a fitted Random Forest pipeline on small sample data."""
    features = valid_frame.copy()
    y = [0] * (len(features) - 1) + [1]
    pipeline = Pipeline(
        [
            ("features", build_feature_pipeline(scale=False)),
            ("model", RandomForestClassifier(n_estimators=10, random_state=42)),
        ]
    )
    pipeline.fit(features, y)
    return pipeline


@pytest.fixture
def sample_trained_lr(valid_frame: pd.DataFrame) -> Pipeline:
    """Return a fitted Logistic Regression pipeline on small sample data."""
    features = valid_frame.copy()
    y = [0] * (len(features) - 1) + [1]
    pipeline = Pipeline(
        [
            ("features", build_feature_pipeline(scale=True)),
            ("model", LogisticRegression(random_state=42)),
        ]
    )
    pipeline.fit(features, y)
    return pipeline


@pytest.fixture
def sample_dummy_pipeline(valid_frame: pd.DataFrame) -> Pipeline:
    """Return a fitted DummyClassifier pipeline."""
    features = valid_frame.copy()
    y = [0] * len(features)
    pipeline = Pipeline(
        [
            ("features", build_feature_pipeline(scale=False)),
            ("model", DummyClassifier(strategy="prior")),
        ]
    )
    pipeline.fit(features, y)
    return pipeline


def test_explainer_rejects_dummy_classifier(sample_dummy_pipeline: Pipeline) -> None:
    """Ensure baseline DummyClassifier cannot be explained and raises ExplainabilityError."""
    with pytest.raises(ExplainabilityError, match="DummyClassifier"):
        ModelExplainer(sample_dummy_pipeline)


def test_explainer_rejects_malformed_pipeline() -> None:
    """Ensure malformed pipeline raises ExplainabilityError."""
    with pytest.raises(ExplainabilityError, match="named_steps"):
        ModelExplainer(object())  # type: ignore[arg-type]


def test_explain_record_rf(sample_trained_rf: Pipeline, valid_frame: pd.DataFrame) -> None:
    """Verify individual explanation structure and feature alignment for Random Forest."""
    explainer = ModelExplainer(sample_trained_rf, model_name="random_forest", model_version="1.0.0")

    record = valid_frame.iloc[[0]]
    result = explainer.explain_record(record, prediction_id="test-pred-123")

    assert result.prediction_id == "test-pred-123"
    assert result.model_name == "random_forest"
    assert result.model_version == "1.0.0"
    assert 0.0 <= result.predicted_probability <= 1.0
    assert len(result.contributions) == len(explainer.feature_names)
    assert CAUSALITY_DISCLAIMER in result.summary_text
    assert result.causality_disclaimer == CAUSALITY_DISCLAIMER

    # Check serialization
    as_dict = result.to_dict()
    assert as_dict["prediction_id"] == "test-pred-123"
    assert "contributions" in as_dict
    assert len(as_dict["contributions"]) == len(explainer.feature_names)


def test_explain_record_dict_input(sample_trained_rf: Pipeline, valid_frame: pd.DataFrame) -> None:
    """Verify explain_record accepts a dictionary."""
    explainer = ModelExplainer(sample_trained_rf)
    record_dict = valid_frame.iloc[0].to_dict()

    result = explainer.explain_record(record_dict)
    assert result.predicted_probability >= 0.0
    assert len(result.contributions) > 0


def test_explain_record_empty_raises(sample_trained_rf: Pipeline) -> None:
    """Verify passing an empty frame raises ExplainabilityError."""
    explainer = ModelExplainer(sample_trained_rf)
    with pytest.raises(ExplainabilityError, match="empty"):
        explainer.explain_record(pd.DataFrame())


def test_explain_lr_model(sample_trained_lr: Pipeline, valid_frame: pd.DataFrame) -> None:
    """Verify LinearExplainer works with LogisticRegression and background data."""
    explainer = ModelExplainer(
        sample_trained_lr,
        model_name="logistic_regression",
        background_data=valid_frame,
    )
    result = explainer.explain_record(valid_frame.iloc[[0]])
    assert 0.0 <= result.predicted_probability <= 1.0
    assert len(result.contributions) == len(explainer.feature_names)


def test_generate_global_explanation(
    sample_trained_rf: Pipeline, valid_frame: pd.DataFrame
) -> None:
    """Verify global feature importance generation."""
    explainer = ModelExplainer(sample_trained_rf)
    global_exp = generate_global_explanation(explainer, valid_frame, max_samples=10)

    assert global_exp.n_samples == len(valid_frame)
    assert len(global_exp.ranking) == len(explainer.feature_names)
    # Check ranking is sorted descending
    scores = [r["mean_abs_shap"] for r in global_exp.ranking]
    assert scores == sorted(scores, reverse=True)
    assert CAUSALITY_DISCLAIMER in global_exp.summary_text


def test_save_explanation_report(
    sample_trained_rf: Pipeline, valid_frame: pd.DataFrame, tmp_path: Path
) -> None:
    """Verify explanation report saving to JSON."""
    explainer = ModelExplainer(sample_trained_rf)
    global_exp = generate_global_explanation(explainer, valid_frame)
    output_path = tmp_path / "shap_report.json"

    saved_path = save_explanation_report(global_exp, output_path)
    assert saved_path.exists()

    with saved_path.open(encoding="utf-8") as f:
        data = json.load(f)
    assert data["n_samples"] == len(valid_frame)
    assert "ranking" in data
