"""Assemble the full feature pipeline.

The pipeline combines stateless numeric feature engineering with one-hot
encoding of the categorical ``Type`` column. It is fit only on training data and
emits a named, consistently ordered feature matrix for both training and
inference.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.data.schema import CATEGORICAL_COLUMNS, SENSOR_COLUMNS
from app.features.definitions import (
    NUMERIC_FEATURE_COLUMNS,
    TYPE_ENCODING_PREFIX,
    safe_feature_name,
)
from app.features.engineering import FeatureNameSanitizer, SensorFeatureEngineer


def build_feature_pipeline(*, scale: bool = False) -> Pipeline:
    """Build the preprocessing pipeline mapping raw rows to a feature matrix.

    Args:
        scale: When ``True``, append a :class:`~sklearn.preprocessing.StandardScaler`.
            Scaling is only needed for scale-sensitive models such as logistic
            regression; tree-based models should use the default ``False``.

    Returns:
        A :class:`~sklearn.pipeline.Pipeline` that outputs a pandas dataframe
        with stable, model-safe column names.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("sensors", SensorFeatureEngineer(), list(SENSOR_COLUMNS)),
            (
                "type",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_COLUMNS),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    steps: list[tuple[str, object]] = [
        ("features", preprocessor),
        ("names", FeatureNameSanitizer()),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))

    pipeline = Pipeline(steps)
    pipeline.set_output(transform="pandas")
    return pipeline


def feature_names(
    categorical_categories: tuple[str, ...] = ("H", "L", "M"),
) -> list[str]:
    """Return the model-safe feature column names produced by the pipeline.

    Args:
        categorical_categories: Category values learned by the one-hot encoder,
            in the order produced by ``OneHotEncoder`` (sorted).

    Returns:
        The sanitised numeric feature names followed by the one-hot encoded
        names.
    """
    encoded = [f"{TYPE_ENCODING_PREFIX}{category}" for category in categorical_categories]
    return [safe_feature_name(name) for name in (*NUMERIC_FEATURE_COLUMNS, *encoded)]


def output_feature_names(pipeline: Pipeline) -> np.ndarray:
    """Return the feature names learned by a fitted pipeline.

    Args:
        pipeline: A fitted pipeline produced by :func:`build_feature_pipeline`.

    Returns:
        Array of output feature names.
    """
    return np.asarray(pipeline.get_feature_names_out(), dtype=object)
