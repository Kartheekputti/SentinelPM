"""Reusable, leakage-safe feature transformations.

The transformations here are deterministic and stateless with respect to the
target: they can be fitted on training data and applied unchanged at inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from app.core.exceptions import FeatureError
from app.core.logging import get_logger
from app.data.schema import (
    CATEGORICAL_COLUMNS,
    RAW_FEATURE_COLUMNS,
    SENSOR_COLUMNS,
    TARGET_COLUMN,
    TARGET_DERIVED_COLUMNS,
)
from app.features.definitions import (
    ENGINEERED_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    ROTATIONAL_POWER,
    TEMPERATURE_DIFFERENCE,
    safe_feature_name,
)

logger = get_logger(__name__)

_RPM_TO_RAD_PER_SEC = 2.0 * np.pi / 60.0


def prepare_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Select and validate the columns permitted as model inputs.

    Args:
        frame: Raw dataframe containing at least the feature columns.

    Returns:
        A new dataframe containing only :data:`RAW_FEATURE_COLUMNS`, in order.

    Raises:
        FeatureError: If ``frame`` is not a dataframe or required columns are
            missing.
    """
    if not isinstance(frame, pd.DataFrame):
        msg = f"prepare_feature_frame expects a DataFrame, got {type(frame).__name__}"
        raise FeatureError(msg)

    missing = [column for column in RAW_FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        msg = f"Missing required feature columns: {missing}"
        raise FeatureError(msg)

    return frame.loc[:, list(RAW_FEATURE_COLUMNS)].copy()


def extract_target(frame: pd.DataFrame, column: str = TARGET_COLUMN) -> pd.Series:
    """Return the target series as integers.

    Args:
        frame: Raw dataframe containing the target column.
        column: Name of the target column.

    Returns:
        The target as an integer series named ``column``.

    Raises:
        FeatureError: If the target column is absent.
    """
    if column not in frame.columns:
        msg = f"Target column not found: {column!r}"
        raise FeatureError(msg)
    return frame[column].astype(int).rename(column)


def _validate_numeric_inputs(frame: pd.DataFrame) -> None:
    missing = [column for column in SENSOR_COLUMNS if column not in frame.columns]
    if missing:
        msg = f"Missing required sensor columns: {missing}"
        raise FeatureError(msg)
    non_numeric = [
        column for column in SENSOR_COLUMNS if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        msg = f"Sensor columns must be numeric: {non_numeric}"
        raise FeatureError(msg)


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add derived sensor features without mutating the input.

    Adds ``temperature_difference [K]`` and ``rotational_power [W]``.

    Args:
        frame: Dataframe containing the required sensor columns.

    Returns:
        A new dataframe with the original columns plus the derived features.

    Raises:
        FeatureError: If ``frame`` is not a dataframe, or required sensor
            columns are missing or non-numeric.
    """
    if not isinstance(frame, pd.DataFrame):
        msg = f"add_engineered_features expects a DataFrame, got {type(frame).__name__}"
        raise FeatureError(msg)
    _validate_numeric_inputs(frame)
    result = frame.copy()
    result[TEMPERATURE_DIFFERENCE] = (
        result["Process temperature [K]"] - result["Air temperature [K]"]
    )
    result[ROTATIONAL_POWER] = (
        result["Torque [Nm]"] * result["Rotational speed [rpm]"] * _RPM_TO_RAD_PER_SEC
    )
    return result


class SensorFeatureEngineer(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer adding derived numeric sensor features.

    The transformer is stateless; ``fit`` validates the input and returns
    ``self`` so it composes cleanly inside a :class:`~sklearn.pipeline.Pipeline`.
    Output columns are always emitted in :data:`NUMERIC_FEATURE_COLUMNS` order.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> SensorFeatureEngineer:
        """Validate the input schema and return ``self``.

        Args:
            X: Dataframe of sensor columns.
            y: Ignored; present for API compatibility.

        Returns:
            ``self``.
        """
        _validate_numeric_inputs(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the numeric feature block with derived features appended.

        Args:
            X: Dataframe of sensor columns.

        Returns:
            Dataframe with columns in :data:`NUMERIC_FEATURE_COLUMNS` order.

        Raises:
            FeatureError: If required sensor columns are missing or non-numeric.
        """
        engineered = add_engineered_features(X)
        return engineered.loc[:, list(NUMERIC_FEATURE_COLUMNS)]

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return output feature names in canonical order."""
        return np.asarray(NUMERIC_FEATURE_COLUMNS, dtype=object)


class FeatureNameSanitizer(BaseEstimator, TransformerMixin):
    """Rename dataframe columns to model-safe identifiers.

    Downstream estimators such as XGBoost require feature names that omit
    ``[``, ``]`` and ``<``. This stateless transformer keeps the mapping in one
    place so training and inference produce identical column names.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FeatureNameSanitizer:
        """Record the sanitised output names and return ``self``.

        Args:
            X: Feature dataframe produced by the previous pipeline step.
            y: Ignored; present for API compatibility.

        Returns:
            ``self``.
        """
        columns = getattr(X, "columns", None)
        self.columns_: list[str] | None = (
            [safe_feature_name(str(column)) for column in columns] if columns is not None else None
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``X`` with sanitised column names.

        Args:
            X: Feature dataframe.

        Returns:
            A new dataframe with sanitised column names.
        """
        result = X.copy()
        result.columns = [safe_feature_name(str(column)) for column in result.columns]
        return result

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return sanitised feature names.

        Args:
            input_features: Names from the previous step, used when the
                transformer has not been fitted.

        Returns:
            Array of sanitised feature names.

        Raises:
            FeatureError: If names are unavailable because the transformer was
                neither fitted on a dataframe nor given ``input_features``.
        """
        stored = getattr(self, "columns_", None)
        if stored is not None:
            return np.asarray(stored, dtype=object)
        if input_features is None:
            msg = "input_features is required to build sanitised feature names"
            raise FeatureError(msg)
        return np.asarray([safe_feature_name(str(name)) for name in input_features], dtype=object)


__all__ = [
    "CATEGORICAL_COLUMNS",
    "ENGINEERED_FEATURE_COLUMNS",
    "NUMERIC_FEATURE_COLUMNS",
    "RAW_FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "TARGET_DERIVED_COLUMNS",
    "FeatureNameSanitizer",
    "SensorFeatureEngineer",
    "add_engineered_features",
    "extract_target",
    "prepare_feature_frame",
]
