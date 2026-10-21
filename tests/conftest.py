"""Shared test fixtures.

The factory below builds small synthetic frames that match the AI4I schema.
These frames exist only to exercise validators and transformers; they are never
used to produce or imply model performance.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest

from app.data.schema import (
    BINARY_COLUMNS,
    CATEGORICAL_COLUMNS,
    ID_COLUMNS,
    SENSOR_COLUMNS,
    TARGET_COLUMN,
)


def _build_frame(n_rows: int, overrides: dict[str, Any]) -> pd.DataFrame:
    data: dict[str, Any] = {
        "UDI": list(range(1, n_rows + 1)),
        "Product ID": [f"M{14860 + i}" for i in range(n_rows)],
        "Type": [["L", "M", "H"][i % 3] for i in range(n_rows)],
        "Air temperature [K]": [298.1 + 0.1 * i for i in range(n_rows)],
        "Process temperature [K]": [308.6 + 0.1 * i for i in range(n_rows)],
        "Rotational speed [rpm]": [1400 + 10 * i for i in range(n_rows)],
        "Torque [Nm]": [40.0 + 0.5 * i for i in range(n_rows)],
        "Tool wear [min]": [i for i in range(n_rows)],
        "Machine failure": [0] * (n_rows - 1) + [1],
    }
    for column in ("TWF", "HDF", "PWF", "OSF", "RNF"):
        data[column] = [0] * n_rows

    frame = pd.DataFrame(data)
    for column, values in overrides.items():
        frame[column] = values
    return frame


@pytest.fixture(scope="session")
def make_frame() -> Callable[..., pd.DataFrame]:
    """Return a factory that builds a valid raw frame of ``n_rows`` rows."""

    def _make(n_rows: int = 6, **overrides: Any) -> pd.DataFrame:
        return _build_frame(n_rows, overrides)

    return _make


@pytest.fixture
def valid_frame(make_frame: Callable[..., pd.DataFrame]) -> pd.DataFrame:
    return make_frame(6)


@pytest.fixture
def sample_trained_rf(valid_frame: pd.DataFrame) -> Any:
    """Return a fitted Random Forest pipeline on small sample data."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline

    from app.features.pipeline import build_feature_pipeline

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


__all__ = [
    "BINARY_COLUMNS",
    "CATEGORICAL_COLUMNS",
    "ID_COLUMNS",
    "SENSOR_COLUMNS",
    "TARGET_COLUMN",
]
