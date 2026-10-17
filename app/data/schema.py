"""Schema definition for the AI4I 2020 predictive maintenance dataset.

The schema is expressed twice: as plain constants (used by ingestion, quality
reporting and feature engineering) and as a :mod:`pandera` schema (used for
value-level validation). Keeping both next to each other ensures the columns,
bounds and categories never drift apart.

Column reference (verified against ``ai4i2020.csv``):
    - ``UDI``: row identifier (1..10000).
    - ``Product ID``: identifier such as ``M14860``; the leading letter is
      redundant with ``Type``.
    - ``Type``: product quality variant, one of ``L``, ``M`` or ``H``.
    - ``Air temperature [K]``: ambient temperature.
    - ``Process temperature [K]``: process temperature.
    - ``Rotational speed [rpm]``: spindle speed.
    - ``Torque [Nm]``: torque applied.
    - ``Tool wear [min]``: accumulated tool wear.
    - ``Machine failure``: target label (1 = failure).
    - ``TWF``, ``HDF``, ``PWF``, ``OSF``, ``RNF``: individual failure-mode
      indicators. These are **derived from the target** and are therefore
      excluded from model inputs to prevent leakage.
"""

from __future__ import annotations

from typing import Final

import pandera.pandas as pa

TARGET_COLUMN: Final[str] = "Machine failure"

#: Identifier columns. Kept in the raw frame but excluded from model features.
ID_COLUMNS: Final[tuple[str, ...]] = ("UDI", "Product ID")

#: Individual failure-mode indicators reported alongside the target.
FAILURE_MODE_COLUMNS: Final[tuple[str, ...]] = ("TWF", "HDF", "PWF", "OSF", "RNF")

#: Columns that are derived from the target and must never be model inputs.
TARGET_DERIVED_COLUMNS: Final[tuple[str, ...]] = FAILURE_MODE_COLUMNS

#: Continuous machine measurements.
SENSOR_COLUMNS: Final[tuple[str, ...]] = (
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
)

#: Categorical operating descriptors.
CATEGORICAL_COLUMNS: Final[tuple[str, ...]] = ("Type",)

#: Columns permitted as model inputs before feature engineering.
RAW_FEATURE_COLUMNS: Final[tuple[str, ...]] = SENSOR_COLUMNS + CATEGORICAL_COLUMNS

#: Every column expected in a valid raw dataset, in canonical order.
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    ID_COLUMNS + CATEGORICAL_COLUMNS + SENSOR_COLUMNS + (TARGET_COLUMN,) + FAILURE_MODE_COLUMNS
)

#: Physical plausibility bounds used for out-of-range detection. Bounds are
#: deliberately wider than the observed operating range so that drift raises a
#: warning instead of silently passing, while impossible values are rejected.
NUMERIC_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "Air temperature [K]": (250.0, 350.0),
    "Process temperature [K]": (250.0, 350.0),
    "Rotational speed [rpm]": (0.0, 10_000.0),
    "Torque [Nm]": (0.0, 200.0),
    "Tool wear [min]": (0.0, 1_000.0),
}

#: Allowed product quality variants.
TYPE_CATEGORIES: Final[tuple[str, ...]] = ("L", "M", "H")

#: Columns constrained to the set {0, 1}.
BINARY_COLUMNS: Final[tuple[str, ...]] = (TARGET_COLUMN,) + FAILURE_MODE_COLUMNS

#: Filename expected inside the raw data directory by default.
DEFAULT_RAW_FILENAME: Final[str] = "ai4i2020.csv"


def build_raw_schema(
    *,
    include_target: bool = True,
    allow_null: bool = False,
) -> pa.DataFrameSchema:
    """Build the Pandera schema for the raw AI4I dataset.

    Numeric sensor columns intentionally do not pin an exact dtype so that both
    integer and floating point representations validate; dtype conformance is
    checked separately in :mod:`app.data.validation` to produce clearer errors.

    Args:
        include_target: Include the target and failure-mode columns. Set to
            ``False`` when validating a feature-only frame.
        allow_null: Permit missing values at the schema level. Set to ``True``
            when missing-value handling is disabled so the validator does not
            report duplicate problems.

    Returns:
        A :class:`pandera.DataFrameSchema` with range, category and value
        constraints. ``strict=False`` allows additional columns (reported as a
        warning by the validator) instead of failing outright.
    """
    nullable = allow_null
    columns: dict[str, pa.Column] = {
        "UDI": pa.Column(int, pa.Check.ge(1), nullable=nullable),
        "Product ID": pa.Column(str, nullable=nullable),
        "Type": pa.Column(str, pa.Check.isin(TYPE_CATEGORIES), nullable=nullable),
    }

    for name, (low, high) in NUMERIC_BOUNDS.items():
        columns[name] = pa.Column(
            checks=pa.Check.in_range(low, high),
            nullable=nullable,
        )

    for name in BINARY_COLUMNS:
        if not include_target and name == TARGET_COLUMN:
            continue
        columns[name] = pa.Column(int, pa.Check.isin([0, 1]), nullable=nullable)

    return pa.DataFrameSchema(columns=columns, strict=False)
