"""Feature definitions and engineering.

Features are documented here with their units and formulas so that training and
inference stay consistent and reviewers can audit the transformations.

Leakage policy
--------------
Only ``RAW_FEATURE_COLUMNS`` may enter the feature matrix. Identifier columns
(``UDI``, ``Product ID``), the target (``Machine failure``) and the
target-derived failure-mode indicators (``TWF``, ``HDF``, ``PWF``, ``OSF``,
``RNF``) are never used as inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.data.schema import RAW_FEATURE_COLUMNS, SENSOR_COLUMNS, TARGET_DERIVED_COLUMNS


@dataclass(frozen=True)
class FeatureDefinition:
    """Human-readable definition of a derived feature."""

    name: str
    unit: str
    formula: str
    rationale: str


TEMPERATURE_DIFFERENCE: str = "temperature_difference [K]"
ROTATIONAL_POWER: str = "rotational_power [W]"

#: Prefix used by the one-hot encoder for the ``Type`` column
#: (``OneHotEncoder`` emits ``<column>_<category>``).
TYPE_ENCODING_PREFIX: str = "Type_"

#: Derived numeric features in their canonical output order.
ENGINEERED_FEATURE_COLUMNS: tuple[str, ...] = (TEMPERATURE_DIFFERENCE, ROTATIONAL_POWER)

#: Documentation for every derived feature.
FEATURE_DEFINITIONS: dict[str, FeatureDefinition] = {
    TEMPERATURE_DIFFERENCE: FeatureDefinition(
        name=TEMPERATURE_DIFFERENCE,
        unit="K",
        formula="Process temperature [K] - Air temperature [K]",
        rationale=(
            "Heat dissipation failure (HDF) occurs when the process-to-air "
            "temperature gap is too small. The difference is more informative "
            "than either temperature alone."
        ),
    ),
    ROTATIONAL_POWER: FeatureDefinition(
        name=ROTATIONAL_POWER,
        unit="W",
        formula="Torque [Nm] * (2 * pi * Rotational speed [rpm] / 60)",
        rationale=(
            "Power failure (PWF) depends on the combined load of torque and "
            "rotational speed, which neither raw sensor captures on its own."
        ),
    ),
}

#: Output order of the numeric sensor block: raw sensors followed by derived.
NUMERIC_FEATURE_COLUMNS: tuple[str, ...] = (*SENSOR_COLUMNS, *ENGINEERED_FEATURE_COLUMNS)

#: Columns that are explicitly forbidden as model inputs.
FORBIDDEN_FEATURE_COLUMNS: tuple[str, ...] = TARGET_DERIVED_COLUMNS


def documented_feature_names() -> tuple[str, ...]:
    """Return all columns that may appear in a feature matrix.

    Returns:
        Raw feature columns followed by the derived numeric features.
    """
    return (*RAW_FEATURE_COLUMNS, *ENGINEERED_FEATURE_COLUMNS)
