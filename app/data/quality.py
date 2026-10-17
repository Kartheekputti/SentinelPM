"""Data-quality reporting.

Produces a reproducible, JSON-serialisable summary of a dataset: dimensions,
missing values, duplicates, per-column numeric statistics and target balance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.core.exceptions import DataValidationError
from app.core.logging import get_logger
from app.data.schema import REQUIRED_COLUMNS, TARGET_COLUMN

logger = get_logger(__name__)


@dataclass(frozen=True)
class NumericSummary:
    """Descriptive statistics for a single numeric column."""

    count: int
    missing: int
    unique: int
    mean: float
    std: float
    minimum: float
    q25: float
    median: float
    q75: float
    maximum: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "missing": self.missing,
            "unique": self.unique,
            "mean": self.mean,
            "std": self.std,
            "min": self.minimum,
            "q25": self.q25,
            "median": self.median,
            "q75": self.q75,
            "max": self.maximum,
        }


@dataclass
class DataQualityReport:
    """Aggregate quality summary for a dataset."""

    n_rows: int
    n_columns: int
    memory_usage_mb: float
    duplicate_rows: int
    missing_values: dict[str, int] = field(default_factory=dict)
    constant_columns: list[str] = field(default_factory=list)
    numeric_summary: dict[str, NumericSummary] = field(default_factory=dict)
    target_balance: dict[str, Any] | None = None

    @property
    def total_missing(self) -> int:
        return int(sum(self.missing_values.values()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "memory_usage_mb": self.memory_usage_mb,
            "duplicate_rows": self.duplicate_rows,
            "total_missing": self.total_missing,
            "missing_values": self.missing_values,
            "constant_columns": self.constant_columns,
            "target_balance": self.target_balance,
            "numeric_summary": {
                column: summary.to_dict() for column, summary in self.numeric_summary.items()
            },
        }

    def summary(self) -> str:
        lines = [
            "Data quality report",
            f"  shape: {self.n_rows} rows x {self.n_columns} columns",
            f"  memory: {self.memory_usage_mb:.3f} MB",
            f"  duplicate rows: {self.duplicate_rows}",
            f"  missing values: {self.total_missing} across {len(self.missing_values)} column(s)",
        ]
        if self.constant_columns:
            lines.append(f"  constant columns: {self.constant_columns}")
        if self.target_balance is not None:
            balance = self.target_balance
            lines.append(
                "  target balance: "
                f"positive={balance['n_positive']} negative={balance['n_negative']} "
                f"positive_rate={balance['positive_rate']:.4f} "
                f"imbalance_ratio={balance['imbalance_ratio']:.2f}"
            )
        return "\n".join(lines)


def generate_quality_report(
    frame: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
) -> DataQualityReport:
    """Compute a data-quality summary for ``frame``.

    Args:
        frame: The dataframe to summarise.
        target_column: Column holding the binary classification target.

    Returns:
        A :class:`DataQualityReport`.

    Raises:
        DataValidationError: If ``frame`` is empty or the target column is
            missing.
    """
    if frame.empty:
        msg = "Cannot generate a quality report for an empty dataframe"
        raise DataValidationError(msg)

    missing_values = {
        column: int(count)
        for column, count in frame.isna().sum().items()
        if int(count) > 0
    }

    constant_columns = [
        column
        for column in frame.columns
        if frame[column].nunique(dropna=False) <= 1
    ]

    numeric_summary: dict[str, NumericSummary] = {}
    for column in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        series = frame[column]
        numeric_summary[column] = NumericSummary(
            count=int(series.count()),
            missing=int(series.isna().sum()),
            unique=int(series.nunique(dropna=True)),
            mean=float(series.mean()),
            std=float(series.std()),
            minimum=float(series.min()),
            q25=float(series.quantile(0.25)),
            median=float(series.median()),
            q75=float(series.quantile(0.75)),
            maximum=float(series.max()),
        )

    target_balance = _target_balance(frame, target_column)

    report = DataQualityReport(
        n_rows=int(frame.shape[0]),
        n_columns=int(frame.shape[1]),
        memory_usage_mb=float(frame.memory_usage(deep=True).sum()) / (1024**2),
        duplicate_rows=int(frame.duplicated().sum()),
        missing_values=missing_values,
        constant_columns=constant_columns,
        numeric_summary=numeric_summary,
        target_balance=target_balance,
    )
    logger.info("Generated quality report for %d rows", report.n_rows)
    return report


def _target_balance(frame: pd.DataFrame, target_column: str) -> dict[str, Any] | None:
    if target_column not in frame.columns:
        return None
    counts = frame[target_column].value_counts()
    n_positive = int(counts.get(1, 0))
    n_negative = int(counts.get(0, 0))
    total = n_positive + n_negative
    positive_rate = n_positive / total if total else 0.0
    imbalance_ratio = (n_negative / n_positive) if n_positive else float("inf")
    return {
        "positive_label": 1,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "positive_rate": positive_rate,
        "imbalance_ratio": imbalance_ratio,
        "n_unique": int(frame[target_column].nunique()),
    }


def ensure_known_columns(frame: pd.DataFrame, columns: tuple[str, ...] = REQUIRED_COLUMNS) -> None:
    """Raise if any expected column is absent from ``frame``.

    Args:
        frame: The dataframe to inspect.
        columns: Columns that must be present.

    Raises:
        DataValidationError: If one or more columns are missing.
    """
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        msg = f"Dataset is missing required columns: {missing}"
        raise DataValidationError(msg)
