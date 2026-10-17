"""Exploratory data analysis.

Provides reproducible summaries of class imbalance, sensor distributions,
feature/target associations, outliers and inter-feature correlations. Results
feed the EDA report artifact and inform model selection; they are never used as
model inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.core.logging import get_logger
from app.data.schema import CATEGORICAL_COLUMNS, TARGET_COLUMN
from app.features.definitions import NUMERIC_FEATURE_COLUMNS
from app.features.engineering import add_engineered_features

logger = get_logger(__name__)

IQR_MULTIPLIER = 1.5


def _clean(value: float) -> float | None:
    if value is None:
        return None
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def class_balance(frame: pd.DataFrame, target: str = TARGET_COLUMN) -> dict[str, Any]:
    """Summarise the binary target distribution.

    Args:
        frame: Dataframe containing the target column.
        target: Target column name.

    Returns:
        Dictionary with positive/negative counts, positive rate and imbalance
        ratio.
    """
    counts = frame[target].value_counts()
    n_positive = int(counts.get(1, 0))
    n_negative = int(counts.get(0, 0))
    total = n_positive + n_negative
    return {
        "n_positive": n_positive,
        "n_negative": n_negative,
        "positive_rate": n_positive / total if total else 0.0,
        "imbalance_ratio": (n_negative / n_positive) if n_positive else None,
    }


def numeric_distributions(
    frame: pd.DataFrame,
    columns: tuple[str, ...] = NUMERIC_FEATURE_COLUMNS,
) -> dict[str, dict[str, float | None]]:
    """Return descriptive statistics per numeric column.

    Args:
        frame: Dataframe with a column the same name as the target.
        columns: Numeric columns to summarise.

    Returns:
        Mapping of column name to statistics including skewness.
    """
    summary: dict[str, dict[str, float | None]] = {}
    for column in columns:
        if column not in frame.columns:
            continue
        series = frame[column]
        summary[column] = {
            "count": int(series.count()),
            "mean": _clean(series.mean()),
            "std": _clean(series.std()),
            "min": _clean(series.min()),
            "q25": _clean(series.quantile(0.25)),
            "median": _clean(series.median()),
            "q75": _clean(series.quantile(0.75)),
            "max": _clean(series.max()),
            "skew": _clean(series.skew()),
        }
    return summary


def feature_target_correlation(
    frame: pd.DataFrame,
    target: str = TARGET_COLUMN,
    columns: tuple[str, ...] = NUMERIC_FEATURE_COLUMNS,
) -> dict[str, float | None]:
    """Compute Pearson correlation of each numeric feature with the target.

    For a binary target this equals the point-biserial correlation.

    Args:
        frame: Dataframe containing the feature and target columns.
        target: Target column name.
        columns: Numeric columns to correlate.

    Returns:
        Mapping of column name to signed correlation with the target.
    """
    result: dict[str, float | None] = {}
    for column in columns:
        if column not in frame.columns or column == target:
            continue
        if frame[column].nunique(dropna=True) <= 1 or frame[target].nunique(dropna=True) <= 1:
            result[column] = None
            continue
        result[column] = _clean(frame[column].corr(frame[target]))
    return result


def failure_rate_by_type(
    frame: pd.DataFrame,
    target: str = TARGET_COLUMN,
    category: str = CATEGORICAL_COLUMNS[0],
) -> dict[str, dict[str, Any]]:
    """Compute failure counts and rates for each category value.

    Args:
        frame: Dataframe containing the category and target columns.
        target: Target column name.
        category: Categorical column name.

    Returns:
        Mapping of category value to counts and positive rate.
    """
    if category not in frame.columns:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for value, group in frame.groupby(category, observed=True):
        n_positive = int(group[target].sum())
        n_total = len(group)
        result[str(value)] = {
            "n_total": n_total,
            "n_positive": n_positive,
            "positive_rate": (n_positive / n_total) if n_total else 0.0,
        }
    return result


def detect_outliers_iqr(
    frame: pd.DataFrame,
    columns: tuple[str, ...] = NUMERIC_FEATURE_COLUMNS,
    multiplier: float = IQR_MULTIPLIER,
) -> dict[str, dict[str, Any]]:
    """Count IQR-based outliers per numeric column.

    Outliers are reported, never removed; the caller decides how to treat them.

    Args:
        frame: Dataframe containing the numeric columns.
        columns: Numeric columns to inspect.
        multiplier: IQR multiplier controlling the fence width.

    Returns:
        Mapping of column name to outlier count, fraction and fence bounds.
    """
    result: dict[str, dict[str, Any]] = {}
    for column in columns:
        if column not in frame.columns:
            continue
        series = frame[column]
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25
        lower = q25 - multiplier * iqr
        upper = q75 + multiplier * iqr
        mask = (series < lower) | (series > upper)
        count = int(mask.sum())
        result[column] = {
            "count": count,
            "fraction": (count / len(series)) if len(series) else 0.0,
            "lower_fence": _clean(lower),
            "upper_fence": _clean(upper),
        }
    return result


def top_feature_correlations(
    frame: pd.DataFrame,
    columns: tuple[str, ...] = NUMERIC_FEATURE_COLUMNS,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Return the strongest absolute pairwise feature correlations.

    Args:
        frame: Dataframe containing the numeric columns.
        columns: Numeric columns to correlate.
        top_n: Number of pairs to return.

    Returns:
        List of ``{feature_a, feature_b, correlation}`` dictionaries sorted by
        descending absolute correlation.
    """
    available = [column for column in columns if column in frame.columns]
    available = [column for column in available if frame[column].nunique(dropna=True) > 1]
    if len(available) < 2:
        return []
    corr = frame[available].corr()
    pairs: list[dict[str, Any]] = []
    for i, feature_a in enumerate(available):
        for feature_b in available[i + 1 :]:
            pairs.append(
                {
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "correlation": _clean(corr.loc[feature_a, feature_b]),
                }
            )
    pairs.sort(key=lambda item: abs(item["correlation"] or 0.0), reverse=True)
    return pairs[:top_n]


@dataclass
class EDAReport:
    """Container for the exploratory analysis results."""

    n_rows: int
    class_balance: dict[str, Any]
    numeric_summary: dict[str, dict[str, float | None]] = field(default_factory=dict)
    feature_target_correlation: dict[str, float | None] = field(default_factory=dict)
    failure_rate_by_type: dict[str, dict[str, Any]] = field(default_factory=dict)
    outlier_summary: dict[str, dict[str, Any]] = field(default_factory=dict)
    top_feature_correlations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "class_balance": self.class_balance,
            "numeric_summary": self.numeric_summary,
            "feature_target_correlation": self.feature_target_correlation,
            "failure_rate_by_type": self.failure_rate_by_type,
            "outlier_summary": self.outlier_summary,
            "top_feature_correlations": self.top_feature_correlations,
        }

    def summary(self) -> str:
        balance = self.class_balance
        lines = [
            "EDA report",
            f"  rows: {self.n_rows}",
            "  class balance: "
            f"positive={balance['n_positive']} negative={balance['n_negative']} "
            f"positive_rate={balance['positive_rate']:.4f}",
        ]
        ranked = sorted(
            self.feature_target_correlation.items(),
            key=lambda item: abs(item[1] or 0.0),
            reverse=True,
        )
        lines.append("  feature/target correlation (top 5):")
        for name, value in ranked[:5]:
            lines.append(f"    {name}: {value:.4f}" if value is not None else f"    {name}: n/a")
        lines.append("  failure rate by type:")
        for value, stats in self.failure_rate_by_type.items():
            lines.append(
                f"    {value}: {stats['n_positive']}/{stats['n_total']} "
                f"({stats['positive_rate']:.4f})"
            )
        total_outliers = sum(item["count"] for item in self.outlier_summary.values())
        lines.append(f"  IQR outliers (numeric features): {total_outliers}")
        return "\n".join(lines)


def generate_eda_report(
    frame: pd.DataFrame,
    *,
    target: str = TARGET_COLUMN,
) -> EDAReport:
    """Run the full exploratory analysis for ``frame``.

    Args:
        frame: Raw dataframe including sensor, categorical and target columns.
        target: Target column name.

    Returns:
        A populated :class:`EDAReport`.
    """
    engineered = add_engineered_features(frame)
    analysis = engineered.loc[:, [*NUMERIC_FEATURE_COLUMNS, target, *CATEGORICAL_COLUMNS]]

    report = EDAReport(
        n_rows=int(frame.shape[0]),
        class_balance=class_balance(analysis, target),
        numeric_summary=numeric_distributions(analysis),
        feature_target_correlation=feature_target_correlation(analysis, target),
        failure_rate_by_type=failure_rate_by_type(analysis, target),
        outlier_summary=detect_outliers_iqr(analysis),
        top_feature_correlations=top_feature_correlations(analysis),
    )
    logger.info("Generated EDA report for %d rows", report.n_rows)
    return report
