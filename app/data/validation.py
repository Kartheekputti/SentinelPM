"""Dataset validation.

Validation never mutates the input. It collects every problem it can find and
returns a structured :class:`ValidationReport` so the caller decides whether to
continue. Nothing is dropped silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd
import pandera.pandas as pa

from app.core.logging import get_logger
from app.data.schema import (
    BINARY_COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERIC_BOUNDS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    TYPE_CATEGORIES,
    build_raw_schema,
)

logger = get_logger(__name__)

IssueLevel = Literal["error", "warning"]


@dataclass(frozen=True)
class ValidationIssue:
    """A single validation finding."""

    level: IssueLevel
    category: str
    message: str
    column: str | None = None
    count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "category": self.category,
            "message": self.message,
            "column": self.column,
            "count": self.count,
        }


@dataclass
class ValidationReport:
    """Aggregate result of validating a dataset."""

    n_rows: int = 0
    n_columns: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.level == "warning"]

    @property
    def is_valid(self) -> bool:
        """A dataset is valid when it has no error-level issues."""
        return not self.errors

    def add(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "is_valid": self.is_valid,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def summary(self) -> str:
        lines = [
            f"Validation: {self.n_rows} rows x {self.n_columns} columns",
            f"  errors={len(self.errors)} warnings={len(self.warnings)}",
        ]
        for issue in self.issues:
            prefix = "ERROR" if issue.level == "error" else "WARN"
            location = f" [{issue.column}]" if issue.column else ""
            count = f" (x{issue.count})" if issue.count else ""
            lines.append(f"  {prefix} {issue.category}{location}{count}: {issue.message}")
        if self.is_valid:
            lines.append("  Result: VALID")
        else:
            lines.append("  Result: INVALID")
        return "\n".join(lines)


def validate_dataset(
    frame: pd.DataFrame,
    schema: pa.DataFrameSchema | None = None,
    *,
    check_duplicates: bool = True,
    check_missing: bool = True,
    check_unexpected_columns: bool = True,
    require_target: bool = True,
) -> ValidationReport:
    """Validate a raw dataset against the AI4I schema.

    Args:
        frame: The dataframe to validate. It is never modified.
        schema: Optional Pandera schema override.
        check_duplicates: Report duplicate rows as warnings.
        check_missing: Report missing values.
        check_unexpected_columns: Report columns outside the expected schema.
        require_target: Treat a missing target column as an error.

    Returns:
        A :class:`ValidationReport` describing every finding.

    Raises:
        TypeError: If ``frame`` is not a :class:`pandas.DataFrame`.
    """
    if not isinstance(frame, pd.DataFrame):
        msg = f"validate_dataset expects a DataFrame, got {type(frame).__name__}"
        raise TypeError(msg)

    report = ValidationReport(n_rows=len(frame), n_columns=frame.shape[1])

    _check_required_columns(frame, report, require_target=require_target)
    if report.errors:
        # Column-level value checks require every required column to exist.
        logger.warning("Skipping value checks: required columns are missing")
        return report

    if check_unexpected_columns:
        _check_unexpected_columns(frame, report)
    _check_dtypes(frame, report)
    if check_missing:
        _check_missing_values(frame, report)
    if check_duplicates:
        _check_duplicate_rows(frame, report)
    active_schema = schema or build_raw_schema(
        include_target=require_target,
        allow_null=not check_missing,
    )
    _check_values(frame, report, active_schema)

    logger.info(
        "Validation finished: valid=%s errors=%d warnings=%d",
        report.is_valid,
        len(report.errors),
        len(report.warnings),
    )
    return report


def _check_required_columns(
    frame: pd.DataFrame, report: ValidationReport, *, require_target: bool
) -> None:
    expected = list(REQUIRED_COLUMNS)
    if not require_target:
        expected = [column for column in expected if column != TARGET_COLUMN]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        report.add(
            ValidationIssue(
                level="error",
                category="missing_columns",
                message=f"Missing required columns: {missing}",
                count=len(missing),
            )
        )


def _check_unexpected_columns(frame: pd.DataFrame, report: ValidationReport) -> None:
    unexpected = [column for column in frame.columns if column not in REQUIRED_COLUMNS]
    if unexpected:
        report.add(
            ValidationIssue(
                level="warning",
                category="unexpected_columns",
                message=f"Unexpected columns not in the known schema: {unexpected}",
                count=len(unexpected),
            )
        )


def _check_dtypes(frame: pd.DataFrame, report: ValidationReport) -> None:
    for column, _ in NUMERIC_BOUNDS.items():
        if column in frame.columns and not pd.api.types.is_numeric_dtype(frame[column]):
            report.add(
                ValidationIssue(
                    level="error",
                    category="invalid_dtype",
                    message=(
                        f"Column must be numeric, got dtype {frame[column].dtype}"
                    ),
                    column=column,
                )
            )
    for column in CATEGORICAL_COLUMNS:
        if column in frame.columns and not pd.api.types.is_object_dtype(
            frame[column]
        ) and not pd.api.types.is_string_dtype(frame[column]):
            report.add(
                ValidationIssue(
                    level="error",
                    category="invalid_dtype",
                    message=f"Column must be categorical/string, got dtype {frame[column].dtype}",
                    column=column,
                )
            )
    for column in BINARY_COLUMNS:
        if column in frame.columns and not pd.api.types.is_integer_dtype(frame[column]):
            report.add(
                ValidationIssue(
                    level="error",
                    category="invalid_dtype",
                    message=f"Column must be integer-valued, got dtype {frame[column].dtype}",
                    column=column,
                )
            )


def _check_missing_values(frame: pd.DataFrame, report: ValidationReport) -> None:
    for column in REQUIRED_COLUMNS:
        if column not in frame.columns:
            continue
        n_missing = int(frame[column].isna().sum())
        if n_missing:
            report.add(
                ValidationIssue(
                    level="error",
                    category="missing_values",
                    message=f"Column contains {n_missing} missing value(s)",
                    column=column,
                    count=n_missing,
                )
            )


def _check_duplicate_rows(frame: pd.DataFrame, report: ValidationReport) -> None:
    n_duplicates = int(frame.duplicated().sum())
    if n_duplicates:
        report.add(
            ValidationIssue(
                level="warning",
                category="duplicate_rows",
                message=f"Dataset contains {n_duplicates} duplicate row(s)",
                count=n_duplicates,
            )
        )


def _check_values(
    frame: pd.DataFrame, report: ValidationReport, schema: pa.DataFrameSchema
) -> None:
    try:
        schema.validate(frame, lazy=True)
    except pa.errors.SchemaErrors as exc:
        for _, row in exc.failure_cases.iterrows():
            report.add(_issue_from_failure_case(row))
    except (TypeError, ValueError, pa.errors.SchemaError) as exc:
        report.add(
            ValidationIssue(
                level="error",
                category="schema",
                message=f"Schema validation failed: {exc}",
            )
        )


def _issue_from_failure_case(row: pd.Series) -> ValidationIssue:
    column = row.get("column")
    check = str(row.get("check", ""))
    failure_case = row.get("failure_case")
    category = _classify_check(check)
    if pd.isna(column):
        column = None
    return ValidationIssue(
        level="error",
        category=category,
        message=f"Value {failure_case!r} failed check {check!r}",
        column=str(column) if column is not None else None,
        count=1,
    )


def _classify_check(check: str) -> str:
    if "isin" in check:
        return "invalid_category"
    if any(token in check for token in ("in_range", "greater", "less", "equal")):
        return "out_of_range"
    if "dtype" in check:
        return "invalid_dtype"
    return "value"
