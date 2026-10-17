"""Data ingestion, schema definition and validation."""

from app.data.loader import load_csv, load_raw_dataset
from app.data.quality import DataQualityReport, generate_quality_report
from app.data.schema import (
    CATEGORICAL_COLUMNS,
    DEFAULT_RAW_FILENAME,
    FAILURE_MODE_COLUMNS,
    ID_COLUMNS,
    RAW_FEATURE_COLUMNS,
    REQUIRED_COLUMNS,
    SENSOR_COLUMNS,
    TARGET_COLUMN,
    TARGET_DERIVED_COLUMNS,
    build_raw_schema,
)
from app.data.validation import (
    ValidationIssue,
    ValidationReport,
    validate_dataset,
)

__all__ = [
    "CATEGORICAL_COLUMNS",
    "DEFAULT_RAW_FILENAME",
    "FAILURE_MODE_COLUMNS",
    "ID_COLUMNS",
    "RAW_FEATURE_COLUMNS",
    "REQUIRED_COLUMNS",
    "SENSOR_COLUMNS",
    "TARGET_COLUMN",
    "TARGET_DERIVED_COLUMNS",
    "DataQualityReport",
    "ValidationIssue",
    "ValidationReport",
    "build_raw_schema",
    "generate_quality_report",
    "load_csv",
    "load_raw_dataset",
    "validate_dataset",
]
