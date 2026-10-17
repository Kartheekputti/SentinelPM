"""Feature engineering, preprocessing pipelines and exploratory analysis."""

from app.features.definitions import (
    ENGINEERED_FEATURE_COLUMNS,
    FEATURE_DEFINITIONS,
    NUMERIC_FEATURE_COLUMNS,
    ROTATIONAL_POWER,
    TEMPERATURE_DIFFERENCE,
    FeatureDefinition,
    documented_feature_names,
)
from app.features.eda import (
    EDAReport,
    class_balance,
    detect_outliers_iqr,
    failure_rate_by_type,
    feature_target_correlation,
    generate_eda_report,
    numeric_distributions,
    top_feature_correlations,
)
from app.features.engineering import (
    SensorFeatureEngineer,
    add_engineered_features,
    extract_target,
    prepare_feature_frame,
)
from app.features.pipeline import build_feature_pipeline, feature_names, output_feature_names

__all__ = [
    "ENGINEERED_FEATURE_COLUMNS",
    "FEATURE_DEFINITIONS",
    "NUMERIC_FEATURE_COLUMNS",
    "ROTATIONAL_POWER",
    "TEMPERATURE_DIFFERENCE",
    "EDAReport",
    "FeatureDefinition",
    "SensorFeatureEngineer",
    "add_engineered_features",
    "build_feature_pipeline",
    "class_balance",
    "detect_outliers_iqr",
    "documented_feature_names",
    "extract_target",
    "failure_rate_by_type",
    "feature_names",
    "feature_target_correlation",
    "generate_eda_report",
    "numeric_distributions",
    "output_feature_names",
    "prepare_feature_frame",
    "top_feature_correlations",
]
