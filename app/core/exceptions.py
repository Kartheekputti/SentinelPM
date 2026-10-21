"""Project-wide exception hierarchy.

Grouping domain errors under :class:`SentinelPMError` lets callers catch every
application-specific failure without accidentally swallowing unrelated bugs.
"""

from __future__ import annotations


class SentinelPMError(Exception):
    """Base class for all SentinelPM application errors."""


class ConfigurationError(SentinelPMError):
    """Raised when configuration cannot be resolved or is inconsistent."""


class DataLoadError(SentinelPMError):
    """Raised when a dataset cannot be located, read or parsed."""


class DataValidationError(SentinelPMError):
    """Raised when a dataset fails schema or quality validation."""


class FeatureError(SentinelPMError):
    """Raised when feature engineering cannot be applied to the input."""


class TrainingError(SentinelPMError):
    """Raised when model training or evaluation cannot be completed."""


class ExplainabilityError(SentinelPMError):
    """Raised when model explanations cannot be computed or formatted."""


class ModelRegistryError(SentinelPMError):
    """Raised when a model version cannot be tracked, registered, or loaded."""


class InferenceError(SentinelPMError):
    """Raised when input validation or model scoring fails during inference."""


class MonitoringError(SentinelPMError):
    """Raised when telemetry collection or drift calculation fails."""


class AlertError(SentinelPMError):
    """Raised when alert evaluation or dispatching fails."""
