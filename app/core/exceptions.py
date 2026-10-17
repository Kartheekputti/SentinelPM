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
