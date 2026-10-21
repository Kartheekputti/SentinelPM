"""Thread-safe operational telemetry and request monitoring for SentinelPM.

Tracks API requests, error rates, latency distributions, model version usage,
input quality issues, and rolling prediction statistics.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import Settings, get_settings


@dataclass
class LatencyMetrics:
    """Request latency percentiles in milliseconds."""

    count: int = 0
    mean_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "count": self.count,
            "mean_ms": round(self.mean_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
        }


@dataclass
class PredictionDistributionMetrics:
    """Rolling distribution statistics for generated predictions."""

    total_predictions: int = 0
    positive_predictions: int = 0
    positive_rate: float = 0.0
    mean_probability: float = 0.0
    p50_probability: float = 0.0
    p95_probability: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "total_predictions": self.total_predictions,
            "positive_predictions": self.positive_predictions,
            "positive_rate": round(self.positive_rate, 4),
            "mean_probability": round(self.mean_probability, 4),
            "p50_probability": round(self.p50_probability, 4),
            "p95_probability": round(self.p95_probability, 4),
        }


@dataclass
class MonitoringSummary:
    """Unified telemetry summary for monitoring endpoints."""

    timestamp: str
    total_requests: int
    status_counts: dict[str, int]
    endpoint_counts: dict[str, int]
    total_errors: int
    error_rate: float
    latency: LatencyMetrics
    model_version_usage: dict[str, int]
    input_quality_issues: dict[str, int]
    predictions: PredictionDistributionMetrics
    recent_errors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_requests": self.total_requests,
            "status_counts": self.status_counts,
            "endpoint_counts": self.endpoint_counts,
            "total_errors": self.total_errors,
            "error_rate": round(self.error_rate, 4),
            "latency": self.latency.to_dict(),
            "model_version_usage": self.model_version_usage,
            "input_quality_issues": self.input_quality_issues,
            "predictions": self.predictions.to_dict(),
            "recent_errors": self.recent_errors,
        }


class MonitoringTracker:
    """Thread-safe telemetry tracker for API and prediction performance."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.window_size = int(self.settings.monitoring_window_size)
        self._lock = threading.Lock()

        # Operational metrics
        self._total_requests: int = 0
        self._status_counts: dict[str, int] = {}
        self._endpoint_counts: dict[str, int] = {}
        self._total_errors: int = 0
        self._recent_errors: deque[dict[str, Any]] = deque(maxlen=50)

        # Latencies (milliseconds)
        self._latencies_ms: deque[float] = deque(maxlen=self.window_size)

        # Model usage
        self._model_version_usage: dict[str, int] = {}

        # Input quality counters
        self._input_quality_issues: dict[str, int] = {
            "missing_features": 0,
            "out_of_bounds": 0,
            "invalid_types": 0,
            "target_leakage_attempts": 0,
        }

        # Rolling predictions
        self._probabilities: deque[float] = deque(maxlen=self.window_size)
        self._predicted_failures: deque[int] = deque(maxlen=self.window_size)

        # Buffered sensor rows for drift analysis
        self._sensor_buffer: deque[dict[str, float]] = deque(maxlen=self.window_size)

    def record_request(
        self,
        endpoint: str,
        status_code: int,
        latency_ms: float,
    ) -> None:
        """Record an API request with latency and status code."""
        with self._lock:
            self._total_requests += 1
            status_bucket = f"{status_code // 100}xx"
            self._status_counts[status_bucket] = self._status_counts.get(status_bucket, 0) + 1
            self._endpoint_counts[endpoint] = self._endpoint_counts.get(endpoint, 0) + 1
            self._latencies_ms.append(max(0.0, float(latency_ms)))

            if status_code >= 400:
                self._total_errors += 1

    def record_error(
        self,
        endpoint: str,
        status_code: int,
        error_message: str,
    ) -> None:
        """Record an error event with details."""
        with self._lock:
            self._recent_errors.append(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "endpoint": endpoint,
                    "status_code": status_code,
                    "error_message": error_message[:200],  # truncate to prevent bloat
                }
            )

    def record_model_usage(self, model_version: str) -> None:
        """Track which model version served a prediction."""
        with self._lock:
            self._model_version_usage[model_version] = (
                self._model_version_usage.get(model_version, 0) + 1
            )

    def record_input_quality_issue(self, issue_type: str) -> None:
        """Record an input data validation issue."""
        with self._lock:
            self._input_quality_issues[issue_type] = (
                self._input_quality_issues.get(issue_type, 0) + 1
            )

    def record_prediction(self, probability: float, predicted_failure: bool) -> None:
        """Record predicted failure probability and outcome flag."""
        with self._lock:
            self._probabilities.append(float(probability))
            self._predicted_failures.append(1 if predicted_failure else 0)

    def record_sensor_telemetry(self, sensor_dict: dict[str, float]) -> None:
        """Buffer a numeric sensor observation for rolling distribution analysis."""
        with self._lock:
            self._sensor_buffer.append(dict(sensor_dict))

    def get_buffered_sensor_frame(self) -> pd.DataFrame:
        """Retrieve snapshot of recent sensor telemetry for drift evaluation."""
        with self._lock:
            if not self._sensor_buffer:
                return pd.DataFrame()
            return pd.DataFrame(list(self._sensor_buffer))

    def _compute_latency_metrics(self) -> LatencyMetrics:
        if not self._latencies_ms:
            return LatencyMetrics()
        arr = np.asarray(self._latencies_ms)
        return LatencyMetrics(
            count=len(arr),
            mean_ms=float(np.mean(arr)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
        )

    def _compute_prediction_metrics(self) -> PredictionDistributionMetrics:
        if not self._probabilities:
            return PredictionDistributionMetrics()
        probs = np.asarray(self._probabilities)
        preds = np.asarray(self._predicted_failures)
        pos = int(np.sum(preds == 1))
        return PredictionDistributionMetrics(
            total_predictions=len(probs),
            positive_predictions=pos,
            positive_rate=pos / max(len(preds), 1),
            mean_probability=float(np.mean(probs)),
            p50_probability=float(np.percentile(probs, 50)),
            p95_probability=float(np.percentile(probs, 95)),
        )

    def get_summary(self) -> MonitoringSummary:
        """Produce a complete operational telemetry summary."""
        with self._lock:
            err_rate = (
                (self._total_errors / self._total_requests) if self._total_requests > 0 else 0.0
            )
            return MonitoringSummary(
                timestamp=datetime.now(UTC).isoformat(),
                total_requests=self._total_requests,
                status_counts=dict(self._status_counts),
                endpoint_counts=dict(self._endpoint_counts),
                total_errors=self._total_errors,
                error_rate=err_rate,
                latency=self._compute_latency_metrics(),
                model_version_usage=dict(self._model_version_usage),
                input_quality_issues=dict(self._input_quality_issues),
                predictions=self._compute_prediction_metrics(),
                recent_errors=list(self._recent_errors),
            )

    def reset(self) -> None:
        """Reset all counters (for testing)."""
        with self._lock:
            self._total_requests = 0
            self._status_counts.clear()
            self._endpoint_counts.clear()
            self._total_errors = 0
            self._recent_errors.clear()
            self._latencies_ms.clear()
            self._model_version_usage.clear()
            for key in self._input_quality_issues:
                self._input_quality_issues[key] = 0
            self._probabilities.clear()
            self._predicted_failures.clear()
            self._sensor_buffer.clear()
