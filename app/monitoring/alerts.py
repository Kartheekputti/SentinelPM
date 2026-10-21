"""Alert management with cooldown and deduplication for SentinelPM.

Evaluates operational metrics and drift reports against configurable thresholds.
Prevents alert flooding using per-condition cooldown timers.
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.monitoring.drift import DriftReport
from app.monitoring.tracker import MonitoringTracker

logger = get_logger(__name__)

Severity = Literal["info", "warning", "critical"]


@dataclass(frozen=True)
class AlertEvent:
    """Dispatched alert event for an operational anomaly or drift trigger."""

    alert_id: str
    timestamp: str
    alert_type: str
    severity: Severity
    metric_name: str
    current_value: float
    threshold_value: float
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "metric_name": self.metric_name,
            "current_value": round(float(self.current_value), 4),
            "threshold_value": round(float(self.threshold_value), 4),
            "message": self.message,
        }


class AlertManager:
    """Manages threshold evaluation, deduplication and cooldown for alerts."""

    def __init__(
        self,
        *,
        cooldown_seconds: int | None = None,
        max_error_rate: float = 0.05,
        max_p95_latency_ms: float = 250.0,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cooldown_seconds = (
            cooldown_seconds
            if cooldown_seconds is not None
            else int(self.settings.alert_cooldown_seconds)
        )
        self.max_error_rate = max_error_rate
        self.max_p95_latency_ms = max_p95_latency_ms

        self._last_alert_time: dict[str, datetime] = {}
        self._history: deque[AlertEvent] = deque(maxlen=100)

    def _should_trigger(self, alert_type: str, now: datetime) -> bool:
        """Check if alert condition is outside the cooldown window."""
        last_time = self._last_alert_time.get(alert_type)
        if last_time is None:
            return True
        elapsed = (now - last_time).total_seconds()
        return elapsed >= self.cooldown_seconds

    def evaluate(
        self,
        tracker: MonitoringTracker,
        drift_report: DriftReport | None = None,
    ) -> list[AlertEvent]:
        """Evaluate monitoring metrics and return newly triggered alerts.

        Args:
            tracker: Operational telemetry tracker.
            drift_report: Optional latest drift evaluation report.

        Returns:
            List of newly triggered (non-suppressed) :class:`AlertEvent` instances.
        """
        now = datetime.now(UTC)
        summary = tracker.get_summary()
        new_alerts: list[AlertEvent] = []

        try:
            # 1. Error rate check
            if summary.total_requests >= 20 and summary.error_rate > self.max_error_rate:
                alert_type = "high_error_rate"
                if self._should_trigger(alert_type, now):
                    alert = AlertEvent(
                        alert_id=str(uuid.uuid4()),
                        timestamp=now.isoformat(),
                        alert_type=alert_type,
                        severity="critical",
                        metric_name="error_rate",
                        current_value=summary.error_rate,
                        threshold_value=self.max_error_rate,
                        message=(
                            f"API error rate ({summary.error_rate:.1%}) exceeded threshold "
                            f"({self.max_error_rate:.1%}) across {summary.total_requests} requests."
                        ),
                    )
                    new_alerts.append(alert)
                    self._last_alert_time[alert_type] = now

            # 2. Latency check
            if summary.latency.count >= 10 and summary.latency.p95_ms > self.max_p95_latency_ms:
                alert_type = "high_latency"
                if self._should_trigger(alert_type, now):
                    alert = AlertEvent(
                        alert_id=str(uuid.uuid4()),
                        timestamp=now.isoformat(),
                        alert_type=alert_type,
                        severity="warning",
                        metric_name="p95_latency_ms",
                        current_value=summary.latency.p95_ms,
                        threshold_value=self.max_p95_latency_ms,
                        message=(
                            f"P95 prediction latency ({summary.latency.p95_ms:.1f}ms) breached "
                            f"SLA threshold ({self.max_p95_latency_ms:.1f}ms)."
                        ),
                    )
                    new_alerts.append(alert)
                    self._last_alert_time[alert_type] = now

            # 3. Data drift check
            if drift_report and drift_report.drift_detected:
                alert_type = "data_drift_detected"
                if self._should_trigger(alert_type, now):
                    alert = AlertEvent(
                        alert_id=str(uuid.uuid4()),
                        timestamp=now.isoformat(),
                        alert_type=alert_type,
                        severity="warning",
                        metric_name="drifted_features_count",
                        current_value=float(drift_report.drifted_features_count),
                        threshold_value=1.0,
                        message=(
                            f"Statistical data drift detected in {drift_report.drifted_features_count} "
                            f"feature(s): {', '.join(drift_report.drifted_features)}."
                        ),
                    )
                    new_alerts.append(alert)
                    self._last_alert_time[alert_type] = now

            # 4. Prediction distribution shift
            if drift_report and drift_report.prediction_shift.get("shift_detected"):
                alert_type = "prediction_distribution_shift"
                if self._should_trigger(alert_type, now):
                    cur_rate = drift_report.prediction_shift["current_failure_rate"]
                    base_rate = drift_report.prediction_shift["baseline_failure_rate"]
                    alert = AlertEvent(
                        alert_id=str(uuid.uuid4()),
                        timestamp=now.isoformat(),
                        alert_type=alert_type,
                        severity="warning",
                        metric_name="prediction_failure_rate",
                        current_value=cur_rate,
                        threshold_value=base_rate,
                        message=(
                            f"Prediction failure rate shifted significantly: currently {cur_rate:.1%} "
                            f"vs baseline {base_rate:.1%}."
                        ),
                    )
                    new_alerts.append(alert)
                    self._last_alert_time[alert_type] = now

        except Exception as exc:
            logger.error("Error evaluating alert rules: %s", exc)

        for a in new_alerts:
            self._history.append(a)
            logger.warning("ALERT [%s] %s: %s", a.severity.upper(), a.alert_type, a.message)

        return new_alerts

    def get_recent_alerts(self) -> list[AlertEvent]:
        """Retrieve recent dispatched alerts."""
        return list(self._history)
