"""Unit tests for alert manager and cooldown logic (M08)."""

from __future__ import annotations

from app.monitoring.alerts import AlertManager
from app.monitoring.drift import DriftReport
from app.monitoring.tracker import MonitoringTracker


def test_alert_manager_error_rate_threshold() -> None:
    """Verify high error rate triggers critical alert."""
    tracker = MonitoringTracker()
    tracker.reset()
    manager = AlertManager(max_error_rate=0.05, cooldown_seconds=300)

    # 18 successful requests, 4 failed requests (error rate = 4/22 ~ 18%)
    for _ in range(18):
        tracker.record_request("/predict", 200, 20.0)
    for _ in range(4):
        tracker.record_request("/predict", 500, 20.0)

    alerts = manager.evaluate(tracker)
    assert len(alerts) == 1
    assert alerts[0].alert_type == "high_error_rate"
    assert alerts[0].severity == "critical"
    assert alerts[0].current_value > 0.05


def test_alert_manager_cooldown_suppression() -> None:
    """Verify duplicate alerts within cooldown window are suppressed."""
    tracker = MonitoringTracker()
    tracker.reset()
    manager = AlertManager(max_error_rate=0.05, cooldown_seconds=300)

    for _ in range(20):
        tracker.record_request("/predict", 500, 20.0)

    # First evaluation triggers alert
    alerts1 = manager.evaluate(tracker)
    assert len(alerts1) == 1

    # Immediate second evaluation is suppressed by cooldown
    alerts2 = manager.evaluate(tracker)
    assert len(alerts2) == 0


def test_alert_manager_data_drift_alert() -> None:
    """Verify data drift report triggers warning alert."""
    tracker = MonitoringTracker()
    tracker.reset()
    manager = AlertManager(cooldown_seconds=300)

    dummy_report = DriftReport(
        timestamp="2026-10-21T00:00:00Z",
        drift_detected=True,
        evaluated_samples=50,
        drifted_features_count=2,
        drifted_features=["Torque [Nm]", "Air temperature [K]"],
        features={},
        prediction_shift={},
        summary="Drift detected",
    )

    alerts = manager.evaluate(tracker, drift_report=dummy_report)
    assert len(alerts) == 1
    assert alerts[0].alert_type == "data_drift_detected"
    assert alerts[0].severity == "warning"
    assert "Torque [Nm]" in alerts[0].message
