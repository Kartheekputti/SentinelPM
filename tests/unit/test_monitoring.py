"""Unit tests for operational monitoring tracker (M08)."""

from __future__ import annotations

from app.monitoring.tracker import MonitoringTracker


def test_monitoring_tracker_record_requests() -> None:
    """Verify request counting and status bucketing."""
    tracker = MonitoringTracker()
    tracker.reset()

    tracker.record_request("/predict", 200, 15.0)
    tracker.record_request("/predict", 200, 25.0)
    tracker.record_request("/predict", 400, 10.0)

    summary = tracker.get_summary()
    assert summary.total_requests == 3
    assert summary.status_counts["2xx"] == 2
    assert summary.status_counts["4xx"] == 1
    assert summary.endpoint_counts["/predict"] == 3
    assert summary.total_errors == 1
    assert 0.33 <= summary.error_rate <= 0.34


def test_monitoring_tracker_latency_metrics() -> None:
    """Verify latency percentile calculations."""
    tracker = MonitoringTracker()
    tracker.reset()

    # Record 100 requests with latencies 1ms..100ms
    for ms in range(1, 101):
        tracker.record_request("/predict", 200, float(ms))

    summary = tracker.get_summary()
    assert summary.latency.count == 100
    assert summary.latency.min_ms == 1.0
    assert summary.latency.max_ms == 100.0
    assert 49.0 <= summary.latency.p50_ms <= 51.0
    assert 94.0 <= summary.latency.p95_ms <= 96.0


def test_monitoring_tracker_predictions() -> None:
    """Verify prediction distribution tracking."""
    tracker = MonitoringTracker()
    tracker.reset()

    tracker.record_prediction(0.05, predicted_failure=False)
    tracker.record_prediction(0.85, predicted_failure=True)
    tracker.record_prediction(0.10, predicted_failure=False)

    summary = tracker.get_summary()
    assert summary.predictions.total_predictions == 3
    assert summary.predictions.positive_predictions == 1
    assert 0.33 <= summary.predictions.positive_rate <= 0.34
    assert 0.33 <= summary.predictions.mean_probability <= 0.34


def test_monitoring_tracker_errors_and_quality() -> None:
    """Verify input quality issues and recent errors tracking."""
    tracker = MonitoringTracker()
    tracker.reset()

    tracker.record_error("/predict", 422, "Missing required feature")
    tracker.record_input_quality_issue("missing_features")
    tracker.record_model_usage("1.0.0")

    summary = tracker.get_summary()
    assert len(summary.recent_errors) == 1
    assert summary.recent_errors[0]["status_code"] == 422
    assert summary.input_quality_issues["missing_features"] == 1
    assert summary.model_version_usage["1.0.0"] == 1
