"""Automated end-to-end smoke test script for SentinelPM service."""

from __future__ import annotations

import argparse
import sys
import time

from app.api.app import create_app
from app.dashboard.client import ApiClient


def run_smoke_test(api_url: str | None = None) -> bool:
    """Execute end-to-end smoke test against live API or in-process ASGI app."""
    print("=" * 65)
    print("  SentinelPM — End-to-End System Smoke Test")
    print("=" * 65)

    if api_url:
        print(f"Connecting to live API target: {api_url}")
        client = ApiClient(base_url=api_url)
        if not client.is_available():
            print(f"[FAIL] Target API at {api_url} is unreachable.")
            return False
    else:
        # Check if local 8000 is running; if not, use ASGI transport
        candidate = ApiClient(base_url="http://127.0.0.1:8000")
        if candidate.is_available():
            print("Detected live SentinelPM API service at http://127.0.0.1:8000")
            client = candidate
        else:
            print("Local service not running. Initializing in-process test harness...")
            from fastapi.testclient import TestClient

            app = create_app()
            test_client = TestClient(app)
            client = ApiClient(client=test_client)

    steps_passed = 0
    total_steps = 6

    # 1. Health Probe
    try:
        t0 = time.perf_counter()
        health = client.check_health()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert health.get("status") == "healthy"
        print(f"[PASS] 1/6: GET /health ({elapsed_ms:.1f}ms) -> service: {health.get('service')}")
        steps_passed += 1
    except Exception as exc:
        print(f"[FAIL] 1/6: GET /health failed: {exc}")
        return False

    # 2. Readiness Probe
    try:
        t0 = time.perf_counter()
        ready = client.check_readiness()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert ready.get("model_loaded") is True
        print(f"[PASS] 2/6: GET /ready ({elapsed_ms:.1f}ms) -> model: {ready.get('model_name')}")
        steps_passed += 1
    except Exception as exc:
        print(f"[FAIL] 2/6: GET /ready failed: {exc}")
        return False

    # 3. Model Metadata Probe
    try:
        t0 = time.perf_counter()
        info = client.get_model_info()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert "feature_names" in info
        print(
            f"[PASS] 3/6: GET /model/info ({elapsed_ms:.1f}ms) -> version: v{info.get('model_version')}, "
            f"features: {len(info.get('feature_names', []))}"
        )
        steps_passed += 1
    except Exception as exc:
        print(f"[FAIL] 3/6: GET /model/info failed: {exc}")
        return False

    # 4. Single Prediction
    normal_record = {
        "Type": "M",
        "Air_temperature_K": 298.2,
        "Process_temperature_K": 308.7,
        "Rotational_speed_rpm": 1408,
        "Torque_Nm": 46.3,
        "Tool_wear_min": 3,
    }
    try:
        t0 = time.perf_counter()
        pred = client.predict(normal_record)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert "failure_probability" in pred
        assert pred.get("risk_level") in ["low", "medium", "high"]
        print(
            f"[PASS] 4/6: POST /predict ({elapsed_ms:.1f}ms) -> risk: {pred.get('risk_level')}, "
            f"probability: {pred.get('failure_probability') * 100:.1f}%"
        )
        steps_passed += 1
    except Exception as exc:
        print(f"[FAIL] 4/6: POST /predict failed: {exc}")
        return False

    # 5. Batch Prediction
    batch_records = [
        {
            "Type": "L",
            "Air_temperature_K": 298.1,
            "Process_temperature_K": 308.6,
            "Rotational_speed_rpm": 1551,
            "Torque_Nm": 42.8,
            "Tool_wear_min": 0,
        },
        {
            "Type": "H",
            "Air_temperature_K": 303.0,
            "Process_temperature_K": 312.0,
            "Rotational_speed_rpm": 1320,
            "Torque_Nm": 65.0,
            "Tool_wear_min": 240,
        },
    ]
    try:
        t0 = time.perf_counter()
        batch = client.predict_batch(batch_records)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert batch.get("count") == 2
        print(
            f"[PASS] 5/6: POST /predict/batch ({elapsed_ms:.1f}ms) -> processed: {batch.get('count')} records"
        )
        steps_passed += 1
    except Exception as exc:
        print(f"[FAIL] 5/6: POST /predict/batch failed: {exc}")
        return False

    # 6. Monitoring Summary
    try:
        t0 = time.perf_counter()
        monitoring = client.get_monitoring_summary()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        reqs = monitoring.get("telemetry", {}).get("total_requests", 0)
        assert reqs >= 2
        print(
            f"[PASS] 6/6: GET /monitoring/summary ({elapsed_ms:.1f}ms) -> total telemetry requests: {reqs}"
        )
        steps_passed += 1
    except Exception as exc:
        print(f"[FAIL] 6/6: GET /monitoring/summary failed: {exc}")
        return False

    print("-" * 65)
    print(f"Smoke Test Result: {steps_passed}/{total_steps} checks PASSED successfully.")
    print("=" * 65)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SentinelPM system smoke test.")
    parser.add_argument(
        "--url", type=str, default=None, help="Target URL (e.g. http://127.0.0.1:8000)"
    )
    args = parser.parse_args()

    success = run_smoke_test(api_url=args.url)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
