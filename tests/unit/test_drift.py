"""Unit tests for statistical drift detection (M08)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.core.exceptions import MonitoringError
from app.monitoring.drift import (
    DRIFT_DISCLAIMER,
    DriftDetector,
    compute_psi,
)
from app.monitoring.reference import build_reference_data


@pytest.fixture
def baseline_frame() -> pd.DataFrame:
    """Create a baseline frame of 100 observations."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "Air temperature [K]": np.random.normal(298.0, 1.5, n),
            "Process temperature [K]": np.random.normal(308.0, 1.5, n),
            "Rotational speed [rpm]": np.random.normal(1500.0, 50.0, n),
            "Torque [Nm]": np.random.normal(40.0, 5.0, n),
            "Tool wear [min]": np.random.uniform(0.0, 200.0, n),
            "Type": np.random.choice(["L", "M", "H"], p=[0.6, 0.3, 0.1], size=n),
            "Machine failure": [0] * 96 + [1] * 4,
        }
    )


def test_reference_data_build_and_save(baseline_frame: pd.DataFrame, tmp_path: Path) -> None:
    """Verify ReferenceData construction, serialization and reloading."""
    ref = build_reference_data(baseline_frame, sample_size=100)
    assert ref.n_samples == 100
    assert "Air temperature [K]" in ref.sensor_stats

    save_path = tmp_path / "ref.json"
    ref.save(save_path)
    assert save_path.exists()

    loaded = ref.load(save_path)
    assert loaded.n_samples == 100
    assert loaded.sensor_stats["Torque [Nm]"].mean == pytest.approx(
        ref.sensor_stats["Torque [Nm]"].mean, rel=1e-4
    )


def test_drift_detector_nominal_distribution(baseline_frame: pd.DataFrame) -> None:
    """Ensure identical distribution from same generation process does not flag drift."""
    ref = build_reference_data(baseline_frame)
    detector = DriftDetector(ref, pvalue_threshold=0.01)

    # Similar fresh sample
    np.random.seed(999)
    fresh_sample = pd.DataFrame(
        {
            "Air temperature [K]": np.random.normal(298.0, 1.5, 50),
            "Process temperature [K]": np.random.normal(308.0, 1.5, 50),
            "Rotational speed [rpm]": np.random.normal(1500.0, 50.0, 50),
            "Torque [Nm]": np.random.normal(40.0, 5.0, 50),
            "Tool wear [min]": np.random.uniform(0.0, 200.0, 50),
            "Type": np.random.choice(["L", "M", "H"], p=[0.6, 0.3, 0.1], size=50),
        }
    )

    report = detector.evaluate_drift(fresh_sample)
    assert report.drift_detected is False
    assert report.drifted_features_count == 0
    assert DRIFT_DISCLAIMER in report.disclaimer


def test_drift_detector_detects_deliberate_sensor_drift(baseline_frame: pd.DataFrame) -> None:
    """Verify KS-test flags drift when sensor data is deliberately shifted."""
    ref = build_reference_data(baseline_frame)
    detector = DriftDetector(ref, pvalue_threshold=0.05)

    shifted_df = baseline_frame.copy()
    # Deliberately shift air temperature by +20K and torque by +30Nm
    shifted_df["Air temperature [K]"] = shifted_df["Air temperature [K]"] + 20.0
    shifted_df["Torque [Nm]"] = shifted_df["Torque [Nm]"] + 30.0

    report = detector.evaluate_drift(shifted_df)
    assert report.drift_detected is True
    assert "Air temperature [K]" in report.drifted_features
    assert "Torque [Nm]" in report.drifted_features
    assert report.features["Air temperature [K]"].drifted is True
    assert report.features["Air temperature [K]"].p_value < 0.05


def test_psi_calculation_and_type_drift() -> None:
    """Verify Population Stability Index identifies categorical distribution shifts."""
    ref_props = {"L": 0.6, "M": 0.3, "H": 0.1}

    # Similar props -> low PSI
    nominal_props = {"L": 0.58, "M": 0.31, "H": 0.11}
    assert compute_psi(ref_props, nominal_props) < 0.10

    # Drastically shifted props -> high PSI (> 0.20)
    shifted_props = {"L": 0.1, "M": 0.1, "H": 0.8}
    assert compute_psi(ref_props, shifted_props) > 0.50


def test_drift_empty_frame_raises(baseline_frame: pd.DataFrame) -> None:
    """Verify empty frame raises MonitoringError."""
    ref = build_reference_data(baseline_frame)
    detector = DriftDetector(ref)
    with pytest.raises(MonitoringError, match="empty"):
        detector.evaluate_drift(pd.DataFrame())
