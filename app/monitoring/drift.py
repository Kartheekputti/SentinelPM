"""Statistical data drift and distribution shift detection for SentinelPM.

Employs the two-sample Kolmogorov-Smirnov test for continuous sensors and Population
Stability Index (PSI) for categorical descriptors.

Keeps separate:
1. Feature data drift.
2. Prediction distribution shifts.
3. True accuracy degradation (which requires verified ground-truth labels).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.core.exceptions import MonitoringError
from app.core.logging import get_logger
from app.monitoring.reference import ReferenceData

logger = get_logger(__name__)

DRIFT_DISCLAIMER: str = (
    "Input feature drift detects statistical distribution shifts in operating sensor telemetry. "
    "Data drift alone does NOT prove that model accuracy has degraded, as machine failure "
    "relationships may remain invariant or unobserved without ground-truth labels."
)


@dataclass
class FeatureDriftResult:
    """Drift test outcome for a single sensor feature."""

    feature_name: str
    method: str
    statistic: float
    p_value: float | None
    drifted: bool
    reference_mean: float
    current_mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "method": self.method,
            "statistic": round(float(self.statistic), 4),
            "p_value": round(float(self.p_value), 4) if self.p_value is not None else None,
            "drifted": self.drifted,
            "reference_mean": round(float(self.reference_mean), 4),
            "current_mean": round(float(self.current_mean), 4),
        }


@dataclass
class DriftReport:
    """Comprehensive data drift and prediction distribution report."""

    timestamp: str
    drift_detected: bool
    evaluated_samples: int
    drifted_features_count: int
    drifted_features: list[str]
    features: dict[str, FeatureDriftResult]
    prediction_shift: dict[str, Any]
    summary: str
    disclaimer: str = DRIFT_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "drift_detected": self.drift_detected,
            "evaluated_samples": self.evaluated_samples,
            "drifted_features_count": self.drifted_features_count,
            "drifted_features": self.drifted_features,
            "features": {k: v.to_dict() for k, v in self.features.items()},
            "prediction_shift": self.prediction_shift,
            "summary": self.summary,
            "disclaimer": self.disclaimer,
        }


def compute_psi(reference_props: dict[str, float], current_props: dict[str, float]) -> float:
    """Calculate the Population Stability Index (PSI) between category distributions."""
    all_categories = set(reference_props.keys()) | set(current_props.keys())
    psi = 0.0
    epsilon = 1e-4

    for cat in all_categories:
        ref_p = max(reference_props.get(cat, 0.0), epsilon)
        cur_p = max(current_props.get(cat, 0.0), epsilon)
        psi += (cur_p - ref_p) * np.log(cur_p / ref_p)

    return float(psi)


class DriftDetector:
    """Statistical detector for sensor telemetry drift and prediction shifts."""

    def __init__(
        self,
        reference_data: ReferenceData,
        *,
        pvalue_threshold: float = 0.05,
        psi_threshold: float = 0.20,
    ) -> None:
        self.reference_data = reference_data
        self.pvalue_threshold = pvalue_threshold
        self.psi_threshold = psi_threshold

    def evaluate_drift(
        self,
        current_frame: pd.DataFrame,
        *,
        current_failure_rate: float | None = None,
        min_samples: int = 10,
    ) -> DriftReport:
        """Run statistical drift tests comparing current observations to reference baseline.

        Args:
            current_frame: Dataframe with recent sensor observations.
            current_failure_rate: Optional observed positive prediction rate.
            min_samples: Minimum observations required to conduct meaningful statistical tests.

        Returns:
            A populated :class:`DriftReport`.

        Raises:
            MonitoringError: If current_frame is empty.
        """
        if len(current_frame) == 0:
            msg = "Cannot evaluate drift on an empty dataframe"
            raise MonitoringError(msg)

        timestamp = datetime.now(UTC).isoformat()
        sample_count = len(current_frame)

        if sample_count < min_samples:
            return DriftReport(
                timestamp=timestamp,
                drift_detected=False,
                evaluated_samples=sample_count,
                drifted_features_count=0,
                drifted_features=[],
                features={},
                prediction_shift={"status": "insufficient_samples"},
                summary=f"Insufficient samples ({sample_count} < {min_samples}) to evaluate distribution drift.",
                disclaimer=DRIFT_DISCLAIMER,
            )

        feature_results: dict[str, FeatureDriftResult] = {}
        drifted_features: list[str] = []

        # 1. Continuous sensor features (Two-sample Kolmogorov-Smirnov test)
        for col, ref_stats in self.reference_data.sensor_stats.items():
            if col in current_frame.columns:
                cur_series = current_frame[col].dropna().astype(float)
                if len(cur_series) < min_samples:
                    continue

                ref_values = np.asarray(ref_stats.sample_values, dtype=float)
                cur_values = cur_series.to_numpy(dtype=float)

                ks_stat, p_val = stats.ks_2samp(ref_values, cur_values)
                is_drifted = bool(p_val < self.pvalue_threshold)

                if is_drifted:
                    drifted_features.append(col)

                feature_results[col] = FeatureDriftResult(
                    feature_name=col,
                    method="ks_2samp",
                    statistic=float(ks_stat),
                    p_value=float(p_val),
                    drifted=is_drifted,
                    reference_mean=ref_stats.mean,
                    current_mean=float(cur_series.mean()),
                )

        # 2. Categorical descriptor (Type) via PSI
        if "Type" in current_frame.columns:
            cur_props = current_frame["Type"].value_counts(normalize=True).to_dict()
            psi_val = compute_psi(self.reference_data.type_proportions, cur_props)
            is_type_drifted = bool(psi_val >= self.psi_threshold)
            if is_type_drifted:
                drifted_features.append("Type")

            feature_results["Type"] = FeatureDriftResult(
                feature_name="Type",
                method="psi",
                statistic=psi_val,
                p_value=None,
                drifted=is_type_drifted,
                reference_mean=0.0,
                current_mean=0.0,
            )

        # 3. Prediction distribution shift
        prediction_shift: dict[str, Any] = {}
        if current_failure_rate is not None:
            baseline_rate = self.reference_data.baseline_failure_rate
            diff = abs(current_failure_rate - baseline_rate)
            # Flag shift if predicted failure rate diverges by > 10 percentage points
            shift_detected = bool(diff > 0.10)
            prediction_shift = {
                "baseline_failure_rate": round(baseline_rate, 4),
                "current_failure_rate": round(current_failure_rate, 4),
                "absolute_shift": round(diff, 4),
                "shift_detected": shift_detected,
            }
        else:
            prediction_shift = {"status": "untracked"}

        overall_drift = len(drifted_features) > 0
        if overall_drift:
            summary = (
                f"Data drift detected in {len(drifted_features)} feature(s): {', '.join(drifted_features)} "
                f"(KS test p < {self.pvalue_threshold} / PSI >= {self.psi_threshold}). {DRIFT_DISCLAIMER}"
            )
        else:
            summary = f"Nominal operation: all {len(feature_results)} evaluated features match baseline distributions."

        return DriftReport(
            timestamp=timestamp,
            drift_detected=overall_drift,
            evaluated_samples=sample_count,
            drifted_features_count=len(drifted_features),
            drifted_features=drifted_features,
            features=feature_results,
            prediction_shift=prediction_shift,
            summary=summary,
            disclaimer=DRIFT_DISCLAIMER,
        )
