"""Reference baseline dataset management for drift detection.

Extracts baseline statistics and sensor distributions from training data.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import Settings, get_settings
from app.core.exceptions import MonitoringError
from app.data.schema import SENSOR_COLUMNS, TARGET_COLUMN


@dataclass
class ColumnReferenceStats:
    """Baseline distribution statistics for a single numeric sensor."""

    name: str
    count: int
    mean: float
    std: float
    min: float
    max: float
    q25: float
    q50: float
    q75: float
    sample_values: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "min": round(self.min, 4),
            "max": round(self.max, 4),
            "q25": round(self.q25, 4),
            "q50": round(self.q50, 4),
            "q75": round(self.q75, 4),
            "sample_values": [round(x, 4) for x in self.sample_values],
        }


@dataclass
class ReferenceData:
    """Baseline reference distribution profile."""

    n_samples: int
    baseline_failure_rate: float
    sensor_stats: dict[str, ColumnReferenceStats]
    type_proportions: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "baseline_failure_rate": round(self.baseline_failure_rate, 4),
            "sensor_stats": {k: v.to_dict() for k, v in self.sensor_stats.items()},
            "type_proportions": {k: round(v, 4) for k, v in self.type_proportions.items()},
        }

    def save(self, path: Path | str) -> Path:
        """Save reference profile to JSON."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return dest

    @classmethod
    def load(cls, path: Path | str) -> ReferenceData:
        """Load reference profile from JSON."""
        src = Path(path)
        if not src.exists():
            msg = f"Reference profile not found at {src}"
            raise MonitoringError(msg)
        with src.open(encoding="utf-8") as f:
            data = json.load(f)

        stats: dict[str, ColumnReferenceStats] = {}
        for col, s in data.get("sensor_stats", {}).items():
            stats[col] = ColumnReferenceStats(
                name=s["name"],
                count=s["count"],
                mean=s["mean"],
                std=s["std"],
                min=s["min"],
                max=s["max"],
                q25=s["q25"],
                q50=s["q50"],
                q75=s["q75"],
                sample_values=s["sample_values"],
            )

        return cls(
            n_samples=data["n_samples"],
            baseline_failure_rate=data["baseline_failure_rate"],
            sensor_stats=stats,
            type_proportions=data["type_proportions"],
        )


def build_reference_data(
    frame: pd.DataFrame,
    *,
    sample_size: int = 1000,
    random_state: int = 42,
) -> ReferenceData:
    """Build a reference distribution profile from training data.

    Args:
        frame: Dataframe containing at least the sensor columns.
        sample_size: Subsample size for distribution testing.
        random_state: Random sampling seed.

    Returns:
        A populated :class:`ReferenceData` instance.
    """
    if len(frame) == 0:
        msg = "Cannot build reference profile from an empty dataframe"
        raise MonitoringError(msg)

    # Subsample if large
    if len(frame) > sample_size:
        sample_df = frame.sample(n=sample_size, random_state=random_state)
    else:
        sample_df = frame.copy()

    # Target failure rate if present
    failure_rate = float(frame[TARGET_COLUMN].mean()) if TARGET_COLUMN in frame.columns else 0.0339

    sensor_stats: dict[str, ColumnReferenceStats] = {}
    for col in SENSOR_COLUMNS:
        if col in sample_df.columns:
            series = sample_df[col].dropna().astype(float)
            sensor_stats[col] = ColumnReferenceStats(
                name=col,
                count=len(series),
                mean=float(series.mean()),
                std=float(series.std()),
                min=float(series.min()),
                max=float(series.max()),
                q25=float(series.quantile(0.25)),
                q50=float(series.quantile(0.50)),
                q75=float(series.quantile(0.75)),
                sample_values=series.tolist(),
            )

    # Type proportions
    type_props: dict[str, float] = {}
    if "Type" in sample_df.columns:
        counts = sample_df["Type"].value_counts(normalize=True).to_dict()
        type_props = {str(k): float(v) for k, v in counts.items()}
    else:
        type_props = {"L": 0.6, "M": 0.3, "H": 0.1}

    return ReferenceData(
        n_samples=len(sample_df),
        baseline_failure_rate=failure_rate,
        sensor_stats=sensor_stats,
        type_proportions=type_props,
    )


def get_default_reference_data(settings: Settings | None = None) -> ReferenceData:
    """Load cached reference data from artifacts or compute from raw dataset."""
    active_settings = settings or get_settings()
    ref_file = active_settings.reports_dir / "reference_profile.json"

    if ref_file.exists():
        with contextlib.suppress(Exception):
            return ReferenceData.load(ref_file)

    raw_file = active_settings.raw_data_dir / "ai4i2020.csv"
    if raw_file.exists():
        df = pd.read_csv(raw_file)
        ref = build_reference_data(df)
        with contextlib.suppress(Exception):
            ref.save(ref_file)
        return ref

    # Synthetic baseline fallback if neither file is available
    synthetic_stats = {
        col: ColumnReferenceStats(
            name=col,
            count=100,
            mean=300.0,
            std=2.0,
            min=290.0,
            max=310.0,
            q25=298.0,
            q50=300.0,
            q75=302.0,
            sample_values=list(np.linspace(290.0, 310.0, 100)),
        )
        for col in SENSOR_COLUMNS
    }
    return ReferenceData(
        n_samples=100,
        baseline_failure_rate=0.0339,
        sensor_stats=synthetic_stats,
        type_proportions={"L": 0.6, "M": 0.3, "H": 0.1},
    )
