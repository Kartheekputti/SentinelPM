"""Pydantic schemas for SentinelPM inference.

Validates sensor telemetry bounds and strictly prevents target label leakage.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.exceptions import InferenceError
from app.data.schema import NUMERIC_BOUNDS, TYPE_CATEGORIES
from app.features.definitions import FORBIDDEN_FEATURE_COLUMNS

RiskLevel = Literal["low", "medium", "high"]


class SensorRecord(BaseModel):
    """Machine sensor telemetry record for prediction.

    Accepts both canonical display names and model-safe snake_case identifiers.
    Strictly forbids target labels and target-derived indicators to prevent leakage.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    air_temperature_k: float = Field(
        ...,
        alias="Air temperature [K]",
        ge=NUMERIC_BOUNDS["Air temperature [K]"][0],
        le=NUMERIC_BOUNDS["Air temperature [K]"][1],
        description="Air / ambient temperature in Kelvin.",
        examples=[298.1],
    )
    process_temperature_k: float = Field(
        ...,
        alias="Process temperature [K]",
        ge=NUMERIC_BOUNDS["Process temperature [K]"][0],
        le=NUMERIC_BOUNDS["Process temperature [K]"][1],
        description="Process temperature in Kelvin.",
        examples=[308.6],
    )
    rotational_speed_rpm: float = Field(
        ...,
        alias="Rotational speed [rpm]",
        ge=NUMERIC_BOUNDS["Rotational speed [rpm]"][0],
        le=NUMERIC_BOUNDS["Rotational speed [rpm]"][1],
        description="Spindle rotational speed in RPM.",
        examples=[1551.0],
    )
    torque_nm: float = Field(
        ...,
        alias="Torque [Nm]",
        ge=NUMERIC_BOUNDS["Torque [Nm]"][0],
        le=NUMERIC_BOUNDS["Torque [Nm]"][1],
        description="Applied torque in Newton-meters.",
        examples=[42.8],
    )
    tool_wear_min: float = Field(
        ...,
        alias="Tool wear [min]",
        ge=NUMERIC_BOUNDS["Tool wear [min]"][0],
        le=NUMERIC_BOUNDS["Tool wear [min]"][1],
        description="Cumulative tool wear time in minutes.",
        examples=[0.0],
    )
    type: Literal["L", "M", "H"] = Field(
        ...,
        alias="Type",
        description="Product quality variant (L=Low, M=Medium, H=High).",
        examples=["M"],
    )

    @field_validator("type")
    @classmethod
    def _validate_type(cls, val: str) -> str:
        upper = val.upper()
        if upper not in TYPE_CATEGORIES:
            msg = f"Type must be one of {TYPE_CATEGORIES}, got {val!r}"
            raise ValueError(msg)
        return upper

    @model_validator(mode="before")
    @classmethod
    def _reject_forbidden_columns(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for forbidden in ("Machine failure", *FORBIDDEN_FEATURE_COLUMNS):
                if forbidden in data or forbidden.lower().replace(" ", "_") in data:
                    msg = (
                        f"Target label or derived failure indicator {forbidden!r} is "
                        "strictly forbidden in prediction features to prevent data leakage."
                    )
                    raise ValueError(msg)
        return data

    def to_raw_dict(self) -> dict[str, Any]:
        """Convert to dictionary with canonical raw column names expected by the pipeline."""
        return {
            "Air temperature [K]": self.air_temperature_k,
            "Process temperature [K]": self.process_temperature_k,
            "Rotational speed [rpm]": self.rotational_speed_rpm,
            "Torque [Nm]": self.torque_nm,
            "Tool wear [min]": self.tool_wear_min,
            "Type": self.type,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert single record to a DataFrame matching the raw feature schema."""
        return pd.DataFrame([self.to_raw_dict()])


class BatchSensorInput(BaseModel):
    """Batch sensor payload with collection of records."""

    records: list[SensorRecord] = Field(
        ...,
        min_length=1,
        description="List of machine sensor records.",
    )


class PredictionResult(BaseModel):
    """Standardized prediction outcome schema."""

    model_config = ConfigDict(frozen=True)

    prediction_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique prediction identifier for correlation and tracing.",
    )
    model_version: str = Field(..., description="Version of the model that generated prediction.")
    model_name: str = Field(..., description="Name of the model estimator.")
    predicted_failure: bool = Field(..., description="True if machine is predicted to fail.")
    failure_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated probability of failure [0.0 - 1.0].",
    )
    risk_level: RiskLevel = Field(
        ...,
        description="Categorical risk tier derived from probability and threshold.",
    )
    threshold: float = Field(..., description="Decision threshold applied.")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="Prediction timestamp (UTC ISO-8601).",
    )


def records_to_dataframe(records: list[SensorRecord]) -> pd.DataFrame:
    """Convert a sequence of SensorRecord objects into a raw feature DataFrame."""
    if not records:
        msg = "Cannot convert empty records list to DataFrame"
        raise InferenceError(msg)
    return pd.DataFrame([r.to_raw_dict() for r in records])
