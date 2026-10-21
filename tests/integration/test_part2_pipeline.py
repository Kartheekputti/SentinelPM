"""Full Part 2 lifecycle integration test: registry -> inference -> explain -> monitor."""

from __future__ import annotations

import pytest

from app.core.config import get_settings
from app.data.loader import load_raw_dataset
from app.explainability.explainer import ModelExplainer
from app.inference.service import InferenceService
from app.monitoring.alerts import AlertManager
from app.monitoring.drift import DriftDetector
from app.monitoring.reference import build_reference_data
from app.monitoring.tracker import MonitoringTracker
from app.registry.manager import ModelRegistryManager


@pytest.mark.integration
def test_full_part2_pipeline_lifecycle() -> None:
    """Test registry loading, inference, SHAP explanations, drift monitoring and alerts."""
    settings = get_settings()

    # 1. Verify model registry loads approved model
    registry = ModelRegistryManager(settings=settings)
    pipeline, metadata = registry.load_approved_model()
    assert metadata.model_name in ("random_forest", "xgboost")
    assert metadata.threshold > 0.0

    # 2. Inference service
    inference_service = InferenceService(pipeline, metadata, settings=settings)
    df = load_raw_dataset(settings=settings)
    sample_rows = df.head(5)

    records = [
        {
            "Air temperature [K]": float(row["Air temperature [K]"]),
            "Process temperature [K]": float(row["Process temperature [K]"]),
            "Rotational speed [rpm]": float(row["Rotational speed [rpm]"]),
            "Torque [Nm]": float(row["Torque [Nm]"]),
            "Tool wear [min]": float(row["Tool wear [min]"]),
            "Type": str(row["Type"]),
        }
        for _, row in sample_rows.iterrows()
    ]

    predictions = inference_service.predict_batch(records)
    assert len(predictions) == 5
    for p in predictions:
        assert 0.0 <= p.failure_probability <= 1.0
        assert p.risk_level in ("low", "medium", "high")

    # 3. SHAP explainer
    explainer = ModelExplainer(
        pipeline,
        model_name=metadata.model_name,
        model_version=metadata.model_version,
    )
    explanation = explainer.explain_record(records[0])
    assert len(explanation.contributions) == len(metadata.feature_names)
    assert explanation.predicted_probability == predictions[0].failure_probability

    # 4. Drift monitoring and alerts
    ref_data = build_reference_data(df, sample_size=1000, random_state=42)
    drift_detector = DriftDetector(ref_data, pvalue_threshold=0.01)
    tracker = MonitoringTracker(settings=settings)
    alert_manager = AlertManager(settings=settings)

    # Nominal batch -> no drift, no alerts
    nominal_df = df.sample(n=50, random_state=101)
    drift_report = drift_detector.evaluate_drift(nominal_df)
    assert drift_report.drift_detected is False
    alerts = alert_manager.evaluate(tracker, drift_report)
    assert len(alerts) == 0

    # Shifted batch -> drift detected, alert triggered
    shifted_df = nominal_df.copy()
    shifted_df["Rotational speed [rpm]"] = shifted_df["Rotational speed [rpm]"] + 500.0
    shifted_report = drift_detector.evaluate_drift(shifted_df)
    assert shifted_report.drift_detected is True
    assert "Rotational speed [rpm]" in shifted_report.drifted_features

    drift_alerts = alert_manager.evaluate(tracker, shifted_report)
    assert len(drift_alerts) >= 1
    assert drift_alerts[0].alert_type == "data_drift_detected"
