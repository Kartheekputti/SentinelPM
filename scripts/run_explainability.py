"""Generate an offline SHAP explanation report.

Usage:
    python -m scripts.run_explainability [--model PATH] [--dataset PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.data.loader import load_raw_dataset
from app.explainability.explainer import ModelExplainer
from app.explainability.report import generate_global_explanation, save_explanation_report
from app.training.artifacts import load_model

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate SHAP explanation report.")
    parser.add_argument("--model", type=Path, default=None, help="Trained model path.")
    parser.add_argument("--dataset", type=Path, default=None, help="Raw dataset path.")
    parser.add_argument("--output", type=Path, default=None, help="Output report JSON path.")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_level, force=True)

    model_file = args.model or (settings.models_dir / f"{settings.model_name}.joblib")
    report_file = args.output or (settings.reports_dir / "shap_explanation_report.json")

    logger.info("Loading model from %s", model_file)
    pipeline = load_model(model_file)

    logger.info("Loading dataset...")
    df = load_raw_dataset(args.dataset, settings=settings)

    explainer = ModelExplainer(
        pipeline,
        model_name=settings.model_name,
        model_version=settings.model_version,
        background_data=df,
    )

    logger.info("Computing global SHAP feature importance...")
    global_exp = generate_global_explanation(explainer, df, max_samples=200)

    save_explanation_report(global_exp, report_file)
    logger.info("SHAP report successfully saved to %s", report_file)
    print(global_exp.summary_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
