"""Run an offline statistical drift evaluation against a dataset.

Usage:
    python -m scripts.check_drift [--dataset PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.data.loader import load_raw_dataset
from app.monitoring.drift import DriftDetector
from app.monitoring.reference import get_default_reference_data

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate statistical drift on a dataset.")
    parser.add_argument("--dataset", type=Path, default=None, help="Target dataset to evaluate.")
    parser.add_argument("--output", type=Path, default=None, help="Output report JSON path.")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_level, force=True)

    logger.info("Loading reference baseline profile...")
    ref_data = get_default_reference_data(settings)
    detector = DriftDetector(ref_data, pvalue_threshold=settings.drift_pvalue_threshold)

    logger.info("Loading target dataset for drift analysis...")
    df = load_raw_dataset(args.dataset, settings=settings)

    logger.info("Running Kolmogorov-Smirnov and PSI drift tests across %d rows...", len(df))
    report = detector.evaluate_drift(df)

    print()
    print(f"Drift Status: {'[DRIFT DETECTED]' if report.drift_detected else '[NOMINAL]'}")
    print(f"Evaluated Samples: {report.evaluated_samples}")
    print(
        f"Drifted Features ({report.drifted_features_count}): {', '.join(report.drifted_features) or 'None'}"
    )
    print(f"Summary: {report.summary}")
    print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        logger.info("Saved drift report to %s", args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
