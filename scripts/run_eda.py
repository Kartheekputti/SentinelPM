"""Generate the exploratory data analysis report.

Writes the report to ``artifacts/reports/eda_report.json``.

Usage:
    python -m scripts.run_eda [--dataset PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.data.loader import load_raw_dataset
from app.features.eda import generate_eda_report

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run exploratory data analysis.")
    parser.add_argument("--dataset", type=Path, default=None, help="Explicit dataset path.")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_level, force=True)

    frame = load_raw_dataset(args.dataset, settings=settings)
    report = generate_eda_report(frame)
    print(report.summary())

    reports_dir = settings.artifacts_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "eda_report.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )
    logger.info("EDA report written to %s", reports_dir / "eda_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
