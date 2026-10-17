"""Load, validate and summarise the raw dataset.

Writes machine-readable reports under ``artifacts/reports`` so results can be
reproduced. Exits with status 1 when the dataset is invalid.

Usage:
    python -m scripts.validate_dataset [--dataset PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.data.loader import load_raw_dataset
from app.data.quality import generate_quality_report
from app.data.validation import validate_dataset

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the raw dataset.")
    parser.add_argument("--dataset", type=Path, default=None, help="Explicit dataset path.")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_level, force=True)

    frame = load_raw_dataset(args.dataset, settings=settings)
    report = validate_dataset(frame)
    quality = generate_quality_report(frame)

    print(report.summary())
    print()
    print(quality.summary())

    reports_dir = settings.artifacts_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "validation_report.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )
    (reports_dir / "data_quality_report.json").write_text(
        json.dumps(quality.to_dict(), indent=2), encoding="utf-8"
    )
    logger.info("Reports written to %s", reports_dir)

    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
