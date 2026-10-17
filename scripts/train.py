"""Train and evaluate candidate models.

Usage:
    python -m scripts.train [--dataset PATH] [--config PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.data.loader import load_raw_dataset
from app.training.config import load_training_config
from app.training.trainer import run_training

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train SentinelPM models.")
    parser.add_argument("--dataset", type=Path, default=None, help="Explicit dataset path.")
    parser.add_argument("--config", type=Path, default=None, help="Training config path.")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_level, force=True)

    frame = load_raw_dataset(args.dataset, settings=settings)
    config = load_training_config(args.config)
    report = run_training(frame, config, settings=settings)

    print(report.summary())
    logger.info("Training report written to %s", report.report_path)
    print()
    print(json.dumps({"best_model": report.best_model}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
