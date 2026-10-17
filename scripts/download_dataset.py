"""Download the AI4I 2020 predictive maintenance dataset into ``data/raw``.

The AI4I dataset is simulated data published by the UCI Machine Learning
Repository, not real factory telemetry.

Usage:
    python -m scripts.download_dataset [--force]
"""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

from app.core.config import get_settings
from app.core.exceptions import DataLoadError
from app.core.logging import get_logger, setup_logging
from app.data.schema import DEFAULT_RAW_FILENAME

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
)
CSV_MEMBER = "ai4i2020.csv"

logger = get_logger(__name__)


def download_dataset(target_dir: Path, *, force: bool = False, timeout: int = 120) -> Path:
    """Download and extract the AI4I dataset CSV.

    Args:
        target_dir: Directory to write the CSV into; created if missing.
        force: Overwrite an existing file when True.
        timeout: Network timeout in seconds.

    Returns:
        Path to the extracted CSV file.

    Raises:
        DataLoadError: If the download or archive extraction fails.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / DEFAULT_RAW_FILENAME
    if destination.exists() and not force:
        logger.info("Dataset already present at %s (use --force to redownload)", destination)
        return destination

    logger.info("Downloading AI4I dataset from %s", DATASET_URL)
    try:
        with urllib.request.urlopen(DATASET_URL, timeout=timeout) as response:
            payload = response.read()
    except OSError as exc:
        msg = f"Failed to download dataset: {exc}"
        raise DataLoadError(msg) from exc

    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive, archive.open(CSV_MEMBER) as member:
            destination.write_bytes(member.read())
    except (zipfile.BadZipFile, KeyError) as exc:
        msg = f"Failed to extract {CSV_MEMBER} from archive: {exc}"
        raise DataLoadError(msg) from exc

    logger.info("Dataset written to %s", destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the AI4I 2020 dataset.")
    parser.add_argument("--force", action="store_true", help="Redownload even if present.")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_level, force=True)
    destination = download_dataset(settings.raw_data_dir, force=args.force)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
