"""CSV loading utilities.

Loading is intentionally separate from training: any component can read the raw
dataset without pulling in model code. Raw files are never modified in place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import Settings, get_settings
from app.core.exceptions import DataLoadError
from app.core.logging import get_logger
from app.data.schema import DEFAULT_RAW_FILENAME

logger = get_logger(__name__)


def load_csv(path: Path | str, **read_csv_kwargs: Any) -> pd.DataFrame:
    """Load a CSV file into a :class:`pandas.DataFrame`.

    Args:
        path: Path to the CSV file.
        **read_csv_kwargs: Extra keyword arguments forwarded to
            :func:`pandas.read_csv`.

    Returns:
        The loaded dataframe.

    Raises:
        DataLoadError: If the file is missing, empty, unreadable or not
            parseable as CSV.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        msg = f"Dataset file not found: {csv_path}"
        raise DataLoadError(msg)
    if not csv_path.is_file():
        msg = f"Dataset path is not a file: {csv_path}"
        raise DataLoadError(msg)

    kwargs: dict[str, Any] = {"dtype": None}
    kwargs.update(read_csv_kwargs)
    try:
        frame = pd.read_csv(csv_path, **kwargs)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        msg = f"Failed to parse CSV file {csv_path}: {exc}"
        raise DataLoadError(msg) from exc
    except UnicodeDecodeError as exc:
        msg = f"Failed to decode CSV file {csv_path}: {exc}"
        raise DataLoadError(msg) from exc

    if frame.empty:
        msg = f"Dataset file contains no rows: {csv_path}"
        raise DataLoadError(msg)

    logger.info("Loaded %s rows x %s columns from %s", len(frame), frame.shape[1], csv_path)
    return frame


def load_raw_dataset(
    path: Path | str | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Load the raw predictive-maintenance dataset.

    Args:
        path: Explicit path to the dataset. When omitted, the file is resolved
            from the configured raw data directory.
        settings: Optional settings override; falls back to the cached settings.

    Returns:
        The raw dataset as a dataframe, unchanged from disk.

    Raises:
        DataLoadError: If the dataset cannot be found or read.
    """
    resolved = Path(path) if path is not None else _default_raw_path(settings)
    return load_csv(resolved)


def _default_raw_path(settings: Settings | None) -> Path:
    active = settings or get_settings()
    return active.raw_data_dir / DEFAULT_RAW_FILENAME
