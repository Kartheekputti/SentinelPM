"""Structured logging setup shared by every SentinelPM module."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_file: Path | str | None = None,
    force: bool = False,
) -> None:
    """Configure the root logger for the application.

    Args:
        level: Logging level name (e.g. ``"INFO"``).
        log_file: Optional file path; when given, logs are written to both
            stdout and this file.
        force: Reconfigure even if handlers already exist. Useful in tests.

    Raises:
        ValueError: If ``level`` is not a valid logging level name.
    """
    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        msg = f"Invalid log level: {level!r}"
        raise ValueError(msg)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))

    logging.basicConfig(
        level=numeric_level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=handlers,
        force=force,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    return logging.getLogger(name)
