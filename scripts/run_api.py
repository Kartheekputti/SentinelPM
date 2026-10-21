"""Run SentinelPM FastAPI server with uvicorn.

Usage:
    python -m scripts.run_api [--host HOST] [--port PORT] [--reload]
"""

from __future__ import annotations

import argparse

import uvicorn

from app.core.config import get_settings
from app.core.logging import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SentinelPM API server.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface.")
    parser.add_argument("--port", type=int, default=8000, help="Port number.")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development.")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_level, force=True)

    uvicorn.run(
        "app.api.app:create_app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
