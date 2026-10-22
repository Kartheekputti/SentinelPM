"""CLI launcher script for the SentinelPM Streamlit dashboard."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from app.core.config import get_settings


def main() -> None:
    """Launch Streamlit dashboard."""
    settings = get_settings()
    app_path = Path(__file__).resolve().parent.parent / "app" / "dashboard" / "dashboard.py"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(settings.dashboard_port),
        "--server.address",
        "0.0.0.0",
    ]
    print(f"Starting SentinelPM dashboard on http://localhost:{settings.dashboard_port} ...")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nSentinelPM dashboard stopped.")


if __name__ == "__main__":
    main()
