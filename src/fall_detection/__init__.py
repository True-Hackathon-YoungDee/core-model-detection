"""Fall detection: MediaPipe pose core.

Native MediaPipe logging is quieted here, before any submodule can import
mediapipe, because the C++ layer reads those variables at load time.
"""

from __future__ import annotations

from .logging_config import quiet_native_logs, setup_logging

quiet_native_logs()

__all__ = ["main", "setup_logging"]


def main() -> int:
    """Console-script entry point (``fall-detection``)."""
    from .cli import main as cli_main

    return cli_main()
