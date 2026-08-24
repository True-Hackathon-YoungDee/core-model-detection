"""Logging setup for the whole package.

Nothing in this package prints; every message goes through ``logging`` so the
caller decides verbosity, destination and format. MediaPipe's C++ core logs via
glog, which never reaches Python logging, so it is quieted through environment
variables that must be set *before* ``import mediapipe`` -- see the package
``__init__``.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import time

DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"
DATE_FORMAT = "%H:%M:%S"

_configured = False

# cv2's bundled Qt (highgui) ships without a Wayland platform plugin and
# without its own fonts -- checked in this order for a real system font dir.
_LINUX_FONT_DIR_CANDIDATES = ("/usr/share/fonts", "/usr/local/share/fonts")


def quiet_native_logs() -> None:
    """Suppress MediaPipe/TensorFlow/Qt C++ chatter. Safe to call repeatedly."""
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    if sys.platform.startswith("linux"):
        # cv2's Qt has no Wayland plugin, so it always ends up on xcb (via
        # XWayland) anyway -- set it explicitly for clarity.
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
        # Qt's GNOME/Wayland bootstrap check prints "Ignoring
        # XDG_SESSION_TYPE=wayland on Gnome" unconditionally to stderr,
        # *before* its logging-category system (and QT_LOGGING_RULES) is
        # even initialised -- confirmed via `strings` on the bundled
        # libQt5Gui and by testing that QT_LOGGING_RULES alone doesn't
        # silence it. Since we're on xcb/XWayland regardless, tell it we're
        # x11 to begin with and it skips the check. Process-local only --
        # does not touch the real desktop session.
        if os.environ.get("XDG_SESSION_TYPE") == "wayland":
            os.environ["XDG_SESSION_TYPE"] = "x11"
        # The QFontDatabase "cannot find font directory" warning ignores
        # QT_QPA_FONTDIR (set below) and prints regardless -- it's a
        # category-based qCWarning, so this does catch it.
        os.environ.setdefault("QT_LOGGING_RULES", "*.warning=false")
        for font_dir in _LINUX_FONT_DIR_CANDIDATES:
            if os.path.isdir(font_dir):
                os.environ.setdefault("QT_QPA_FONTDIR", font_dir)
                break


def setup_logging(
    level: str | int = "INFO",
    log_file: str | None = None,
    fmt: str = DEFAULT_FORMAT,
) -> None:
    """Configure root logging once; later calls only adjust the level."""
    global _configured

    quiet_native_logs()

    if isinstance(level, str):
        numeric = logging.getLevelNamesMapping().get(level.upper())
        if numeric is None:
            raise ValueError(f"invalid log level: {level!r}")
    else:
        numeric = level

    root = logging.getLogger()
    if _configured:
        root.setLevel(numeric)
        return

    formatter = logging.Formatter(fmt, DATE_FORMAT)
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root.setLevel(numeric)
    for noisy in ("absl", "matplotlib", "matplotlib.font_manager", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.captureWarnings(True)
    _configured = True

    if log_file:
        logging.getLogger(__name__).info("logging to stderr and %s", log_file)


class RateLimiter:
    """Lets a hot loop log at most once per interval per key."""

    def __init__(self, interval: float = 1.0) -> None:
        self._interval = interval
        self._last: dict[str, float] = {}

    def ready(self, key: str) -> bool:
        now = time.monotonic()
        previous = self._last.get(key)
        if previous is not None and now - previous < self._interval:
            return False
        self._last[key] = now
        return True


@contextlib.contextmanager
def suppress_native_stderr():
    """Silence C++ logs that go straight to file descriptor 2.

    MediaPipe's TFLite layer writes several lines per model at creation time,
    before glog is initialised -- which is why ``GLOG_minloglevel`` does not
    catch them. The cascade builds five models, so it is five times the noise.

    Only the fd is redirected, and only around ``create_from_options``: our own
    handlers keep writing through the ``sys.stderr`` object. At DEBUG the
    redirect is skipped, because then the native chatter is worth having.
    """
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        yield
        return
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stderr.flush()
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)
