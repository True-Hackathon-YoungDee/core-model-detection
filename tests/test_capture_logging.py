import logging

import cv2

from fall_detection.runner import _configure_capture_logging


def test_capture_logging_silent_by_default(monkeypatch):
    logging.getLogger().setLevel(logging.INFO)
    calls = []
    monkeypatch.setattr(cv2.utils.logging, "setLogLevel", calls.append)

    _configure_capture_logging()

    assert calls == [cv2.utils.logging.LOG_LEVEL_SILENT]


def test_capture_logging_verbose_at_debug(monkeypatch):
    logging.getLogger().setLevel(logging.DEBUG)
    calls = []
    monkeypatch.setattr(cv2.utils.logging, "setLogLevel", calls.append)

    try:
        _configure_capture_logging()
    finally:
        logging.getLogger().setLevel(logging.INFO)

    assert calls == [cv2.utils.logging.LOG_LEVEL_DEBUG]
