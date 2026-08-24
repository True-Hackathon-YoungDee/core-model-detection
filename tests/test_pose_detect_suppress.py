"""First-detect-call stderr suppression around the one-shot MediaPipe
landmark_projection_calculator NORM_RECT warning (see logging_config.py).

Bypasses PoseDetector.__init__ (it builds a real landmarker from a model
file) and stubs just the pieces detect_video touches.
"""

import contextlib

from fall_detection.logging_config import RateLimiter
from fall_detection.pose import PoseDetector


class _FakeLandmarker:
    def __init__(self):
        self.calls = 0

    def detect_for_video(self, image, timestamp_ms):
        self.calls += 1
        return "result"


def _bare_detector():
    detector = object.__new__(PoseDetector)
    detector._last_timestamp_ms = -1
    detector._limiter = RateLimiter(1.0)
    detector._landmarker = _FakeLandmarker()
    detector._first_detect_pending = True
    return detector


def test_first_detect_call_is_wrapped_in_suppress_native_stderr(monkeypatch):
    calls = []

    @contextlib.contextmanager
    def fake_suppress():
        calls.append("enter")
        yield
        calls.append("exit")

    monkeypatch.setattr("fall_detection.pose.suppress_native_stderr", fake_suppress)
    detector = _bare_detector()

    detector.detect_video(_frame(), 1)

    assert calls == ["enter", "exit"]
    assert detector._first_detect_pending is False


def test_second_detect_call_is_not_wrapped(monkeypatch):
    calls = []

    @contextlib.contextmanager
    def fake_suppress():
        calls.append("enter")
        yield

    monkeypatch.setattr("fall_detection.pose.suppress_native_stderr", fake_suppress)
    detector = _bare_detector()

    detector.detect_video(_frame(), 1)
    detector.detect_video(_frame(), 2)

    assert calls == ["enter"]


def _frame():
    import numpy as np

    return np.zeros((4, 4, 3), dtype=np.uint8)
