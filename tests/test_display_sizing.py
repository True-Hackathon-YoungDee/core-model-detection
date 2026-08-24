import cv2
import numpy as np

from fall_detection.runner import Display


def test_first_show_makes_window_resizable_and_caps_size(monkeypatch):
    named = []
    resized = []
    monkeypatch.setattr(cv2, "namedWindow", lambda *a: named.append(a))
    monkeypatch.setattr(cv2, "resizeWindow", lambda *a: resized.append(a))
    monkeypatch.setattr(cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *a: -1)

    display = Display(True, max_width=1280)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    display.show(frame)

    assert named == [(display.window, cv2.WINDOW_NORMAL)]
    assert resized == [(display.window, 1280, 720)]


def test_only_sizes_window_once(monkeypatch):
    named = []
    resized = []
    monkeypatch.setattr(cv2, "namedWindow", lambda *a: named.append(a))
    monkeypatch.setattr(cv2, "resizeWindow", lambda *a: resized.append(a))
    monkeypatch.setattr(cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *a: -1)

    display = Display(True, max_width=1280)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    display.show(frame)
    display.show(frame)

    assert len(named) == 1
    assert len(resized) == 1


def test_small_frame_is_not_shrunk(monkeypatch):
    resized = []
    monkeypatch.setattr(cv2, "namedWindow", lambda *a: None)
    monkeypatch.setattr(cv2, "resizeWindow", lambda *a: resized.append(a))
    monkeypatch.setattr(cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *a: -1)

    display = Display(True, max_width=1280)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    display.show(frame)

    assert resized == []
