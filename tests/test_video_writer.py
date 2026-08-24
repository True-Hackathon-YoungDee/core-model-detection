import cv2
import numpy as np

from fall_detection.runner import VideoWriter


class _FakeCvWriter:
    def __init__(self, opened: bool = True) -> None:
        self.frames = []
        self.released = False
        self._opened = opened

    def isOpened(self) -> bool:
        return self._opened

    def write(self, frame) -> None:
        self.frames.append(frame)

    def release(self) -> None:
        self.released = True


def test_disabled_when_no_path():
    writer = VideoWriter(None, fps=30.0)
    assert writer.enabled is False
    writer.write(np.zeros((10, 10, 3), dtype=np.uint8))  # must not raise


def test_opens_lazily_once_and_writes_every_frame(monkeypatch, tmp_path):
    fake = _FakeCvWriter()
    opened_with = []

    def fake_video_writer(path, fourcc, fps, size):
        opened_with.append((path, fourcc, fps, size))
        return fake

    monkeypatch.setattr(cv2, "VideoWriter", fake_video_writer)

    out_path = tmp_path / "out.mp4"
    writer = VideoWriter(str(out_path), fps=25.0)
    assert writer.enabled is True

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    writer.write(frame)
    writer.write(frame)

    assert len(opened_with) == 1  # only opened on the first frame
    path, _fourcc, fps, size = opened_with[0]
    assert path == str(out_path)
    assert fps == 25.0
    assert size == (1920, 1080)
    assert len(fake.frames) == 2


def test_creates_parent_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(cv2, "VideoWriter", lambda *a: _FakeCvWriter())
    out_path = tmp_path / "nested" / "dir" / "out.mp4"
    writer = VideoWriter(str(out_path), fps=30.0)
    writer.write(np.zeros((10, 10, 3), dtype=np.uint8))
    assert out_path.parent.is_dir()


def test_close_releases_underlying_writer(monkeypatch, tmp_path):
    fake = _FakeCvWriter()
    monkeypatch.setattr(cv2, "VideoWriter", lambda *a: fake)
    writer = VideoWriter(str(tmp_path / "out.mp4"), fps=30.0)
    writer.write(np.zeros((10, 10, 3), dtype=np.uint8))
    writer.close()
    assert fake.released is True


def test_close_on_never_opened_writer_does_not_raise(tmp_path):
    writer = VideoWriter(str(tmp_path / "out.mp4"), fps=30.0)
    writer.close()  # never wrote a frame, so cv2.VideoWriter was never created


def test_disables_itself_if_open_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(cv2, "VideoWriter", lambda *a: _FakeCvWriter(opened=False))
    writer = VideoWriter(str(tmp_path / "out.mp4"), fps=30.0)
    writer.write(np.zeros((10, 10, 3), dtype=np.uint8))  # must not raise
    assert writer.enabled is False
