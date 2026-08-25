from dataclasses import dataclass


@dataclass
class _Frame:
    width: int = 640
    height: int = 480


class _PoseInference:
    def infer(self, frame: _Frame, timestamp_ms: int) -> list[str]:
        assert frame.width == 640
        assert timestamp_ms == 1_250
        return ["person-1"]


class _Pipeline:
    def process(self, persons: list[str], t_seconds: float) -> list[str]:
        assert persons == ["person-1"]
        assert t_seconds == 1.25
        return ["tracked-person-1"]


class _FallState:
    def update(
        self, persons: list[str], t_seconds: float, frame_width: int, frame_height: int
    ) -> list[str]:
        assert persons == ["tracked-person-1"]
        assert t_seconds == 1.25
        assert (frame_width, frame_height) == (640, 480)
        return ["detected-event"]


class _EventSink:
    def __init__(self) -> None:
        self.published: list[str] = []

    def publish(self, event: str) -> None:
        self.published.append(event)


def test_process_frame_returns_and_publishes_fall_events() -> None:
    """Fails if a use case drops detected events or bypasses a supplied port."""
    from fall_detection.application.frame_processing import FrameProcessingService

    sink = _EventSink()
    service = FrameProcessingService(
        inference=_PoseInference(),
        pipeline=_Pipeline(),
        fall_state=_FallState(),
        event_sink=sink,
    )

    result = service.process(_Frame(), timestamp_ms=1_250)

    assert result.persons == ["tracked-person-1"]
    assert result.events == ["detected-event"]
    assert sink.published == ["detected-event"]
