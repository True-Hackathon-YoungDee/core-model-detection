from fall_detection.application.fall_events import FallEventService


class _FallState:
    def update(self, persons, t_seconds, frame_width, frame_height):
        assert persons == ["tracked"]
        assert t_seconds == 2.5
        assert (frame_width, frame_height) == (320, 240)
        return ["event-a", "event-b"]


class _Sink:
    def __init__(self) -> None:
        self.events = []

    def publish(self, event) -> None:
        self.events.append(event)


def test_fall_event_service_returns_and_publishes_manager_events() -> None:
    """Fails if a migrated callback no longer sends each manager event to its sink."""
    sink = _Sink()
    service = FallEventService(fall_state=_FallState(), event_sink=sink)

    events = service.process(["tracked"], t_seconds=2.5, frame_width=320, frame_height=240)

    assert events == ["event-a", "event-b"]
    assert sink.events == ["event-a", "event-b"]
