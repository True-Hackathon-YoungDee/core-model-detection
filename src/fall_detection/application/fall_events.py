"""Application use case for publishing fall-state decisions."""

from __future__ import annotations

from typing import Any, Protocol


class FallStatePort(Protocol):
    def update(
        self, persons: list[Any], t_seconds: float, frame_width: int, frame_height: int
    ) -> list[Any]: ...


class EventSinkPort(Protocol):
    def publish(self, event: Any) -> None: ...


class FallEventService:
    """Apply a fall-state policy and publish every resulting event."""

    def __init__(self, *, fall_state: FallStatePort, event_sink: EventSinkPort) -> None:
        self._fall_state = fall_state
        self._event_sink = event_sink

    def process(
        self,
        persons: list[Any],
        *,
        t_seconds: float,
        frame_width: int,
        frame_height: int,
    ) -> list[Any]:
        events = self._fall_state.update(persons, t_seconds, frame_width, frame_height)
        for event in events:
            self._event_sink.publish(event)
        return events
