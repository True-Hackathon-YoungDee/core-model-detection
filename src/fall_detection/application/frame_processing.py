"""Port-driven per-frame processing use case."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class PoseInferencePort(Protocol):
    def infer(self, frame: Any, timestamp_ms: int) -> list[Any]: ...


class PersonPipelinePort(Protocol):
    def process(self, persons: list[Any], t_seconds: float) -> list[Any]: ...


class FallStatePort(Protocol):
    def update(
        self, persons: list[Any], t_seconds: float, frame_width: int, frame_height: int
    ) -> list[Any]: ...


class EventSinkPort(Protocol):
    def publish(self, event: Any) -> None: ...


@dataclass(frozen=True)
class FrameProcessingResult:
    persons: list[Any]
    events: list[Any]


class FrameProcessingService:
    """Turn a frame into tracked people and fall events through supplied ports."""

    def __init__(
        self,
        *,
        inference: PoseInferencePort,
        pipeline: PersonPipelinePort,
        fall_state: FallStatePort,
        event_sink: EventSinkPort,
    ) -> None:
        self._inference = inference
        self._pipeline = pipeline
        self._fall_state = fall_state
        self._event_sink = event_sink

    def process(self, frame: Any, timestamp_ms: int) -> FrameProcessingResult:
        t_seconds = timestamp_ms / 1000.0
        persons = self._pipeline.process(self._inference.infer(frame, timestamp_ms), t_seconds)
        height, width = frame.height, frame.width
        events = self._fall_state.update(persons, t_seconds, width, height)
        for event in events:
            self._event_sink.publish(event)
        return FrameProcessingResult(persons=persons, events=events)
