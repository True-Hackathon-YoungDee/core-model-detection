"""Event-delivery adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class CallableEventSink:
    """Adapt a delivery callback to the application event-sink port."""

    def __init__(self, callback: Callable[[Any], None]) -> None:
        self._callback = callback

    def publish(self, event: Any) -> None:
        self._callback(event)
