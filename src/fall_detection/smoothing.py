"""One-Euro smoothing of pose landmarks.

Raw landmarks jitter frame to frame because the tracked ROI wobbles. A plain
EMA trades that jitter for lag during fast motion -- which is exactly the motion
a fall detector cares about -- so the adaptive One-Euro filter is used instead:
its cutoff rises with the estimated speed of the signal.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from typing import Any, Iterable

logger = logging.getLogger(__name__)

_AXES = ("x", "y", "z")


class OneEuroFilter:
    """Adaptive low-pass filter for one scalar signal."""

    def __init__(
        self,
        t0: float,
        x0: float,
        dx0: float = 0.0,
        min_cutoff: float = 1.0,
        beta: float = 0.005,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        if dt <= 0.0:
            return 1.0
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, t: float, x: float) -> float:
        t = float(t)
        x = float(x)
        dt = t - self.t_prev
        if dt <= 0.0:
            # Duplicate or out-of-order timestamp: hold the last estimate.
            return self.x_prev

        dx = (x - self.x_prev) / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        edx = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(edx)
        alpha = self._alpha(cutoff, dt)
        x_hat = alpha * x + (1.0 - alpha) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = edx
        self.t_prev = t
        return x_hat


class LandmarkSmoother:
    """Keeps one One-Euro filter per (person, landmark, axis)."""

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.005,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._filters: dict[tuple[int, int, str], OneEuroFilter] = {}

    def smooth(
        self, person_id: int, landmarks: Iterable[Any], t_seconds: float
    ) -> list[Any]:
        """Return copies of ``landmarks`` with x/y/z filtered."""
        smoothed = []
        for index, landmark in enumerate(landmarks):
            updates: dict[str, float] = {}
            for axis in _AXES:
                value = getattr(landmark, axis, None)
                if value is None:
                    continue
                key = (person_id, index, axis)
                state = self._filters.get(key)
                if state is None:
                    state = OneEuroFilter(
                        t_seconds,
                        value,
                        min_cutoff=self.min_cutoff,
                        beta=self.beta,
                        d_cutoff=self.d_cutoff,
                    )
                    self._filters[key] = state
                    updates[axis] = float(value)
                else:
                    updates[axis] = state.filter(t_seconds, value)
            smoothed.append(dataclasses.replace(landmark, **updates))
        return smoothed

    def forget(self, person_id: int) -> None:
        """Drop filter state for a track that died, so a re-entering person
        does not inherit a stale position."""
        stale = [key for key in self._filters if key[0] == person_id]
        for key in stale:
            del self._filters[key]
        if stale:
            logger.debug("dropped %d filters for person %d", len(stale), person_id)

    def reset(self) -> None:
        self._filters.clear()
