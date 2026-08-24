"""Stable person ids across frames.

MediaPipe Pose Landmarker has no re-identification: with ``num_poses > 1`` the
index of a person inside ``result.pose_landmarks`` can swap between frames when
subjects cross or overlap. Greedy nearest-centroid matching is enough to keep
ids steady for smoothing and for downstream per-person state, and it needs no
extra dependency. The interface is small on purpose so a Hungarian assignment
can replace the matching later without touching callers.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

logger = logging.getLogger(__name__)


@dataclass
class _Track:
    person_id: int
    centroid: tuple[float, float]
    first_seen: float
    last_seen: float
    frames: int = field(default=1)


class IdentityTracker:
    """Assigns persistent ids to per-frame centroids (normalized 0..1 coords)."""

    def __init__(
        self,
        max_distance: float = 0.15,
        max_unseen_s: float = 2.5,
        on_lost: Callable[[int], None] | None = None,
    ) -> None:
        if (
            isinstance(max_unseen_s, bool)
            or not isinstance(max_unseen_s, (int, float))
            or not math.isfinite(max_unseen_s)
            or max_unseen_s <= 0.0
        ):
            raise ValueError("max_unseen_s must be a positive finite number")
        self.max_distance = max_distance
        self.max_unseen_s = float(max_unseen_s)
        self._on_lost = on_lost
        self._tracks: dict[int, _Track] = {}
        self._next_id = 0

    @property
    def active_ids(self) -> list[int]:
        return sorted(self._tracks)

    def assign(self, centroids: Sequence[tuple[float, float]], now: float) -> list[int]:
        """Return one id per centroid, in the same order."""
        self._age_out(set(), now)
        assigned: list[int | None] = [None] * len(centroids)

        pairs = [
            (math.dist(track.centroid, centroid), track_id, index)
            for track_id, track in self._tracks.items()
            for index, centroid in enumerate(centroids)
        ]
        pairs.sort()

        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for distance, track_id, index in pairs:
            if distance > self.max_distance:
                break
            if track_id in used_tracks or index in used_detections:
                continue
            used_tracks.add(track_id)
            used_detections.add(index)
            assigned[index] = track_id
            track = self._tracks[track_id]
            track.centroid = centroids[index]
            track.last_seen = now
            track.frames += 1

        for index, person_id in enumerate(assigned):
            if person_id is not None:
                continue
            person_id = self._next_id
            self._next_id += 1
            self._tracks[person_id] = _Track(
                person_id=person_id,
                centroid=centroids[index],
                first_seen=now,
                last_seen=now,
            )
            assigned[index] = person_id
            used_tracks.add(person_id)
            logger.info("person %d entered (%d tracked)", person_id, len(self._tracks))

        self._age_out(used_tracks, now)
        return [person_id for person_id in assigned if person_id is not None]

    def _age_out(self, seen: set[int], now: float) -> None:
        for track_id in list(self._tracks):
            if track_id in seen:
                continue
            track = self._tracks[track_id]
            if now - track.last_seen <= self.max_unseen_s:
                continue
            del self._tracks[track_id]
            logger.info(
                "person %d left after %.1fs (%d frames)",
                track_id,
                now - track.first_seen,
                track.frames,
            )
            if self._on_lost is not None:
                self._on_lost(track_id)

    def reset(self) -> None:
        for track_id in list(self._tracks):
            if self._on_lost is not None:
                self._on_lost(track_id)
        self._tracks.clear()
        logger.debug("tracker reset")
