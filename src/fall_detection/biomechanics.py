"""De Leva anthropometric center-of-mass, torso angle, and base-of-support physics.

Duck-typed on landmark objects exposing ``.x``/``.y``/``.z`` (works for both
MediaPipe's ``NormalizedLandmark`` and ``Landmark``). Consumers should pass
``PersonPose.world_landmarks`` (metric, hip-centered) for physically meaningful
results -- see :mod:`fall_detection.pose`.

Axis convention (verified against a real MediaPipe inference, see
``tests/conftest.py``): y increases *downward*, same as the normalized image
landmarks. "Vertical" (up) is therefore the ``-y`` direction. The ground plane
is ``(x, z)``.

Mass fractions below are the four segments given verbatim in the project's
source spec, not the full 14-segment De Leva 1996 table -- they fold arm mass
into the trunk fraction and omit forearms/hands/feet as individual segments.
``validate_segment_table`` flags the resulting ~6% overshoot against 1.0 at
import-adjacent call time rather than silently trusting it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, Sequence, Union

import numpy as np

from .geometry import point_to_hull_signed_distance, convex_hull_2d
from .pose import PoseLandmark

logger = logging.getLogger(__name__)


class LandmarkLike(Protocol):
    x: float | None
    y: float | None
    z: float | None


Endpoint = Union[PoseLandmark, tuple[PoseLandmark, PoseLandmark]]


def _position(landmarks: Sequence[LandmarkLike], endpoint: Endpoint) -> np.ndarray | None:
    if isinstance(endpoint, tuple):
        a = _position(landmarks, endpoint[0])
        b = _position(landmarks, endpoint[1])
        if a is None or b is None:
            return None
        return (a + b) / 2.0
    landmark = landmarks[endpoint]
    if landmark.x is None or landmark.y is None:
        return None
    z = landmark.z if landmark.z is not None else 0.0
    return np.array([landmark.x, landmark.y, z], dtype=float)


@dataclass(frozen=True)
class DeLevaSegment:
    name: str
    proximal: Endpoint
    distal: Endpoint
    mass_fraction: float
    com_ratio: float


DE_LEVA_SEGMENTS: tuple[DeLevaSegment, ...] = (
    DeLevaSegment(
        "trunk",
        (PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER),
        (PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP),
        0.6211,
        0.4310,
    ),
    DeLevaSegment(
        "head_neck",
        PoseLandmark.NOSE,
        (PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER),
        0.0681,
        0.5000,
    ),
    DeLevaSegment("left_thigh", PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE, 0.1416, 0.4095),
    DeLevaSegment("right_thigh", PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE, 0.1416, 0.4095),
    DeLevaSegment("left_shank", PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE, 0.0433, 0.4395),
    DeLevaSegment("right_shank", PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE, 0.0433, 0.4395),
)


def validate_segment_table(
    segments: Sequence[DeLevaSegment] = DE_LEVA_SEGMENTS, tolerance: float = 0.02
) -> float:
    """Return the total mass fraction, warning (not raising) if it isn't ~1.0."""
    total = sum(segment.mass_fraction for segment in segments)
    if abs(total - 1.0) > tolerance:
        logger.warning(
            "De Leva segment table sums to %.4f, not 1.0 (tolerance %.4f) -- "
            "the spec's 4-segment table folds arm mass into the trunk fraction",
            total,
            tolerance,
        )
    return total


def segment_com(landmarks: Sequence[LandmarkLike], segment: DeLevaSegment) -> np.ndarray | None:
    proximal = _position(landmarks, segment.proximal)
    distal = _position(landmarks, segment.distal)
    if proximal is None or distal is None:
        return None
    return proximal + segment.com_ratio * (distal - proximal)


def center_of_mass(
    landmarks: Sequence[LandmarkLike], segments: Sequence[DeLevaSegment] = DE_LEVA_SEGMENTS
) -> np.ndarray | None:
    """Weighted sum of segment CoMs, renormalized over segments actually present.

    Renormalizing (rather than dividing by a fixed 1.0) keeps the estimate
    sane when a segment is occluded, at the cost of departing slightly from
    the spec's literal un-normalized sum.
    """
    total = np.zeros(3)
    total_weight = 0.0
    for segment in segments:
        com = segment_com(landmarks, segment)
        if com is None:
            continue
        total += segment.mass_fraction * com
        total_weight += segment.mass_fraction
    if total_weight == 0.0:
        return None
    return total / total_weight


def torso_angle_from_vertical(landmarks: Sequence[LandmarkLike]) -> float | None:
    """Angle in degrees between the hip->shoulder vector and straight up (-y)."""
    hip = _position(landmarks, (PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP))
    shoulder = _position(landmarks, (PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER))
    if hip is None or shoulder is None:
        return None
    torso = shoulder - hip
    norm = np.linalg.norm(torso)
    if norm == 0.0:
        return None
    up = np.array([0.0, -1.0, 0.0])
    cosine = float(np.dot(torso, up) / norm)
    cosine = min(max(cosine, -1.0), 1.0)
    return float(np.degrees(np.arccos(cosine)))


BASE_OF_SUPPORT_LANDMARKS: tuple[PoseLandmark, ...] = (
    PoseLandmark.LEFT_ANKLE,
    PoseLandmark.RIGHT_ANKLE,
    PoseLandmark.LEFT_HEEL,
    PoseLandmark.RIGHT_HEEL,
    PoseLandmark.LEFT_FOOT_INDEX,
    PoseLandmark.RIGHT_FOOT_INDEX,
)


def postural_instability_index(
    com_ground_xy: np.ndarray,
    landmarks: Sequence[LandmarkLike],
    bos_landmarks: Sequence[PoseLandmark] = BASE_OF_SUPPORT_LANDMARKS,
) -> float | None:
    """Psi: signed distance from the ground-projected CoM to the base of support.

    Positive = outside the support polygon (unstable), negative = inside.
    """
    points = []
    for landmark_index in bos_landmarks:
        position = _position(landmarks, landmark_index)
        if position is not None:
            points.append([position[0], position[2]])
    if len(points) < 2:
        return None
    hull = convex_hull_2d(np.array(points))
    return point_to_hull_signed_distance(np.asarray(com_ground_xy, dtype=float), hull)


class FloorEstimator:
    """Per-person running floor reference from world-landmark ankle height.

    No scene calibration exists in this MVP, so Z_CoM/floor-clearance
    thresholds need *some* floor reference to be computable at all: this
    tracks an exponential moving average of ankle height while the person is
    upright (torso within ``upright_gate_deg`` of vertical), which is the
    closest thing to "known floor contact" available without calibration.
    """

    def __init__(self, decay: float = 0.98) -> None:
        self.decay = decay
        self._floor_y: float | None = None
        self._max_ankle_y: float | None = None

    def update(self, ankle_world_y: float, torso_angle_deg: float, upright_gate_deg: float = 30.0) -> float | None:
        self._max_ankle_y = max(self._max_ankle_y, ankle_world_y) if self._max_ankle_y is not None else ankle_world_y
        if torso_angle_deg <= upright_gate_deg:
            if self._floor_y is None:
                self._floor_y = ankle_world_y
            else:
                self._floor_y = self.decay * self._floor_y + (1.0 - self.decay) * ankle_world_y
        return self._floor_y

    def height_above_floor(self, world_y: float) -> float:
        """Distance above the floor. Falls back to the deepest observed ankle
        position when the subject has never been upright (so a person prone
        from frame 1 still gets a usable estimate). Returns +inf only when no
        frame has been observed at all."""
        effective_floor = self._floor_y if self._floor_y is not None else self._max_ankle_y
        if effective_floor is None:
            return float("inf")
        return effective_floor - world_y
