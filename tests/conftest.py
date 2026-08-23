"""Synthetic PersonPose builders shared across the fall-detection test suite.

World-landmark axis convention (verified against a real MediaPipe inference on
storage.googleapis.com/mediapipe-assets/pose.jpg): x is lateral, y increases
*downward* (same as the normalized image landmarks -- not flipped), z is depth.
Origin is the mid-hip point. All fixtures below follow that convention.
"""

from __future__ import annotations

import dataclasses

from mediapipe.tasks.python.components.containers.landmark import Landmark, NormalizedLandmark

from fall_detection.pose import PersonPose, PoseLandmark, person_from_landmarks

NUM_LANDMARKS = 33

# Coarse anatomical proportions as fractions of standing height, hip-centered,
# y positive = downward. Precise enough for physics-function unit tests; no
# test relies on facial-landmark accuracy.
_REGION_Y_FRACTION: dict[range, float] = {
    range(0, 11): -0.34,  # face (nose .. mouth)
    range(11, 13): -0.29,  # shoulders
    range(13, 15): -0.10,  # elbows
    range(15, 23): 0.05,  # wrists + fingers
    range(23, 25): 0.0,  # hips
    range(25, 27): 0.245,  # knees
    range(27, 33): 0.49,  # ankles, heels, foot indices
}
_REGION_X_FRACTION: dict[range, float] = {
    range(0, 11): 0.04,
    range(11, 13): 0.115,
    range(13, 15): 0.16,
    range(15, 23): 0.18,
    range(23, 25): 0.095,
    range(25, 27): 0.09,
    range(27, 33): 0.09,
}


def _region_value(index: int, table: dict[range, float]) -> float:
    for region, value in table.items():
        if index in region:
            return value
    raise KeyError(f"landmark index {index} not covered by region table")


def _side_sign(index: int) -> float:
    name = PoseLandmark(index).name
    if name.startswith("LEFT_"):
        return 1.0
    if name.startswith("RIGHT_"):
        return -1.0
    return 0.0


# Front/back (z) offsets for the foot-region landmarks, so a standing person's
# base-of-support polygon has real area instead of collapsing onto a line.
_Z_OVERRIDE: dict[int, float] = {
    PoseLandmark.LEFT_HEEL: -0.05,
    PoseLandmark.RIGHT_HEEL: -0.05,
    PoseLandmark.LEFT_FOOT_INDEX: 0.10,
    PoseLandmark.RIGHT_FOOT_INDEX: 0.10,
}


def make_landmark(
    x: float, y: float, z: float = 0.0, visibility: float = 1.0, presence: float = 1.0
) -> NormalizedLandmark:
    return NormalizedLandmark(x=x, y=y, z=z, visibility=visibility, presence=presence)


def make_world_landmark(
    x: float, y: float, z: float = 0.0, visibility: float = 1.0, presence: float = 1.0
) -> Landmark:
    return Landmark(x=x, y=y, z=z, visibility=visibility, presence=presence)


def standing_pose(scale_m: float = 1.7) -> tuple[list[NormalizedLandmark], list[Landmark]]:
    """A canonical upright person: 33 normalized + 33 world landmarks.

    World landmarks are hip-centered meters; normalized landmarks are a rough
    image-space projection of the same pose, just for shape/visibility
    plausibility (bbox, centroid, score derivation).
    """
    landmarks: list[NormalizedLandmark] = []
    world_landmarks: list[Landmark] = []
    for index in range(NUM_LANDMARKS):
        wx = _side_sign(index) * _region_value(index, _REGION_X_FRACTION) * scale_m
        wy = _region_value(index, _REGION_Y_FRACTION) * scale_m
        wz = _Z_OVERRIDE.get(index, 0.0) * scale_m
        world_landmarks.append(make_world_landmark(wx, wy, wz))

        nx = min(max(0.5 + wx * 0.3, 0.0), 1.0)
        ny = min(max(0.5 + wy * 0.3, 0.0), 1.0)
        landmarks.append(make_landmark(nx, ny, 0.0))
    return landmarks, world_landmarks


def make_person(
    person_id: int = 0,
    landmarks: list[NormalizedLandmark] | None = None,
    world_landmarks: list[Landmark] | None = None,
    **overrides,
) -> PersonPose:
    """Build a full PersonPose, defaulting to a standing pose.

    Extra keyword args are applied via dataclasses.replace after construction,
    for tests that just need a PersonPose in a particular state (e.g. score).
    """
    if landmarks is None or world_landmarks is None:
        default_landmarks, default_world = standing_pose()
        landmarks = default_landmarks if landmarks is None else landmarks
        world_landmarks = default_world if world_landmarks is None else world_landmarks
    person = person_from_landmarks(landmarks, world_landmarks, person_id)
    if overrides:
        person = dataclasses.replace(person, **overrides)
    return person
