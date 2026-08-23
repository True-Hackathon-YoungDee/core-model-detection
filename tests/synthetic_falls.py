"""Synthetic keyframed PersonPose sequences for orchestrator integration tests.

A "fast fall" sequence: standing still, then a sharp transition to lying flat
on the floor (only a handful of frames -- fast enough to produce a genuine
deceleration spike and to avoid the transitional postures sitting in the
30-frame vote buffer, which would otherwise depress the vote fraction below
the 70% confirmation threshold), then lying still for long enough to fill and
evaluate the buffer.
"""

from __future__ import annotations

from mediapipe.tasks.python.components.containers.landmark import Landmark, NormalizedLandmark

from fall_detection.pose import PersonPose, PoseLandmark, person_from_landmarks

from conftest import NUM_LANDMARKS, standing_pose


def _lying_world_points(scale_m: float, floor_y: float) -> list[tuple[float, float, float]]:
    """Lying flat: standing's vertical extent becomes horizontal spread, and
    every point settles near floor height (same axis-swap trick used for the
    supine fixture in test_biomechanics.py, plus a floor-height offset)."""
    _, standing_world = standing_pose(scale_m)
    points = []
    for lm in standing_world:
        x = lm.y
        y = floor_y + 0.03 * (lm.x / scale_m)
        z = lm.x
        points.append((x, y, z))
    return points


def _project_normalized(x: float, y: float) -> tuple[float, float]:
    nx = min(max(0.5 + x * 0.3, 0.0), 1.0)
    ny = min(max(0.5 + y * 0.3, 0.0), 1.0)
    return nx, ny


def _lerp(a: float, b: float, fraction: float) -> float:
    return a + (b - a) * fraction


def fall_sequence(
    fps: float = 30.0,
    standing_s: float = 1.0,
    transition_frames: int = 5,
    lying_s: float = 2.0,
    scale_m: float = 1.7,
    person_id: int = 1,
) -> list[tuple[float, PersonPose]]:
    _, standing_world = standing_pose(scale_m)
    floor_y = (
        standing_world[PoseLandmark.LEFT_ANKLE].y + standing_world[PoseLandmark.RIGHT_ANKLE].y
    ) / 2.0
    lying_world = _lying_world_points(scale_m, floor_y)

    dt = 1.0 / fps
    frames: list[tuple[float, PersonPose]] = []
    t = 0.0

    def _emit(fraction: float) -> None:
        nonlocal t
        world_landmarks: list[Landmark] = []
        landmarks: list[NormalizedLandmark] = []
        for index in range(NUM_LANDMARKS):
            sx, sy, sz = standing_world[index].x, standing_world[index].y, standing_world[index].z
            lx, ly, lz = lying_world[index]
            x = _lerp(sx, lx, fraction)
            y = _lerp(sy, ly, fraction)
            z = _lerp(sz, lz, fraction)
            world_landmarks.append(Landmark(x=x, y=y, z=z, visibility=1.0, presence=1.0))
            nx, ny = _project_normalized(x, y)
            landmarks.append(NormalizedLandmark(x=nx, y=ny, z=0.0, visibility=1.0, presence=1.0))
        person = person_from_landmarks(landmarks, world_landmarks, person_id)
        frames.append((t, person))
        t += dt

    for _ in range(max(1, int(standing_s / dt))):
        _emit(0.0)
    for i in range(1, transition_frames + 1):
        _emit(i / transition_frames)
    for _ in range(max(1, int(lying_s / dt))):
        _emit(1.0)

    return frames
