"""Per-person orchestrator: PersonPose -> EKF stabilization -> physics ->
discriminators -> FSM, one call per frame, keyed by person_id like
:class:`fall_detection.runner.PosePipeline`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .biomechanics import (
    BASE_OF_SUPPORT_LANDMARKS,
    FloorEstimator,
    center_of_mass,
    postural_instability_index,
    torso_angle_from_vertical,
)
from .discriminators import (
    bed_rest,
    directional_correlation,
    energy_dissipation_rate,
    ground_bound,
    kinetic_energy_ratio,
    sliding_vertical_displacement,
)
from .fall_fsm import DiscriminatorFlags, FallFeatures, FallState, FallThresholds, PersonFallFSM
from .kalman import LandmarkKalmanStabilizer
from .pose import PersonPose, PoseLandmark

# Wrists/ankles aren't individual segments in the project's 4-segment De Leva
# table (see biomechanics.py); this is a rough per-point extremity mass used
# only for the kinetic-energy-ratio discriminator, not the CoM calculation.
_LIMB_POINT_MASS_FRACTION = 0.02
_TRUNK_MASS_FRACTION = 0.6211  # matches DE_LEVA_SEGMENTS[0] ("trunk")


@dataclass(frozen=True)
class FallEvent:
    person_id: int
    state: FallState
    state_changed: bool
    t_seconds: float
    torso_angle_deg: float
    com_height: float
    instability_index: float | None
    vote_fraction: float


@dataclass
class _StabilizedPoint:
    x: float
    y: float
    z: float


def _derivative(history: Sequence[tuple[float, np.ndarray]]) -> np.ndarray:
    if len(history) < 2:
        dims = len(history[-1][1]) if history else 3
        return np.zeros(dims)
    (t0, v0), (t1, v1) = history[-2], history[-1]
    dt = t1 - t0
    if dt <= 0:
        return np.zeros_like(v1)
    return (v1 - v0) / dt


class PersonFallTracker:
    """Owns the physics state for a single tracked person."""

    def __init__(
        self, person_id: int, thresholds: FallThresholds | None = None, body_mass_kg: float = 70.0
    ) -> None:
        self.person_id = person_id
        self.thresholds = thresholds or FallThresholds()
        self.body_mass_kg = body_mass_kg
        self._kalman = LandmarkKalmanStabilizer()
        self._floor = FloorEstimator()
        self._fsm = PersonFallFSM(thresholds=self.thresholds)
        history_len = 300
        self._com_history: deque[tuple[float, np.ndarray]] = deque(maxlen=history_len)
        self._height_history: deque[tuple[float, float]] = deque(maxlen=history_len)
        self._velocity_history: deque[tuple[float, np.ndarray]] = deque(maxlen=history_len)

    def update(
        self,
        person: PersonPose,
        t_seconds: float,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> FallEvent:
        kinematics = self._kalman.stabilize(self.person_id, person.world_landmarks, t_seconds)
        stabilized = [_StabilizedPoint(x=k.position[0], y=k.position[1], z=k.position[2]) for k in kinematics]

        com = center_of_mass(stabilized)
        if com is None:
            com = np.zeros(3)
        torso_angle = torso_angle_from_vertical(stabilized)
        torso_angle = torso_angle if torso_angle is not None else 0.0

        ankle_y = (stabilized[PoseLandmark.LEFT_ANKLE].y + stabilized[PoseLandmark.RIGHT_ANKLE].y) / 2.0
        self._floor.update(ankle_world_y=ankle_y, torso_angle_deg=torso_angle)
        com_height = self._floor.height_above_floor(com[1])

        self._com_history.append((t_seconds, com))
        com_velocity = _derivative(self._com_history)
        self._velocity_history.append((t_seconds, com_velocity))
        com_acceleration = _derivative(self._velocity_history)

        self._height_history.append((t_seconds, com_height))

        psi = postural_instability_index(
            np.array([com[0], com[2]]), stabilized, BASE_OF_SUPPORT_LANDMARKS
        )

        if frame_width and frame_height:
            x1, y1, x2, y2 = person.bbox_in_pixels(frame_width, frame_height)
        else:
            x1, y1, x2, y2 = person.bbox
        aspect_ratio = (x2 - x1) / max(y2 - y1, 1e-6)

        vertical_displacement = sliding_vertical_displacement(
            list(self._height_history), self.thresholds.slump_window_s
        )
        dissipation = energy_dissipation_rate(list(self._velocity_history), self.body_mass_kg)
        is_ground_bound = ground_bound(
            list(self._height_history), self.thresholds.ground_bound_window_s, self.thresholds.ground_bound_height
        )

        left_wrist, right_wrist = kinematics[PoseLandmark.LEFT_WRIST], kinematics[PoseLandmark.RIGHT_WRIST]
        wrist = (
            left_wrist
            if np.linalg.norm(left_wrist.velocity) >= np.linalg.norm(right_wrist.velocity)
            else right_wrist
        )
        left_hip, right_hip = kinematics[PoseLandmark.LEFT_HIP], kinematics[PoseLandmark.RIGHT_HIP]
        hip_velocity = (left_hip.velocity + right_hip.velocity) / 2.0
        gamma = directional_correlation(wrist.velocity, hip_velocity)

        left_ankle, right_ankle = kinematics[PoseLandmark.LEFT_ANKLE], kinematics[PoseLandmark.RIGHT_ANKLE]
        zeta = kinetic_energy_ratio(
            limb_velocities=[left_wrist.velocity, right_wrist.velocity, left_ankle.velocity, right_ankle.velocity],
            limb_masses_kg=[self.body_mass_kg * _LIMB_POINT_MASS_FRACTION] * 4,
            trunk_velocity=com_velocity,
            trunk_mass_kg=self.body_mass_kg * _TRUNK_MASS_FRACTION,
        )

        features = FallFeatures(
            t_seconds=t_seconds,
            com=com,
            com_velocity=com_velocity,
            com_acceleration=com_acceleration,
            com_height=com_height,
            torso_angle_deg=torso_angle,
            bbox_aspect_ratio=aspect_ratio,
            instability_index=psi,
        )
        discriminators = DiscriminatorFlags(
            kinetic_energy_ratio=zeta,
            directional_correlation=gamma,
            vertical_displacement_2s=vertical_displacement,
            energy_dissipation_w=dissipation,
            ground_bound=is_ground_bound,
            bed_rest=bed_rest(None, com_height, None),
        )

        previous_state = self._fsm.state
        new_state = self._fsm.step(features, discriminators)

        return FallEvent(
            person_id=self.person_id,
            state=new_state,
            state_changed=new_state != previous_state,
            t_seconds=t_seconds,
            torso_angle_deg=torso_angle,
            com_height=com_height,
            instability_index=psi,
            vote_fraction=self._fsm.vote_fraction,
        )


class FallStateManager:
    """One call per frame, keyed by person_id -- same shape as PosePipeline."""

    def __init__(self, thresholds: FallThresholds | None = None, body_mass_kg: float = 70.0) -> None:
        self.thresholds = thresholds or FallThresholds()
        self.body_mass_kg = body_mass_kg
        self._trackers: dict[int, PersonFallTracker] = {}

    def update(
        self,
        persons: list[PersonPose],
        t_seconds: float,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> list[FallEvent]:
        events = []
        for person in persons:
            tracker = self._trackers.get(person.person_id)
            if tracker is None:
                tracker = PersonFallTracker(person.person_id, self.thresholds, self.body_mass_kg)
                self._trackers[person.person_id] = tracker
            events.append(tracker.update(person, t_seconds, frame_width, frame_height))
        return events

    def forget(self, person_id: int) -> None:
        self._trackers.pop(person_id, None)

    def reset(self) -> None:
        self._trackers.clear()
