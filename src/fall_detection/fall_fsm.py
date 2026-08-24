"""Deterministic fall-detection FSM with 30-frame majority-vote confirmation.

States: UPRIGHT -> DESCENDING -> IMPACT -> SLUMPING -> POST_STABILITY_EVALUATION
-> FALL_CONFIRMED | BED_REST (terminal), or back to UPRIGHT if recovery is
detected before confirmation.

Design calls made when the source spec under-specified a transition (see the
approved implementation plan for full rationale):

* DESCENDING -> IMPACT fires on a high acceleration peak, OR a slow-slump
  vertical-displacement discriminator, OR a soft-landing energy-dissipation
  discriminator -- the spec describes those as standalone metrics without
  wiring them into the FSM; this OR-set is the concrete synthesis.
* DESCENDING -> UPRIGHT reverts after a 2.0s dwell timeout below v_trigger
  (not in the spec -- prevents the FSM getting stuck mid-descent forever).
* IMPACT -> SLUMPING is unconditional, one frame later; SLUMPING is where
  the 30-frame vote buffer actually accumulates, transitioning to
  POST_STABILITY_EVALUATION once full.
* FALL_CONFIRMED/BED_REST are sticky/terminal until an external reset() --
  an alert should not silently self-clear.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class FallState(IntEnum):
    UPRIGHT = 0
    DESCENDING = 1
    IMPACT = 2
    SLUMPING = 3
    POST_STABILITY_EVALUATION = 4
    FALL_CONFIRMED = 5
    BED_REST = 6


_TERMINAL_STATES = (FallState.FALL_CONFIRMED, FallState.BED_REST)


@dataclass(frozen=True)
class FallThresholds:
    v_trigger: float = 0.8
    a_impact: float = 21.58
    theta_threshold: float = 65.0
    n_buffer: int = 30
    r_vote: float = 0.70
    floor_clearance: float = 0.20
    aspect_ratio_threshold: float = 1.2
    ground_bound_window_s: float = 3.0
    ground_bound_height: float = 0.20
    slump_window_s: float = 2.0
    slump_displacement_m: float = 0.30
    dissipation_threshold_w: float = 150.0
    descending_dwell_timeout_s: float = 2.0
    static_prone_frames: int = 15


@dataclass(frozen=True)
class FallFeatures:
    t_seconds: float
    com: np.ndarray
    com_velocity: np.ndarray
    com_acceleration: np.ndarray
    com_height: float
    torso_angle_deg: float
    bbox_aspect_ratio: float
    instability_index: float | None


@dataclass(frozen=True)
class DiscriminatorFlags:
    kinetic_energy_ratio: float | None = None
    directional_correlation: float | None = None
    vertical_displacement_2s: float | None = None
    energy_dissipation_w: float | None = None
    ground_bound: bool = False
    bed_rest: bool = False


def frame_vote(features: FallFeatures, thresholds: FallThresholds = FallThresholds()) -> bool:
    """V(k): torso inclined AND bbox elongated AND close to the floor."""
    return (
        features.torso_angle_deg > thresholds.theta_threshold
        and features.bbox_aspect_ratio > thresholds.aspect_ratio_threshold
        and features.com_height <= thresholds.floor_clearance
    )


@dataclass
class PersonFallFSM:
    thresholds: FallThresholds = field(default_factory=FallThresholds)
    state: FallState = FallState.UPRIGHT
    _buffer: deque = field(default_factory=deque, repr=False)
    _descending_since: float | None = field(default=None, repr=False)
    _static_prone_streak: int = field(default=0, repr=False)

    def step(
        self, features: FallFeatures, discriminators: DiscriminatorFlags | None = None
    ) -> FallState:
        if self.state in _TERMINAL_STATES:
            return self.state

        discriminators = discriminators or DiscriminatorFlags()
        thresholds = self.thresholds
        downward_speed = float(features.com_velocity[1])
        accel_magnitude = float(np.linalg.norm(features.com_acceleration))

        is_prone_pose = (
            features.torso_angle_deg > thresholds.theta_threshold
            and features.bbox_aspect_ratio > thresholds.aspect_ratio_threshold
        )
        self._static_prone_streak = (
            self._static_prone_streak + 1 if is_prone_pose else max(0, self._static_prone_streak - 1)
        )

        if self.state == FallState.UPRIGHT:
            if downward_speed > thresholds.v_trigger:
                self.state = FallState.DESCENDING
                self._descending_since = features.t_seconds
            elif self._static_prone_streak >= thresholds.static_prone_frames:
                self.state = FallState.SLUMPING
                self._buffer.clear()

        elif self.state == FallState.DESCENDING:
            slump_path = (
                discriminators.vertical_displacement_2s is not None
                and discriminators.vertical_displacement_2s >= thresholds.slump_displacement_m
            )
            soft_landing_path = (
                discriminators.energy_dissipation_w is not None
                and discriminators.energy_dissipation_w >= thresholds.dissipation_threshold_w
            )
            if accel_magnitude >= thresholds.a_impact or slump_path or soft_landing_path:
                self.state = FallState.IMPACT
            elif downward_speed <= thresholds.v_trigger:
                dwell = features.t_seconds - (self._descending_since or features.t_seconds)
                if dwell >= thresholds.descending_dwell_timeout_s:
                    self.state = FallState.UPRIGHT
                    self._descending_since = None

        elif self.state == FallState.IMPACT:
            self.state = FallState.SLUMPING
            self._buffer.clear()

        elif self.state == FallState.SLUMPING:
            self._buffer.append(frame_vote(features, thresholds))
            if len(self._buffer) >= thresholds.n_buffer:
                self.state = FallState.POST_STABILITY_EVALUATION

        elif self.state == FallState.POST_STABILITY_EVALUATION:
            vote_fraction = sum(self._buffer) / len(self._buffer) if self._buffer else 0.0
            if vote_fraction >= thresholds.r_vote or discriminators.ground_bound:
                self.state = FallState.BED_REST if discriminators.bed_rest else FallState.FALL_CONFIRMED
            else:
                self.state = FallState.UPRIGHT
                self._buffer.clear()

        return self.state

    def reset(self) -> None:
        self.state = FallState.UPRIGHT
        self._buffer.clear()
        self._descending_since = None
        self._static_prone_streak = 0

    @property
    def vote_fraction(self) -> float:
        return sum(self._buffer) / len(self._buffer) if self._buffer else 0.0
