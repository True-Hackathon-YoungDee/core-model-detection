"""Edge-case discriminator functions.

Each is a small pure function over physics-state snapshots, not a subsystem
of its own -- they exist to tell known-tricky Activities of Daily Living
(bed flop, sleep tossing, isolated arm drop, slow slump, soft landing) apart
from a real fall. Feed their outputs into :class:`fall_detection.fall_fsm.DiscriminatorFlags`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def kinetic_energy_ratio(
    limb_velocities: Sequence[np.ndarray],
    limb_masses_kg: Sequence[float],
    trunk_velocity: np.ndarray,
    trunk_mass_kg: float,
    eps: float = 1e-6,
) -> float:
    """zeta: combined limb kinetic energy over trunk kinetic energy.

    Large during sleep-tossing (limbs move, trunk stays still); near 1 or
    below during a whole-body fall (trunk moves too).
    """
    limb_ke = sum(
        0.5 * mass * float(np.dot(velocity, velocity))
        for velocity, mass in zip(limb_velocities, limb_masses_kg)
    )
    trunk_velocity = np.asarray(trunk_velocity, dtype=float)
    trunk_ke = 0.5 * trunk_mass_kg * float(np.dot(trunk_velocity, trunk_velocity))
    return limb_ke / (trunk_ke + eps)


def directional_correlation(
    wrist_velocity: np.ndarray, hip_velocity: np.ndarray, eps: float = 1e-6
) -> float:
    """gamma: cosine similarity between a wrist's velocity and the hip's.

    Near 1 for a whole-body fall (limbs and torso descend together); near 0
    for an isolated limb movement (torso stays near-stationary, so its
    velocity norm collapses toward zero and the ratio is driven to 0 by the
    eps-guarded denominator rather than blowing up).
    """
    wrist_velocity = np.asarray(wrist_velocity, dtype=float)
    hip_velocity = np.asarray(hip_velocity, dtype=float)
    denominator = np.linalg.norm(wrist_velocity) * np.linalg.norm(hip_velocity) + eps
    return float(np.dot(wrist_velocity, hip_velocity) / denominator)


def sliding_vertical_displacement(
    com_heights: Sequence[tuple[float, float]], window_s: float = 2.0
) -> float:
    """Net height drop (earliest minus latest sample) within the trailing window.

    Catches slow slumping falls that never produce a sharp velocity spike.
    """
    if not com_heights:
        return 0.0
    t_now = com_heights[-1][0]
    window = [(t, h) for t, h in com_heights if t_now - t <= window_s]
    if len(window) < 2:
        return 0.0
    return window[0][1] - window[-1][1]


def energy_dissipation_rate(
    com_velocities: Sequence[tuple[float, np.ndarray]], mass_kg: float
) -> float:
    """dKe/dt magnitude, positive while dissipating (KE decreasing).

    Stays large on a soft landing even though peak acceleration is damped,
    since the total kinetic-energy change is the same as a hard-floor fall.
    """
    if len(com_velocities) < 2:
        return 0.0
    (t0, v0), (t1, v1) = com_velocities[-2], com_velocities[-1]
    dt = t1 - t0
    if dt <= 0:
        return 0.0
    ke0 = 0.5 * mass_kg * float(np.dot(v0, v0))
    ke1 = 0.5 * mass_kg * float(np.dot(v1, v1))
    return (ke0 - ke1) / dt


def ground_bound(
    com_heights: Sequence[tuple[float, float]],
    window_s: float = 3.0,
    height_threshold: float = 0.20,
) -> bool:
    """Mean CoM height over the trailing window stays at floor level.

    Using the mean (not the instantaneous height) is what lets a person's
    post-fall micro-movements -- trying to push themselves up, shivering --
    stay classified as distress rather than flipping the system back to
    "recovered."
    """
    if not com_heights:
        return False
    t_now = com_heights[-1][0]
    window = [height for t, height in com_heights if t_now - t <= window_s]
    if not window:
        return False
    return (sum(window) / len(window)) <= height_threshold


def bed_rest(
    com_ground_xy: tuple[float, float] | None,
    com_height: float,
    bed_roi: object | None = None,
) -> bool:
    """Whether a descent terminated on a bed/furniture ROI, not the floor.

    Stubbed to always return False: no scene calibration exists yet (see
    the deferred homography/bed-ROI phase). ``bed_roi`` is typed loosely so
    the eventual ``calibration.BedROI`` slots in without an API change.
    """
    return False
