import logging

import numpy as np
import pytest

from fall_detection.biomechanics import (
    BASE_OF_SUPPORT_LANDMARKS,
    FloorEstimator,
    center_of_mass,
    postural_instability_index,
    torso_angle_from_vertical,
    validate_segment_table,
)

from conftest import make_world_landmark, standing_pose


def _supine_world_landmarks(scale_m: float = 1.7):
    """Lay the standing pose on its side: swap the vertical (y) and one
    horizontal (x) axis so the torso vector becomes horizontal (lying flat)."""
    _, world = standing_pose(scale_m)
    flattened = []
    for lm in world:
        flattened.append(make_world_landmark(x=lm.y, y=0.0, z=lm.z, visibility=lm.visibility))
    return flattened


def test_validate_segment_table_warns_but_does_not_raise(caplog):
    with caplog.at_level(logging.WARNING):
        total = validate_segment_table()
    assert total != pytest.approx(1.0, abs=0.001)
    assert any("De Leva" in record.message for record in caplog.records)


def test_center_of_mass_standing_lands_between_hip_and_shoulder():
    _, world = standing_pose(scale_m=1.7)
    com = center_of_mass(world)
    assert com is not None
    hip_y = 0.0
    shoulder_y = (world[11].y + world[12].y) / 2.0
    assert shoulder_y < com[1] < hip_y


def test_torso_angle_standing_is_near_zero():
    _, world = standing_pose()
    angle = torso_angle_from_vertical(world)
    assert angle == pytest.approx(0.0, abs=5.0)


def test_torso_angle_supine_is_near_ninety():
    world = _supine_world_landmarks()
    angle = torso_angle_from_vertical(world)
    assert angle == pytest.approx(90.0, abs=5.0)


def test_postural_instability_index_standing_is_stable():
    _, world = standing_pose()
    com = center_of_mass(world)
    com_ground_xy = np.array([com[0], com[2]])
    psi = postural_instability_index(com_ground_xy, world, BASE_OF_SUPPORT_LANDMARKS)
    assert psi is not None
    assert psi <= 0.0


def test_postural_instability_index_leaning_far_out_is_unstable():
    _, world = standing_pose()
    com_ground_xy = np.array([10.0, 10.0])  # nowhere near the feet
    psi = postural_instability_index(com_ground_xy, world, BASE_OF_SUPPORT_LANDMARKS)
    assert psi is not None
    assert psi > 0.0


def test_floor_estimator_locks_onto_standing_ankle_height_then_reports_zero_at_contact():
    estimator = FloorEstimator()
    standing_ankle_y = 0.83  # matches standing_pose's ankle fraction * scale_m
    for _ in range(10):
        estimator.update(ankle_world_y=standing_ankle_y, torso_angle_deg=2.0)
    height_while_standing_hip = estimator.height_above_floor(world_y=0.0)
    assert height_while_standing_hip == pytest.approx(standing_ankle_y, abs=0.05)

    height_at_floor_contact = estimator.height_above_floor(world_y=standing_ankle_y)
    assert height_at_floor_contact == pytest.approx(0.0, abs=0.05)


def test_floor_estimator_uncalibrated_reports_unknown_not_zero():
    estimator = FloorEstimator()
    assert estimator.height_above_floor(world_y=0.0) == float("inf")
