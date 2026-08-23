import numpy as np
import pytest

from fall_detection.discriminators import (
    directional_correlation,
    energy_dissipation_rate,
    ground_bound,
    kinetic_energy_ratio,
    sliding_vertical_displacement,
)


def test_flop_whole_body_fall_has_high_directional_correlation():
    gamma = directional_correlation(wrist_velocity=[0.0, 5.0, 0.0], hip_velocity=[0.0, 5.0, 0.0])
    assert gamma == pytest.approx(1.0, abs=1e-3)


def test_tossing_limb_motion_with_still_trunk_has_high_energy_ratio():
    zeta = kinetic_energy_ratio(
        limb_velocities=[np.array([3.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0])],
        limb_masses_kg=[0.5, 0.5],
        trunk_velocity=np.array([0.01, 0.0, 0.0]),
        trunk_mass_kg=40.0,
    )
    assert zeta > 100.0


def test_arm_drop_isolated_limb_has_low_directional_correlation():
    gamma = directional_correlation(wrist_velocity=[0.0, 5.0, 0.0], hip_velocity=[0.0, 0.0, 0.0])
    assert gamma < 0.3


def test_slump_sliding_displacement_matches_net_height_drop_over_window():
    com_heights = [(0.0, 1.5), (0.5, 1.3), (1.0, 1.1), (1.5, 0.9), (2.0, 0.7)]
    displacement = sliding_vertical_displacement(com_heights, window_s=2.0)
    assert displacement == pytest.approx(0.8, abs=0.01)


def test_slump_sliding_displacement_ignores_samples_outside_window():
    com_heights = [(0.0, 5.0), (10.0, 1.5), (10.5, 1.3), (11.0, 1.1)]
    displacement = sliding_vertical_displacement(com_heights, window_s=2.0)
    assert displacement == pytest.approx(0.4, abs=0.01)


def test_soft_landing_dissipation_rate_is_large_despite_damped_impact():
    com_velocities = [(0.0, np.array([0.0, 3.0, 0.0])), (0.05, np.array([0.0, 0.2, 0.0]))]
    rate = energy_dissipation_rate(com_velocities, mass_kg=70.0)
    assert rate > 150.0


def test_ground_bound_true_when_mean_height_under_threshold_despite_a_bump():
    com_heights = [(t, 0.15) for t in [0.0, 0.5, 1.0, 2.0, 2.5, 3.0]]
    com_heights.insert(3, (1.5, 0.35))  # a "trying to get up" bump
    assert ground_bound(com_heights, window_s=3.0, height_threshold=0.20) is True


def test_ground_bound_false_when_standing():
    com_heights = [(t, 1.5) for t in [0.0, 1.0, 2.0, 3.0]]
    assert ground_bound(com_heights, window_s=3.0, height_threshold=0.20) is False
