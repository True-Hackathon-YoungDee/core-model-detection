import numpy as np
import pytest

from fall_detection.kalman import LandmarkKalman, LandmarkKalmanStabilizer

from conftest import make_world_landmark


def test_constant_velocity_trajectory_converges_to_true_velocity():
    dt = 1.0 / 30.0
    velocity = np.array([1.0, 0.0, 0.0])
    kf = LandmarkKalman(t0=0.0, x0=np.zeros(3), process_noise=1.0, measurement_noise=0.001)
    t = 0.0
    kinematics = None
    for frame in range(1, 61):
        t = frame * dt
        position = velocity * t
        kinematics = kf.update(t, position, visibility=1.0)
    assert kinematics.velocity == pytest.approx(velocity, abs=0.2)


def test_occlusion_extrapolates_along_last_known_velocity():
    dt = 1.0 / 30.0
    velocity = np.array([1.0, 0.0, 0.0])
    kf = LandmarkKalman(t0=0.0, x0=np.zeros(3), process_noise=1.0, measurement_noise=0.001)
    t = 0.0
    for frame in range(1, 31):
        t = frame * dt
        kf.update(t, velocity * t, visibility=1.0)

    kinematics = None
    for frame in range(31, 41):
        t = frame * dt
        stale_measurement = velocity * (30 * dt)  # a stuck detector reports a frozen position
        kinematics = kf.update(t, stale_measurement, visibility=0.1)

    true_position = velocity * t
    assert kinematics.position == pytest.approx(true_position, abs=0.15)


def test_stabilizer_forget_resets_that_persons_filters():
    stabilizer = LandmarkKalmanStabilizer()
    moving = [make_world_landmark(x=0.0, y=0.0, z=0.0, visibility=1.0)]
    stabilizer.stabilize(person_id=1, world_landmarks=moving, t_seconds=0.0)
    moving2 = [make_world_landmark(x=1.0, y=0.0, z=0.0, visibility=1.0)]
    result = stabilizer.stabilize(person_id=1, world_landmarks=moving2, t_seconds=1.0)
    assert result[0].velocity[0] > 0.0

    stabilizer.forget(person_id=1)
    fresh = stabilizer.stabilize(person_id=1, world_landmarks=moving2, t_seconds=2.0)
    assert fresh[0].velocity[0] == pytest.approx(0.0, abs=1e-6)


def test_prediction_only_output_becomes_unavailable_after_configured_gap():
    stabilizer = LandmarkKalmanStabilizer(max_prediction_gap_s=0.5)
    visible = [make_world_landmark(x=0.0, y=0.0, z=0.0, visibility=1.0)]
    invisible = [make_world_landmark(x=0.0, y=0.0, z=0.0, visibility=0.1)]

    observed = stabilizer.stabilize(1, visible, 0.0)
    predicted = stabilizer.stabilize(1, invisible, 0.4)
    stale = stabilizer.stabilize(1, invisible, 0.6)

    assert len(observed) == len(predicted) == len(stale) == 1
    assert observed[0].available and observed[0].observed
    assert predicted[0].available and not predicted[0].observed
    assert predicted[0].prediction_age_s == pytest.approx(0.4)
    assert not stale[0].available and not stale[0].observed
    assert stale[0].prediction_age_s == pytest.approx(0.6)
