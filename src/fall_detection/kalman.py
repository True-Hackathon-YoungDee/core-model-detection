"""Extended Kalman Filter landmark stabilizer.

Each landmark is a 9D constant-acceleration state
``[px, py, pz, vx, vy, vz, ax, ay, az]``. When a landmark's visibility drops
below ``gate`` (MediaPipe reports it occluded), the measurement-update step
is skipped and only the kinematic prediction propagates -- this is what lets
a landmark survive a limb going behind a blanket or furniture without the
whole trajectory freezing or jumping.

Consumes the already tracked+smoothed output of
:class:`fall_detection.runner.PosePipeline` -- keyed the same way as
:class:`fall_detection.smoothing.LandmarkSmoother`, by ``(person_id,
landmark_index)``, with the same ``forget``/``reset`` lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

_H = np.hstack([np.eye(3), np.zeros((3, 6))])


@dataclass(frozen=True)
class LandmarkKinematics:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


def _f_matrix(dt: float) -> np.ndarray:
    f_meta = np.array([[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
    return np.kron(f_meta, np.eye(3))


def _q_matrix(dt: float, process_noise: float) -> np.ndarray:
    if dt <= 0.0:
        return np.zeros((9, 9))
    q_meta = process_noise * np.array(
        [
            [dt**5 / 20.0, dt**4 / 8.0, dt**3 / 6.0],
            [dt**4 / 8.0, dt**3 / 3.0, dt**2 / 2.0],
            [dt**3 / 6.0, dt**2 / 2.0, dt],
        ]
    )
    return np.kron(q_meta, np.eye(3))


class LandmarkKalman:
    """9D constant-acceleration EKF for one landmark's 3D trajectory."""

    def __init__(
        self,
        t0: float,
        x0: np.ndarray,
        process_noise: float = 1.0,
        measurement_noise: float = 0.01,
    ) -> None:
        self.t = t0
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.state = np.zeros(9)
        self.state[0:3] = x0
        self.P = np.eye(9)

    def predict(self, t: float) -> np.ndarray:
        dt = t - self.t
        if dt <= 0.0:
            return self.state.copy()
        F = _f_matrix(dt)
        Q = _q_matrix(dt, self.process_noise)
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q
        self.t = t
        return self.state.copy()

    def update(
        self, t: float, measurement: np.ndarray, visibility: float, gate: float = 0.5
    ) -> LandmarkKinematics:
        self.predict(t)
        if visibility >= gate:
            R = self.measurement_noise * np.eye(3)
            residual = np.asarray(measurement, dtype=float) - _H @ self.state
            S = _H @ self.P @ _H.T + R
            K = self.P @ _H.T @ np.linalg.inv(S)
            self.state = self.state + K @ residual
            self.P = (np.eye(9) - K @ _H) @ self.P
        return LandmarkKinematics(
            position=self.state[0:3].copy(),
            velocity=self.state[3:6].copy(),
            acceleration=self.state[6:9].copy(),
        )


class LandmarkKalmanStabilizer:
    """Owns one :class:`LandmarkKalman` per ``(person_id, landmark_index)``."""

    def __init__(self, process_noise: float = 1.0, measurement_noise: float = 0.01) -> None:
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self._filters: dict[tuple[int, int], LandmarkKalman] = {}

    def stabilize(
        self, person_id: int, world_landmarks: Iterable[Any], t_seconds: float
    ) -> list[LandmarkKinematics]:
        results: list[LandmarkKinematics] = []
        for index, landmark in enumerate(world_landmarks):
            key = (person_id, index)
            x = landmark.x if landmark.x is not None else 0.0
            y = landmark.y if landmark.y is not None else 0.0
            z = landmark.z if landmark.z is not None else 0.0
            position = np.array([x, y, z])
            visibility = landmark.visibility if landmark.visibility is not None else 0.0

            existing = self._filters.get(key)
            if existing is None:
                self._filters[key] = LandmarkKalman(
                    t_seconds, position, self.process_noise, self.measurement_noise
                )
                results.append(
                    LandmarkKinematics(
                        position=position.copy(),
                        velocity=np.zeros(3),
                        acceleration=np.zeros(3),
                    )
                )
            else:
                results.append(existing.update(t_seconds, position, visibility))
        return results

    def forget(self, person_id: int) -> None:
        for key in [key for key in self._filters if key[0] == person_id]:
            del self._filters[key]

    def reset(self) -> None:
        self._filters.clear()
