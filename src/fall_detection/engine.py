"""One interface over the two ways of turning a frame into people.

``NativeEngine`` gives the whole frame to a single landmarker -- fast, and the
right answer when there is one subject. ``CascadeEngine`` detects person boxes
first and landmarks each body separately -- roughly twice the cost, and the only
one of the two that reliably sees everybody in a crowded frame.

``Strategy.AUTO`` picks between them from ``num_poses``.
"""

from __future__ import annotations

import logging
from typing import Protocol, Sequence

import numpy as np

from .cascade import CascadePoseEstimator
from .pose import (
    PersonPose,
    PoseConfig,
    PoseDetector,
    RunningMode,
    Strategy,
    build_persons,
)

logger = logging.getLogger(__name__)


class PoseEngine(Protocol):
    """Synchronous frame -> people."""

    label: str

    def infer(self, bgr_frame: np.ndarray, timestamp_ms: int) -> list[PersonPose]: ...

    def close(self) -> None: ...


def resolve_strategy(config: PoseConfig) -> Strategy:
    """Turn ``Strategy.AUTO`` into an actual choice, and say why."""
    if config.strategy is not Strategy.AUTO:
        logger.info("strategy=%s (requested)", config.strategy)
        return config.strategy

    if config.num_poses > 1:
        logger.info(
            "strategy=cascade (auto: --num-poses=%d; one landmarker over the whole "
            "frame reports a single person in a crowd)",
            config.num_poses,
        )
        return Strategy.CASCADE

    logger.info("strategy=native (auto: --num-poses=1, no need to pay for person boxes)")
    return Strategy.NATIVE


class NativeEngine:
    """Full-frame landmarking in IMAGE or VIDEO mode."""

    label = "native"

    def __init__(self, config: PoseConfig, running_mode: RunningMode = RunningMode.VIDEO) -> None:
        self.detector = PoseDetector(config, running_mode)
        self.running_mode = running_mode

    def infer(self, bgr_frame: np.ndarray, timestamp_ms: int) -> list[PersonPose]:
        if self.running_mode is RunningMode.VIDEO:
            result = self.detector.detect_video(bgr_frame, timestamp_ms)
        else:
            result = self.detector.detect_image(bgr_frame)
        return build_persons(result)

    def close(self) -> None:
        self.detector.close()


class CascadeEngine:
    """Person boxes, then one landmarker per body.

    Holds the previous frame's people so the estimator can carry regions of
    interest forward when ``detect_interval > 1``.
    """

    label = "cascade"

    def __init__(self, config: PoseConfig) -> None:
        self.estimator = CascadePoseEstimator(config)
        self._previous: Sequence[PersonPose] = ()

    def infer(self, bgr_frame: np.ndarray, timestamp_ms: int) -> list[PersonPose]:
        persons = self.estimator.infer(bgr_frame, timestamp_ms, self._previous)
        self._previous = persons
        return persons

    def reset(self) -> None:
        self._previous = ()

    def close(self) -> None:
        self.estimator.close()


def build_engine(
    config: PoseConfig,
    running_mode: RunningMode = RunningMode.VIDEO,
    strategy: Strategy | None = None,
) -> PoseEngine:
    """Build the engine the config asks for."""
    strategy = strategy or resolve_strategy(config)
    if strategy is Strategy.CASCADE:
        return CascadeEngine(config)
    return NativeEngine(config, running_mode)
