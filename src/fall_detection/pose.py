"""Core pose detection built on the MediaPipe Tasks Pose Landmarker.

MediaPipe 1.x ships the Tasks API only -- the old ``mediapipe.solutions``
namespace no longer exists in the wheel -- so everything here goes through
``mediapipe.tasks.python.vision.PoseLandmarker``.

BlazePose runs a full-frame detector to find a person, then tracks that ROI on
following frames and only re-runs the detector when tracking confidence drops.
That is why VIDEO and LIVE_STREAM modes need strictly increasing timestamps: the
ROI carried between frames is keyed on them.

Person identity: the landmarker has no re-identification, so the order of
``result.pose_landmarks`` can swap between frames when subjects cross. Stable
ids come from :mod:`fall_detection.tracking`.
"""

from __future__ import annotations

import contextlib
import logging
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .logging_config import RateLimiter, suppress_native_stderr
from .models import DetectorVariant, ModelVariant, resolve_model
from .strategy import Strategy

logger = logging.getLogger(__name__)

RunningMode = vision.RunningMode
PoseLandmark = vision.PoseLandmark
POSE_CONNECTIONS = vision.PoseLandmarksConnections.POSE_LANDMARKS

# Shoulders and hips: the landmarks that decide whether we are really looking at
# a body, and the ones a fall heuristic will lean on later.
TORSO_LANDMARKS = (
    PoseLandmark.LEFT_SHOULDER,
    PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.LEFT_HIP,
    PoseLandmark.RIGHT_HIP,
)


def _visibility(landmark: Any) -> float:
    value = getattr(landmark, "visibility", None)
    return float(value) if value is not None else 0.0


@dataclass(frozen=True)
class PoseConfig:
    """Everything that shapes one landmarker instance."""

    model_variant: ModelVariant = ModelVariant.FULL
    model_path: Path | None = None
    num_poses: int = 1
    min_pose_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    output_segmentation_masks: bool = False
    use_gpu: bool = False
    cache_dir: Path = Path("models")

    # --- cascade-only knobs; ignored by the native path -------------------
    strategy: Strategy = Strategy.AUTO
    detector_variant: DetectorVariant = DetectorVariant.LITE0
    person_score_threshold: float = 0.4
    crop_padding: float = 0.15
    crop_workers: int = 0  # 0 -> min(4, num_poses)
    detect_interval: int = 1
    min_box_px: int = 48

    @property
    def resolved_crop_workers(self) -> int:
        if self.crop_workers > 0:
            return self.crop_workers
        return min(4, max(1, self.num_poses))


@dataclass
class PersonPose:
    """One detected person, ready for drawing or downstream analysis."""

    person_id: int
    landmarks: list[Any]
    world_landmarks: list[Any] = field(default_factory=list)
    score: float = 0.0
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    centroid: tuple[float, float] = (0.0, 0.0)

    def bbox_in_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.bbox
        return (
            int(x1 * width),
            int(y1 * height),
            int(x2 * width),
            int(y2 * height),
        )


def _torso_score(landmarks: Sequence[Any]) -> float:
    values = [
        _visibility(landmarks[index])
        for index in TORSO_LANDMARKS
        if index < len(landmarks)
    ]
    return sum(values) / len(values) if values else 0.0


def _bbox(landmarks: Sequence[Any], visibility_threshold: float = 0.5) -> tuple[float, float, float, float]:
    points = [
        (lm.x, lm.y)
        for lm in landmarks
        if lm.x is not None
        and lm.y is not None
        and _visibility(lm) >= visibility_threshold
    ]
    if not points:
        points = [(lm.x, lm.y) for lm in landmarks if lm.x is not None and lm.y is not None]
    if not points:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _centroid(landmarks: Sequence[Any], bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    left, right = PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP
    if len(landmarks) > max(left, right):
        hips = [landmarks[left], landmarks[right]]
        if all(lm.x is not None and lm.y is not None for lm in hips):
            return (
                sum(lm.x for lm in hips) / 2.0,
                sum(lm.y for lm in hips) / 2.0,
            )
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def person_from_landmarks(
    landmarks: Sequence[Any],
    world_landmarks: Sequence[Any] = (),
    person_id: int = 0,
) -> PersonPose:
    """Build one :class:`PersonPose`, deriving score, bbox and centroid.

    Shared by the native path and the cascade so both produce identical records
    whatever route the landmarks arrived by.
    """
    landmarks = list(landmarks)
    bbox = _bbox(landmarks)
    return PersonPose(
        person_id=person_id,
        landmarks=landmarks,
        world_landmarks=list(world_landmarks),
        score=_torso_score(landmarks),
        bbox=bbox,
        centroid=_centroid(landmarks, bbox),
    )


def build_persons(result: Any, person_ids: Sequence[int] | None = None) -> list[PersonPose]:
    """Turn a raw ``PoseLandmarkerResult`` into :class:`PersonPose` records."""
    if result is None or not result.pose_landmarks:
        return []

    world = result.pose_world_landmarks or []
    return [
        person_from_landmarks(
            landmarks,
            world[index] if index < len(world) else (),
            person_ids[index] if person_ids is not None else index,
        )
        for index, landmarks in enumerate(result.pose_landmarks)
    ]


def best_person(persons: Sequence[PersonPose]) -> PersonPose | None:
    """Pick the most convincing subject -- the "best one person" path.

    With ``num_poses=1`` MediaPipe already returns only the top ROI; this matters
    when several poses are tracked but one subject must be chosen downstream.
    """
    if not persons:
        return None
    return max(persons, key=lambda person: person.score)


class PoseDetector:
    """Thin, logged wrapper around one ``vision.PoseLandmarker``."""

    def __init__(
        self,
        config: PoseConfig,
        running_mode: RunningMode = RunningMode.IMAGE,
        result_callback: Callable[[Any, Any, int], None] | None = None,
    ) -> None:
        self.config = config
        self.running_mode = running_mode
        self._last_timestamp_ms = -1
        self._limiter = RateLimiter(1.0)
        self._first_detect_pending = True

        if running_mode is RunningMode.LIVE_STREAM and result_callback is None:
            raise ValueError("LIVE_STREAM mode requires a result_callback")
        if running_mode is not RunningMode.LIVE_STREAM and result_callback is not None:
            raise ValueError("result_callback is only valid in LIVE_STREAM mode")

        self.model_path = resolve_model(
            config.model_variant, config.model_path, config.cache_dir
        )

        use_gpu = config.use_gpu
        if use_gpu and platform.system() != "Linux":
            logger.warning(
                "GPU delegate is only supported on Linux (host is %s); using CPU",
                platform.system(),
            )
            use_gpu = False

        self._landmarker = self._create(use_gpu, result_callback)

    def _create(self, use_gpu: bool, result_callback):
        delegate_enum = mp_python.BaseOptions.Delegate
        for delegate in ([delegate_enum.GPU] if use_gpu else []) + [delegate_enum.CPU]:
            options = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(self.model_path), delegate=delegate
                ),
                running_mode=self.running_mode,
                num_poses=self.config.num_poses,
                min_pose_detection_confidence=self.config.min_pose_detection_confidence,
                min_pose_presence_confidence=self.config.min_pose_presence_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
                output_segmentation_masks=self.config.output_segmentation_masks,
                result_callback=result_callback,
            )
            try:
                with suppress_native_stderr():
                    landmarker = vision.PoseLandmarker.create_from_options(options)
            except Exception as error:
                if delegate is delegate_enum.CPU:
                    logger.error("failed to create pose landmarker: %s", error)
                    raise
                logger.warning("GPU delegate unavailable (%s); falling back to CPU", error)
                continue

            self.delegate = delegate
            logger.info(
                "pose landmarker created: variant=%s delegate=%s mode=%s num_poses=%d "
                "det=%.2f presence=%.2f track=%.2f",
                self.config.model_variant,
                delegate.name,
                self.running_mode.name,
                self.config.num_poses,
                self.config.min_pose_detection_confidence,
                self.config.min_pose_presence_confidence,
                self.config.min_tracking_confidence,
            )
            return landmarker
        raise RuntimeError("unreachable: no delegate succeeded")  # pragma: no cover

    @staticmethod
    def _to_mp_image(bgr_frame: np.ndarray) -> mp.Image:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

    def _monotonic(self, timestamp_ms: int) -> int:
        """MediaPipe rejects non-increasing timestamps; nudge instead of crash."""
        if timestamp_ms <= self._last_timestamp_ms:
            if self._limiter.ready("timestamp"):
                logger.warning(
                    "timestamp %d not greater than previous %d; bumping",
                    timestamp_ms,
                    self._last_timestamp_ms,
                )
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def _first_call_ctx(self):
        """The graph logs its NORM_RECT/IMAGE_DIMENSIONS warning once, on the
        first detect call after creation -- not per frame -- so only that one
        call needs its stderr swallowed."""
        if self._first_detect_pending:
            self._first_detect_pending = False
            return suppress_native_stderr()
        return contextlib.nullcontext()

    def detect_image(self, bgr_frame: np.ndarray):
        """Blocking single-frame detection (IMAGE mode)."""
        with self._first_call_ctx():
            return self._landmarker.detect(self._to_mp_image(bgr_frame))

    def detect_video(self, bgr_frame: np.ndarray, timestamp_ms: int):
        """Blocking detection for decoded video frames (VIDEO mode)."""
        with self._first_call_ctx():
            return self._landmarker.detect_for_video(
                self._to_mp_image(bgr_frame), self._monotonic(timestamp_ms)
            )

    def detect_async(self, bgr_frame: np.ndarray, timestamp_ms: int) -> int:
        """Non-blocking submit (LIVE_STREAM mode); results reach the callback."""
        timestamp_ms = self._monotonic(timestamp_ms)
        with self._first_call_ctx():
            self._landmarker.detect_async(self._to_mp_image(bgr_frame), timestamp_ms)
        return timestamp_ms

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
            logger.info("pose landmarker closed")

    def __enter__(self) -> "PoseDetector":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
