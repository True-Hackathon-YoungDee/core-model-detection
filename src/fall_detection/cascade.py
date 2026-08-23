"""Cascaded pose estimation: person boxes first, then one landmarker per body.

Stage 1 finds every person in the frame (:mod:`fall_detection.person_detector`).
Stage 2 crops each one and letterboxes it to a square. Stage 3 runs a
single-person landmarker over each crop, several at a time. Stage 4 maps the
crop-local coordinates back onto the full frame.

Why this exists: handing a multi-person frame straight to one landmarker with
``num_poses > 1`` returns one skeleton and silently discards the rest. Measured
on 1280x720 frames with three real bodies, the native path reported one person
and this path reported three.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .logging_config import suppress_native_stderr
from .models import resolve_model
from .person_detector import PersonBox, PersonDetector, merge_boxes, pad_box
from .pose import PoseConfig, PersonPose, person_from_landmarks

logger = logging.getLogger(__name__)


def square_letterbox(crop: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    """Pad a crop to a square with a constant border.

    The landmarker resizes its input to a square, so a tall person box gets
    squashed horizontally. Padding instead of squashing measurably raises torso
    confidence on edge-clipped bodies (0.91 -> 0.99 in the probe), and padding
    beats widening the box with real pixels, which just drags in the neighbour.

    Returns ``(square, side, pad_x, pad_y)``.
    """
    height, width = crop.shape[:2]
    side = max(height, width)
    pad_x = (side - width) // 2
    pad_y = (side - height) // 2
    if side == height and side == width:
        return crop, side, 0, 0
    square = cv2.copyMakeBorder(
        crop,
        pad_y,
        side - height - pad_y,
        pad_x,
        side - width - pad_x,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return square, side, pad_x, pad_y


def remap_landmarks(
    landmarks: Sequence[Any],
    box: PersonBox,
    side: int,
    pad_x: int,
    pad_y: int,
    frame_width: int,
    frame_height: int,
) -> list[Any]:
    """Lift crop-local normalized coordinates back into frame coordinates."""
    out = []
    for landmark in landmarks:
        updates: dict[str, float] = {}
        if landmark.x is not None:
            updates["x"] = (box.x1 + landmark.x * side - pad_x) / frame_width
        if landmark.y is not None:
            updates["y"] = (box.y1 + landmark.y * side - pad_y) / frame_height
        if landmark.z is not None:
            # z is expressed on the same scale as x, so it has to shrink by the
            # same factor the crop did -- otherwise depth reads far too deep.
            updates["z"] = landmark.z * side / frame_width
        out.append(replace(landmark, **updates) if updates else landmark)
    return out


def boxes_from_persons(
    persons: Sequence[PersonPose],
    frame_width: int,
    frame_height: int,
    padding: float,
    min_box_px: int,
) -> list[PersonBox]:
    """Reuse last frame's pose bboxes as this frame's crops (ROI carry)."""
    boxes = []
    for person in persons:
        x1, y1, x2, y2 = person.bbox_in_pixels(frame_width, frame_height)
        box = pad_box(x1, y1, x2, y2, person.score, frame_width, frame_height, padding)
        if box.short_side >= min_box_px:
            boxes.append(box)
    return merge_boxes(boxes)


class CascadePoseEstimator:
    """Person detector + a pool of single-person landmarkers."""

    def __init__(self, config: PoseConfig) -> None:
        self.config = config
        self.workers = config.resolved_crop_workers

        # GPU delegates are bound to the thread that created them and the crop
        # pool is multi-threaded, so the cascade stays on CPU by design.
        if config.use_gpu:
            logger.warning(
                "cascade runs on CPU: GPU delegates are thread-affine and crops "
                "are inferred in parallel; use --detector native for GPU"
            )

        self.detector = PersonDetector(
            variant=config.detector_variant,
            score_threshold=config.person_score_threshold,
            cache_dir=config.cache_dir,
            use_gpu=False,
        )

        model_path = resolve_model(config.model_variant, config.model_path, config.cache_dir)
        self._pool = [self._create_landmarker(model_path) for _ in range(self.workers)]
        self._free = list(self._pool)
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=self.workers, thread_name_prefix="crop"
        )
        self._frame_index = 0
        self._last_box_count = 0
        logger.info(
            "cascade created: crop model=%s workers=%d padding=%.2f "
            "detect_interval=%d min_box=%dpx",
            config.model_variant,
            self.workers,
            config.crop_padding,
            config.detect_interval,
            config.min_box_px,
        )

    def _create_landmarker(self, model_path: Path):
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(model_path),
                delegate=mp_python.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=self.config.min_pose_detection_confidence,
            min_pose_presence_confidence=self.config.min_pose_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            output_segmentation_masks=self.config.output_segmentation_masks,
        )
        with suppress_native_stderr():
            return vision.PoseLandmarker.create_from_options(options)

    def _acquire(self):
        with self._lock:
            return self._free.pop()

    def _release(self, landmarker) -> None:
        with self._lock:
            self._free.append(landmarker)

    def _infer_crop(
        self, frame: np.ndarray, box: PersonBox, frame_width: int, frame_height: int
    ) -> PersonPose | None:
        crop = frame[box.y1 : box.y2, box.x1 : box.x2]
        if crop.size == 0:
            return None
        square, side, pad_x, pad_y = square_letterbox(crop)
        rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

        landmarker = self._acquire()
        try:
            result = landmarker.detect(image)
        finally:
            self._release(landmarker)

        if not result.pose_landmarks:
            return None
        world = result.pose_world_landmarks or []
        landmarks = remap_landmarks(
            result.pose_landmarks[0], box, side, pad_x, pad_y, frame_width, frame_height
        )
        # World landmarks stay untouched: they are metric and hip-centred, so the
        # per-crop values are already correct -- and better than full-frame ones.
        return person_from_landmarks(landmarks, world[0] if world else ())

    def infer(
        self,
        bgr_frame: np.ndarray,
        timestamp_ms: int = 0,
        previous: Sequence[PersonPose] = (),
    ) -> list[PersonPose]:
        """Detect every person in the frame and landmark each of them."""
        frame_height, frame_width = bgr_frame.shape[:2]
        interval = max(1, self.config.detect_interval)
        carry_only = bool(previous) and self._frame_index % interval != 0
        self._frame_index += 1

        started = time.monotonic()
        boxes: list[PersonBox] = []
        if carry_only:
            boxes = boxes_from_persons(
                previous,
                frame_width,
                frame_height,
                self.config.crop_padding,
                self.config.min_box_px,
            )
            # Carrying fewer regions than the detector last found would hide
            # someone until the next scheduled detector run. Compare against what
            # was actually there, not against --num-poses: a scene with three
            # people and a cap of four must still be allowed to carry.
            if len(boxes) < self._last_box_count:
                boxes = []
        if not boxes:
            boxes = self.detector.detect(
                bgr_frame,
                padding=self.config.crop_padding,
                min_box_px=self.config.min_box_px,
                limit=self.config.num_poses,
            )
            self._last_box_count = len(boxes)
            carry_only = False
        detect_ms = (time.monotonic() - started) * 1000.0

        started = time.monotonic()
        if not boxes:
            persons: list[PersonPose] = []
        elif len(boxes) == 1 or self.workers == 1:
            persons = [
                person
                for person in (
                    self._infer_crop(bgr_frame, box, frame_width, frame_height)
                    for box in boxes
                )
                if person is not None
            ]
        else:
            futures = [
                self._executor.submit(
                    self._infer_crop, bgr_frame, box, frame_width, frame_height
                )
                for box in boxes
            ]
            persons = [person for person in (f.result() for f in futures) if person is not None]
        crop_ms = (time.monotonic() - started) * 1000.0

        for index, person in enumerate(persons):
            person.person_id = index

        logger.debug(
            "cascade ts=%dms boxes=%d poses=%d %s=%.0fms crops=%.0fms",
            timestamp_ms,
            len(boxes),
            len(persons),
            "carry" if carry_only else "detect",
            detect_ms,
            crop_ms,
        )
        return persons

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        for landmarker in self._pool:
            landmarker.close()
        self._pool = []
        self._free = []
        self.detector.close()
        logger.info("cascade closed (%d crop landmarkers)", self.workers)

    def __enter__(self) -> "CascadePoseEstimator":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
