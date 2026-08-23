"""Stage one of the cascade: where are the people?

MediaPipe's pose landmarker carries its own person detector, but that one is
built to propose a *single* dominant region per frame. Handing it a room full of
people gets you one skeleton. This module runs a general object detector first,
keeps only the ``person`` class, and hands each body to the landmarker
separately.

The detector runs in ``RunningMode.IMAGE`` on purpose: stateless, no timestamp
contract to honour, and therefore safe to call from whichever thread owns the
cascade.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .logging_config import suppress_native_stderr
from .models import DEFAULT_CACHE_DIR, DetectorVariant, resolve_model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PersonBox:
    """A padded, frame-clamped person region in pixels."""

    x1: int
    y1: int
    x2: int
    y2: int
    score: float

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    @property
    def short_side(self) -> int:
        return min(self.width, self.height)


def _iou(a: PersonBox, b: PersonBox) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    overlap = max(0, x2 - x1) * max(0, y2 - y1)
    if overlap == 0:
        return 0.0
    union = a.area + b.area - overlap
    return overlap / union if union > 0 else 0.0


def merge_boxes(boxes: list[PersonBox], iou_threshold: float = 0.55) -> list[PersonBox]:
    """Suppress overlapping boxes, highest score first.

    efficientdet already applies NMS, but padding each box re-introduces
    overlaps, and two padded neighbours would otherwise both crop the same body.
    """
    kept: list[PersonBox] = []
    for box in sorted(boxes, key=lambda b: b.score, reverse=True):
        if all(_iou(box, other) < iou_threshold for other in kept):
            kept.append(box)
    return kept


def pad_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    score: float,
    width: int,
    height: int,
    padding: float,
) -> PersonBox:
    """Grow a box by a fraction of its own size, clamped to the frame.

    Detector boxes hug the torso; an outstretched arm or a kicked-out leg sits
    outside them, and a clipped limb is exactly the landmark a fall heuristic
    needs most.
    """
    pad_x = int((x2 - x1) * padding)
    pad_y = int((y2 - y1) * padding)
    return PersonBox(
        x1=max(0, x1 - pad_x),
        y1=max(0, y1 - pad_y),
        x2=min(width, x2 + pad_x),
        y2=min(height, y2 + pad_y),
        score=score,
    )


class PersonDetector:
    """Thin, logged wrapper around one ``vision.ObjectDetector``."""

    def __init__(
        self,
        variant: DetectorVariant = DetectorVariant.LITE0,
        score_threshold: float = 0.4,
        max_results: int = 10,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        use_gpu: bool = False,
    ) -> None:
        self.variant = variant
        self.score_threshold = score_threshold
        self.model_path = resolve_model(variant, None, cache_dir)

        if use_gpu and platform.system() != "Linux":
            logger.warning("GPU delegate is Linux only (host is %s); using CPU", platform.system())
            use_gpu = False

        delegate_enum = mp_python.BaseOptions.Delegate
        for delegate in ([delegate_enum.GPU] if use_gpu else []) + [delegate_enum.CPU]:
            options = vision.ObjectDetectorOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(self.model_path), delegate=delegate
                ),
                running_mode=vision.RunningMode.IMAGE,
                score_threshold=score_threshold,
                category_allowlist=["person"],
                max_results=max_results,
            )
            try:
                with suppress_native_stderr():
                    self._detector = vision.ObjectDetector.create_from_options(options)
            except Exception as error:
                if delegate is delegate_enum.CPU:
                    logger.error("failed to create person detector: %s", error)
                    raise
                logger.warning("GPU delegate unavailable (%s); falling back to CPU", error)
                continue
            self.delegate = delegate
            logger.info(
                "person detector created: variant=%s delegate=%s score>=%.2f max_results=%d",
                variant,
                delegate.name,
                score_threshold,
                max_results,
            )
            break

    def detect(
        self,
        bgr_frame: np.ndarray,
        padding: float = 0.15,
        min_box_px: int = 48,
        limit: int | None = None,
    ) -> list[PersonBox]:
        """Return padded, deduplicated person boxes, best score first."""
        height, width = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._detector.detect(image)

        boxes: list[PersonBox] = []
        too_small = 0
        for detection in result.detections or []:
            raw = detection.bounding_box
            score = detection.categories[0].score if detection.categories else 0.0
            box = pad_box(
                raw.origin_x,
                raw.origin_y,
                raw.origin_x + raw.width,
                raw.origin_y + raw.height,
                float(score),
                width,
                height,
                padding,
            )
            # A 20px crop upscaled to the model input is noise, not a person.
            if box.short_side < min_box_px:
                too_small += 1
                continue
            boxes.append(box)

        boxes = merge_boxes(boxes)
        if limit is not None and len(boxes) > limit:
            logger.debug("capping %d person boxes to --num-poses=%d", len(boxes), limit)
            boxes = boxes[:limit]
        if too_small:
            logger.debug("dropped %d person boxes under %dpx", too_small, min_box_px)
        return boxes

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()
            self._detector = None
            logger.info("person detector closed")

    def __enter__(self) -> "PersonDetector":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
