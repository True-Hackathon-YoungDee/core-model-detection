"""Resolution and caching of MediaPipe Pose Landmarker model bundles."""

from __future__ import annotations

import logging
import os
import time
import urllib.request
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

_URL_TEMPLATE = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{variant}/float16/latest/pose_landmarker_{variant}.task"
)
_DETECTOR_URL_TEMPLATE = (
    "https://storage.googleapis.com/mediapipe-models/object_detector/"
    "{name}/{precision}/1/{name}.tflite"
)

DEFAULT_CACHE_DIR = Path("models")
_CHUNK = 1 << 16


class ModelVariant(str, Enum):
    """Pose Landmarker bundles, cheapest to most accurate."""

    LITE = "lite"
    FULL = "full"
    HEAVY = "heavy"

    @property
    def filename(self) -> str:
        return f"pose_landmarker_{self.value}.task"

    @property
    def url(self) -> str:
        return _URL_TEMPLATE.format(variant=self.value)

    def __str__(self) -> str:
        return self.value


class DetectorVariant(str, Enum):
    """Person-box models for the cascade's first stage.

    ``lite0`` is the default on measurement, not on size: probed side by side on
    crowded frames, the int8 lite0 bundle recalled people that the larger
    float16 lite2 bundle missed entirely.
    """

    LITE0 = "efficientdet_lite0"
    LITE2 = "efficientdet_lite2"

    @property
    def precision(self) -> str:
        return "int8" if self is DetectorVariant.LITE0 else "float16"

    @property
    def filename(self) -> str:
        return f"{self.value}.tflite"

    @property
    def url(self) -> str:
        return _DETECTOR_URL_TEMPLATE.format(name=self.value, precision=self.precision)

    def __str__(self) -> str:
        return self.value


# Anything carrying .filename and .url can be cached by ensure_model().
Variant = ModelVariant | DetectorVariant


def ensure_model(
    variant: Variant = ModelVariant.FULL,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    timeout: float = 120.0,
) -> Path:
    """Return a local path to the bundle, downloading it on first use."""
    cache_dir = Path(cache_dir)
    target = cache_dir / variant.filename

    if target.exists() and target.stat().st_size > 0:
        logger.info("model %s ready at %s", variant, target)
        return target

    cache_dir.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    logger.info("downloading %s model from %s", variant, variant.url)
    started = time.monotonic()

    try:
        with urllib.request.urlopen(variant.url, timeout=timeout) as response:
            with open(partial, "wb") as handle:
                while chunk := response.read(_CHUNK):
                    handle.write(chunk)
        # Rename only once complete: a killed download must never leave a
        # truncated .task behind for MediaPipe to choke on later.
        os.replace(partial, target)
    except BaseException:
        partial.unlink(missing_ok=True)
        logger.error("download of %s model failed", variant)
        raise

    size_mb = target.stat().st_size / (1024 * 1024)
    logger.info(
        "model %s downloaded to %s (%.1f MB in %.1fs)",
        variant,
        target,
        size_mb,
        time.monotonic() - started,
    )
    return target


def resolve_model(
    variant: Variant = ModelVariant.FULL,
    model_path: Path | str | None = None,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
) -> Path:
    """Use an explicit bundle path when given, otherwise the cached variant."""
    if model_path is not None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"model bundle not found: {path}")
        logger.info("using model bundle %s", path)
        return path
    return ensure_model(variant, cache_dir)
