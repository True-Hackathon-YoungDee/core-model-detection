#!/usr/bin/env python3
"""Extract deterministic, finite RGB fall features from labelled video clips."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Callable, Iterable

import cv2

from fall_detection.engine import build_engine
from fall_detection.evaluation import (
    EvaluationManifest,
    ManifestClip,
    fall_config_fingerprint,
    load_manifest,
)
from fall_detection.fall_config import FallConfig, FallProfile, load_fall_config
from fall_detection.fall_evidence import ImageEvidenceExtractor
from fall_detection.pose import PoseConfig, RunningMode
from fall_detection.runner import PosePipeline
from fall_detection.strategy import Strategy

logger = logging.getLogger("fall-trace-extractor")

ClipExtractor = Callable[
    [ManifestClip, Path, Path, FallConfig, str],
    Iterable[dict[str, object]],
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path.cwd(),
        help="root joined to portable manifest source paths",
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--fall-config", type=Path)
    parser.add_argument(
        "--fall-profile", choices=tuple(profile.value for profile in FallProfile)
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing trace even when its source checksums differ",
    )
    return parser


def validate_output_compatibility(
    output: Path,
    expected_source_checksums: dict[str, str],
    *,
    force: bool,
) -> None:
    """Protect an existing trace from accidental cross-source replacement."""
    if force or not output.exists():
        return
    existing: dict[str, str] = {}
    try:
        for line in output.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("record_type") == "clip":
                existing[str(record["clip_id"])] = str(record["source_sha256"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(
            f"cannot validate existing trace {output}; use --force to replace it"
        ) from error
    if existing != expected_source_checksums:
        raise ValueError(
            "existing trace source checksum set differs from the manifest; "
            "use --force to replace it"
        )


def encode_record(record: dict[str, object]) -> str:
    """Encode one canonical JSONL record while refusing NaN and infinity."""
    try:
        return (
            json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True)
            + "\n"
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"trace records must contain only finite JSON values: {error}") from error


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_sources(
    manifest: EvaluationManifest, source_root: Path
) -> dict[str, Path]:
    """Resolve portable clip paths under *source_root* and verify their bytes."""
    root = source_root.resolve()
    resolved: dict[str, Path] = {}
    for clip in manifest.clips:
        source = (root / clip.source).resolve()
        try:
            source.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"source path for clip {clip.clip_id} escapes --source-root"
            ) from error
        if not source.is_file():
            raise ValueError(
                f"source file does not exist for clip {clip.clip_id}: {source}"
            )
        actual_checksum = file_sha256(source)
        if actual_checksum != clip.source_sha256:
            raise ValueError(
                f"source checksum mismatch for clip {clip.clip_id}: "
                f"expected {clip.source_sha256}, got {actual_checksum}"
            )
        resolved[clip.clip_id] = source
    return resolved


def _clip_records(
    clip: ManifestClip,
    source_path: Path,
    model_path: Path,
    fall_config: FallConfig,
    model_sha256: str,
) -> Iterable[dict[str, object]]:
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"cannot open source for clip {clip.clip_id}: {source_path}")
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_width <= 0 or frame_height <= 0 or not math.isfinite(fps) or fps <= 0.0:
        capture.release()
        raise ValueError(f"invalid video metadata for clip {clip.clip_id}")
    if frame_count <= 0:
        capture.release()
        raise ValueError(f"clip {clip.clip_id} contains no frames")
    measured_duration_s = frame_count / fps
    if abs(measured_duration_s - clip.duration_s) > 1.0 / fps + 1e-6:
        capture.release()
        raise ValueError(
            f"duration mismatch for clip {clip.clip_id}: "
            f"manifest={clip.duration_s}, video={measured_duration_s}"
        )

    yield {
        "schema_version": 1,
        "record_type": "clip",
        "clip_id": clip.clip_id,
        "source_sha256": clip.source_sha256,
        "model_sha256": model_sha256,
        "fall_config_fingerprint": fall_config_fingerprint(fall_config),
        "duration_s": clip.duration_s,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "fps": fps,
        "frame_count": frame_count,
    }

    config = PoseConfig(
        model_path=model_path,
        num_poses=1,
        strategy=Strategy.NATIVE,
    )
    engine = None
    pipeline = PosePipeline(smoothing=True, best_only=True)
    extractors: dict[int, ImageEvidenceExtractor] = {}
    frame_index = 0
    try:
        engine = build_engine(config, RunningMode.VIDEO, Strategy.NATIVE)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_ms = int(frame_index / fps * 1000)
            t_seconds = timestamp_ms / 1000.0
            persons = pipeline.process(engine.infer(frame, timestamp_ms), t_seconds)
            if not persons:
                yield {
                    "schema_version": 1,
                    "record_type": "observation",
                    "clip_id": clip.clip_id,
                    "source_sha256": clip.source_sha256,
                    "frame_index": frame_index,
                    "t_seconds": t_seconds,
                    "person_id": None,
                    "features": None,
                }
            for person in sorted(persons, key=lambda candidate: candidate.person_id):
                extractor = extractors.setdefault(
                    person.person_id, ImageEvidenceExtractor(fall_config)
                )
                features = extractor.update(
                    person,
                    t_seconds,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
                yield {
                    "schema_version": 1,
                    "record_type": "observation",
                    "clip_id": clip.clip_id,
                    "source_sha256": clip.source_sha256,
                    "frame_index": frame_index,
                    "t_seconds": t_seconds,
                    "person_id": person.person_id,
                    "features": dataclasses.asdict(features),
                }
            frame_index += 1
    finally:
        capture.release()
        if engine is not None:
            engine.close()
    if frame_index != frame_count:
        raise ValueError(
            f"decoded frame count mismatch for clip {clip.clip_id}: "
            f"metadata={frame_count}, decoded={frame_index}"
        )


def extract_manifest(
    manifest: EvaluationManifest,
    sources: dict[str, Path],
    model_path: Path,
    output: Path,
    *,
    fall_config: FallConfig,
    force: bool,
    clip_extractor: ClipExtractor = _clip_records,
) -> None:
    """Extract all manifest clips to one atomically replaced JSONL trace."""
    if not model_path.is_file():
        raise ValueError(f"model file does not exist: {model_path}")
    model_sha256 = file_sha256(model_path)
    expected = {clip.clip_id: clip.source_sha256 for clip in manifest.clips}
    validate_output_compatibility(output, expected, force=force)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            for clip in manifest.clips:
                logger.info("extracting %s from %s", clip.clip_id, sources[clip.clip_id])
                for record in clip_extractor(
                    clip,
                    sources[clip.clip_id],
                    model_path,
                    fall_config,
                    model_sha256,
                ):
                    temporary.write(encode_record(record))
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        manifest = load_manifest(args.manifest)
        fall_config = load_fall_config(args.fall_config, args.fall_profile)
        sources = resolve_sources(manifest, args.source_root)
        extract_manifest(
            manifest,
            sources,
            args.model_path.resolve(),
            args.output.resolve(),
            fall_config=fall_config,
            force=args.force,
        )
    except ValueError as error:
        logger.error("extraction failed: %s", error)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
