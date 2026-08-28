"""Durable, resumable inference and cleanup for manifest-bound video batches."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from time import monotonic
from typing import Callable

from .data_lifecycle import (
    DataBatch,
    _contained,
    _fingerprint,
    delete_verified_clip,
    download_selected_clips,
    load_batch,
    source_is_verified,
)
from .fall_config import load_fall_config
from .fall_state import FallStateManager
from .fall_telemetry import event_record, jsonl_line
from .pose import PoseConfig
from .runner import VideoFileRunner


SCHEMA_VERSION = 1


def _identity(batch: DataBatch, clip_id: str) -> dict[str, object]:
    manifest_clip = next(clip for clip in batch.manifest.clips if clip.clip_id == clip_id)
    return {
        "schema_version": SCHEMA_VERSION,
        "descriptor_sha256": batch.fingerprint,
        "manifest_sha256": _fingerprint(batch.manifest_path),
        "clip_id": clip_id,
        "source_sha256": manifest_clip.source_sha256,
    }


def _read_state(path: Path, batch: DataBatch) -> tuple[set[str], set[str]]:
    classified: set[str] = set()
    deleted: set[str] = set()
    if not path.exists():
        return classified, deleted
    allowed = {clip.clip_id for clip in batch.clips}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"malformed result log line {number}: {error.msg}") from error
        if not isinstance(record, dict) or record.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"malformed result log line {number}")
        record_type = record.get("type")
        if record_type == "complete":
            expected = {
                "schema_version": SCHEMA_VERSION,
                "descriptor_sha256": batch.fingerprint,
                "manifest_sha256": _fingerprint(batch.manifest_path),
            }
            if any(record.get(key) != value for key, value in expected.items()):
                raise ValueError(f"result log does not match this batch on line {number}")
            continue
        if record_type not in {"download", "classify", "delete"}:
            raise ValueError(f"unknown result log record type on line {number}")
        clip_id = record.get("clip_id")
        if clip_id not in allowed or any(record.get(key) != value for key, value in _identity(batch, clip_id).items()):
            raise ValueError(f"result log does not match this batch on line {number}")
        if record_type == "classify" and record.get("status") == "success":
            classified.add(clip_id)
        if record_type == "delete" and record.get("status") == "success":
            deleted.add(clip_id)
    return classified, deleted


def _append(handle, record: dict[str, object]) -> None:
    handle.write(jsonl_line(record))
    handle.flush()
    os.fsync(handle.fileno())


def run_batch(
    batch_path: str | Path,
    data_root: str | Path,
    result_log: str | Path,
    *,
    emit: Callable[[dict[str, object]], None] | None = None,
) -> int:
    """Run one batch, returning 0, 1, or 2 according to lifecycle outcome."""
    try:
        batch = load_batch(batch_path)
        root = Path(data_root)
        destination = _contained(root, root / batch.storage_prefix)
        log_path = Path(result_log).resolve()
        if log_path.is_relative_to(destination.resolve()):
            raise ValueError("result log must be outside the removable batch directory")
        classified, deleted = _read_state(log_path, batch)
    except (OSError, ValueError) as error:
        sys.stderr.write(f"fall-data: {error}\n")
        return 2

    total = len(batch.clips)
    emitter = emit or (lambda record: sys.stdout.write(jsonl_line(record)))
    completed = {"download": 0, "classify": 0, "delete": 0}

    def progress(kind: str, clip_id: str, status: str, error: str | None = None) -> None:
        completed[kind] += 1
        record: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "type": kind,
            "clip_id": clip_id,
            "status": status,
            "completed": completed[kind],
            "total": total,
            "percent": completed[kind] / total * 100,
        }
        if error is not None:
            record["error"] = error
        emitter(record)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    failed = False
    with log_path.open("a", encoding="utf-8") as log:
        try:
            to_download = {
                clip.clip_id for clip in batch.clips
                if clip.clip_id not in classified and not source_is_verified(batch, root, clip.clip_id)
            }
        except ValueError as error:
            sys.stderr.write(f"fall-data: {error}\n")
            return 2
        outcomes = download_selected_clips(batch, root, to_download) if to_download else {}
        for clip in batch.clips:
            clip_id = clip.clip_id
            error = outcomes.get(clip_id)
            if clip_id in to_download:
                record = {**_identity(batch, clip_id), "type": "download", "status": "success" if error is None else "failed"}
                if error is not None:
                    record["error"] = error
                    failed = True
                _append(log, record)
                progress("download", clip_id, record["status"], error)

        for clip in batch.clips:
            clip_id = clip.clip_id
            if clip_id in classified:
                continue
            try:
                verified = source_is_verified(batch, root, clip_id)
            except ValueError as error:
                sys.stderr.write(f"fall-data: {error}\n")
                return 2
            if not verified:
                failed = True
                continue
            manager = FallStateManager(load_fall_config())
            incidents: list[dict[str, object]] = []
            manifest_clip = next(item for item in batch.manifest.clips if item.clip_id == clip_id)

            def on_frame(persons, t_seconds, frame) -> None:
                events = manager.update(persons, t_seconds, frame_width=frame.shape[1], frame_height=frame.shape[0])
                incidents.extend(event_record(event, event.incident_event) for event in events if event.incident_event in ("detected", "recovered"))

            started = monotonic()
            try:
                frames = VideoFileRunner(PoseConfig(), _contained(destination, destination / manifest_clip.source.relative_to(batch.storage_prefix)), display=False, on_frame=on_frame, on_person_lost=manager.forget).run()
            except Exception as error:
                failed = True
                record = {**_identity(batch, clip_id), "type": "classify", "status": "failed", "error": str(error)}
            else:
                record = {**_identity(batch, clip_id), "type": "classify", "status": "success", "frames": frames, "elapsed_s": monotonic() - started, "incidents": incidents}
                classified.add(clip_id)
            _append(log, record)
            progress("classify", clip_id, record["status"], record.get("error"))

        for clip in batch.clips:
            clip_id = clip.clip_id
            if clip_id not in classified or clip_id in deleted:
                continue
            try:
                delete_verified_clip(batch, root, clip_id)
            except ValueError as error:
                sys.stderr.write(f"fall-data: {error}\n")
                return 2
            except OSError as error:
                failed = True
                record = {**_identity(batch, clip_id), "type": "delete", "status": "failed", "error": str(error)}
            else:
                deleted.add(clip_id)
                record = {**_identity(batch, clip_id), "type": "delete", "status": "success"}
            _append(log, record)
            progress("delete", clip_id, record["status"], record.get("error"))

        success = len(classified) == total and len(deleted) == total and not failed
        _append(log, {"schema_version": SCHEMA_VERSION, "descriptor_sha256": batch.fingerprint, "manifest_sha256": _fingerprint(batch.manifest_path), "type": "complete", "status": "success" if success else "failed", "classified": len(classified), "deleted": len(deleted), "total": total})
        emitter({"schema_version": SCHEMA_VERSION, "type": "complete", "status": "success" if success else "failed", "completed": total if success else min(len(classified), len(deleted)), "total": total, "percent": 100.0 if success else min(len(classified), len(deleted)) / total * 100})
    return 0 if success else 1
