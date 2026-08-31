"""Sequential, labelled HTTPS video batches without source checksums."""

from __future__ import annotations

import json
import os
import sys
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Mapping

from .evaluation import binary_classification_summary
from .fall_config import load_fall_config
from .fall_state import FallStateManager
from .fall_telemetry import event_record, jsonl_line
from .pose import PoseConfig
from .runner import VideoFileRunner


SCHEMA_VERSION = 1
_LABELS = frozenset(("fall", "normal"))


@dataclass(frozen=True)
class LinkClip:
    clip_id: str
    url: str
    label: str


@dataclass(frozen=True)
class LinkBatch:
    path: Path
    dataset: str
    batch: str
    clips: tuple[LinkClip, ...]


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _simple_name(value: object, name: str) -> str:
    result = _nonempty_string(value, name)
    if result in (".", "..") or Path(result).name != result:
        raise ValueError(f"{name} must be a simple name")
    return result


def _exact_keys(table: Mapping[str, Any], required: set[str], name: str) -> None:
    missing = required - set(table)
    unknown = set(table) - required
    if missing:
        raise ValueError(f"{name} requires {sorted(missing)[0]}")
    if unknown:
        raise ValueError(f"unknown {name} key: {sorted(unknown)[0]}")


def _https_url(value: object, name: str) -> str:
    url = _nonempty_string(value, name)
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ValueError(f"{name} must be an HTTPS URL")
    return url


def load_link_batch(path: str | Path) -> LinkBatch:
    """Load a strict direct-link descriptor with one binary label per clip."""
    batch_path = Path(path)
    try:
        with batch_path.open("rb") as file:
            document = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"cannot load link batch {batch_path}: {error}") from error
    _exact_keys(document, {"schema_version", "dataset", "batch", "clips"}, "link batch")
    if type(document["schema_version"]) is not int or document["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported link batch schema_version: {document['schema_version']!r}")
    dataset = _simple_name(document["dataset"], "dataset")
    batch = _simple_name(document["batch"], "batch")
    raw_clips = document["clips"]
    if not isinstance(raw_clips, list) or not raw_clips:
        raise ValueError("clips must be a non-empty array of TOML tables")
    clips: list[LinkClip] = []
    seen: set[str] = set()
    for index, raw_clip in enumerate(raw_clips):
        name = f"clips[{index}]"
        if not isinstance(raw_clip, dict):
            raise ValueError(f"{name} must be a TOML table")
        _exact_keys(raw_clip, {"id", "url", "label"}, name)
        clip_id = _simple_name(raw_clip["id"], f"{name}.id")
        if clip_id in seen:
            raise ValueError(f"duplicate clip id: {clip_id}")
        seen.add(clip_id)
        label = _nonempty_string(raw_clip["label"], f"{name}.label")
        if label not in _LABELS:
            raise ValueError(f"{name}.label must be 'fall' or 'normal'")
        clips.append(LinkClip(clip_id, _https_url(raw_clip["url"], f"{name}.url"), label))
    return LinkBatch(batch_path.resolve(), dataset, batch, tuple(clips))


class _HTTPSOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme.lower() != "https":
            raise urllib.error.URLError("refusing redirect to non-HTTPS URL")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download_url(url: str, target: Path) -> None:
    """Stream one HTTPS URL atomically into the caller-owned data root."""
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
    request = urllib.request.Request(url)
    opener = urllib.request.build_opener(_HTTPSOnlyRedirectHandler())
    try:
        with opener.open(request, timeout=30) as response, staging.open("wb") as output:
            status = response.getcode()
            if not 200 <= status < 300:
                raise OSError(f"HTTP {status}")
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        staging.replace(target)
    finally:
        staging.unlink(missing_ok=True)


def _contained(root: Path, candidate: Path) -> Path:
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    if not resolved_candidate.is_relative_to(resolved_root):
        raise ValueError(f"path escapes data-root: {candidate}")
    return resolved_candidate


def _clip_source(destination: Path, clip: LinkClip) -> Path:
    suffix = Path(urllib.parse.urlsplit(clip.url).path).suffix.lower()
    return _contained(destination, destination / f"{clip.clip_id}{suffix or '.video'}")


def _append(handle, record: dict[str, object]) -> None:
    handle.write(jsonl_line(record))
    handle.flush()
    os.fsync(handle.fileno())


def _outcome(label: str, predicted_positive: bool) -> str:
    actual_positive = label == "fall"
    if actual_positive and predicted_positive:
        return "TP"
    if predicted_positive:
        return "FP"
    if actual_positive:
        return "FN"
    return "TN"


def _summary_from_jsonl(path: Path, batch: LinkBatch) -> dict[str, object]:
    records: list[tuple[bool, bool]] = []
    failures = 0
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"malformed result log line {number}: {error.msg}") from error
        if not isinstance(record, dict) or record.get("type") != "clip_result":
            continue
        if record.get("dataset") != batch.dataset or record.get("batch") != batch.batch:
            raise ValueError(f"result log does not match this link batch on line {number}")
        if record.get("status") != "success":
            failures += 1
            continue
        actual_label = record.get("actual_label")
        predicted_label = record.get("predicted_label")
        if actual_label not in _LABELS or predicted_label not in _LABELS:
            raise ValueError(f"malformed successful clip result on line {number}")
        records.append((actual_label == "fall", predicted_label == "fall"))
    return {
        "classified": len(records),
        "failed": failures,
        "total": len(batch.clips),
        **binary_classification_summary(records),
    }


def run_link_batch(
    batch_path: str | Path,
    data_root: str | Path,
    result_log: str | Path,
    *,
    emit: Callable[[dict[str, object]], None] | None = None,
) -> int:
    """Download, classify, and score every labelled link in descriptor order."""
    try:
        batch = load_link_batch(batch_path)
        root = Path(data_root)
        destination = _contained(root, root / batch.dataset / batch.batch)
        log_path = Path(result_log).resolve()
        if log_path.is_relative_to(destination.resolve()):
            raise ValueError("result log must be outside the removable batch directory")
        if log_path.exists() and log_path.stat().st_size:
            raise ValueError("result log must be empty for an unpinned link batch")
        destination.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError) as error:
        sys.stderr.write(f"fall-data: {error}\n")
        return 2

    emitter = emit or (lambda record: sys.stdout.write(jsonl_line(record)))
    failed = False
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        for clip in batch.clips:
            try:
                source = _clip_source(destination, clip)
            except ValueError as error:
                sys.stderr.write(f"fall-data: {error}\n")
                return 2
            if source.exists():
                failed = True
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "type": "clip_result",
                    "status": "failed",
                    "stage": "download",
                    "dataset": batch.dataset,
                    "batch": batch.batch,
                    "clip_id": clip.clip_id,
                    "url": clip.url,
                    "actual_label": clip.label,
                    "error": "local input already exists; choose another data root",
                }
                _append(log, record)
                emitter(record)
                continue
            try:
                _download_url(clip.url, source)
            except (OSError, urllib.error.URLError) as error:
                failed = True
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "type": "clip_result",
                    "status": "failed",
                    "stage": "download",
                    "dataset": batch.dataset,
                    "batch": batch.batch,
                    "clip_id": clip.clip_id,
                    "url": clip.url,
                    "actual_label": clip.label,
                    "error": str(error),
                }
                _append(log, record)
                emitter(record)
                continue

            manager = FallStateManager(load_fall_config())
            incidents: list[dict[str, object]] = []

            def on_frame(persons, t_seconds, frame) -> None:
                events = manager.update(persons, t_seconds, frame_width=frame.shape[1], frame_height=frame.shape[0])
                incidents.extend(
                    event_record(event, event.incident_event)
                    for event in events
                    if event.incident_event in ("detected", "recovered")
                )

            started = monotonic()
            try:
                frames = VideoFileRunner(
                    PoseConfig(), source, display=False, on_frame=on_frame,
                    on_person_lost=manager.forget,
                ).run()
            except Exception as error:
                failed = True
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "type": "clip_result",
                    "status": "failed",
                    "stage": "classify",
                    "dataset": batch.dataset,
                    "batch": batch.batch,
                    "clip_id": clip.clip_id,
                    "url": clip.url,
                    "actual_label": clip.label,
                    "error": str(error),
                }
                _append(log, record)
                emitter(record)
                continue

            predicted_positive = any(incident["event"] == "detected" for incident in incidents)
            record = {
                "schema_version": SCHEMA_VERSION,
                "type": "clip_result",
                "status": "success",
                "dataset": batch.dataset,
                "batch": batch.batch,
                "clip_id": clip.clip_id,
                "url": clip.url,
                "actual_label": clip.label,
                "predicted_label": "fall" if predicted_positive else "normal",
                "outcome": _outcome(clip.label, predicted_positive),
                "frames": frames,
                "elapsed_s": monotonic() - started,
                "incidents": incidents,
            }
            _append(log, record)
            emitter(record)
            try:
                source.unlink()
            except OSError as error:
                failed = True
                cleanup = {
                    "schema_version": SCHEMA_VERSION,
                    "type": "cleanup",
                    "status": "failed",
                    "dataset": batch.dataset,
                    "batch": batch.batch,
                    "clip_id": clip.clip_id,
                    "error": str(error),
                }
                _append(log, cleanup)
                emitter(cleanup)

    summary = _summary_from_jsonl(log_path, batch)
    status = "success" if not failed and summary["classified"] == len(batch.clips) else "failed"
    record = {"schema_version": SCHEMA_VERSION, "type": "summary", "status": status, "dataset": batch.dataset, "batch": batch.batch, **summary}
    with log_path.open("a", encoding="utf-8") as log:
        _append(log, record)
    emitter(record)
    return 0 if status == "success" else 1
