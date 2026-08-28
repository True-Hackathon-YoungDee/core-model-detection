"""Safe, manifest-bound lifecycle operations for approved fall-video batches."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import sys
import time
import tomllib
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .evaluation import EvaluationManifest, _file_sha256, _nonempty_string, load_manifest


BATCH_SCHEMA_VERSION = 1
RECEIPT_NAME = ".fall-data-receipt.json"
_WORKERS = 4


@dataclass(frozen=True)
class BatchClip:
    clip_id: str
    urls: tuple[str, ...]


@dataclass(frozen=True)
class DataBatch:
    path: Path
    dataset: str
    batch: str
    manifest_path: Path
    manifest: EvaluationManifest
    storage_prefix: Path
    clips: tuple[BatchClip, ...]
    fingerprint: str


def _exact_keys(table: Mapping[str, Any], required: set[str], name: str) -> None:
    missing = required - set(table)
    unknown = set(table) - required
    if missing:
        raise ValueError(f"{name} requires {sorted(missing)[0]}")
    if unknown:
        raise ValueError(f"unknown {name} key: {sorted(unknown)[0]}")


def _safe_relative(value: object, name: str) -> Path:
    path = Path(_nonempty_string(value, name))
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"{name} must be a contained relative path")
    return path


def _contained(root: Path, candidate: Path) -> Path:
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    if not resolved_candidate.is_relative_to(resolved_root):
        raise ValueError(f"path escapes data-root: {candidate}")
    return resolved_candidate


def _fingerprint(path: Path) -> str:
    return _file_sha256(path)


def load_batch(path: str | Path) -> DataBatch:
    """Load a strict batch descriptor and bind every entry to its manifest clip."""
    batch_path = Path(path)
    try:
        with batch_path.open("rb") as file:
            document = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"cannot load batch {batch_path}: {error}") from error
    _exact_keys(
        document,
        {"schema_version", "dataset", "batch", "manifest", "storage_prefix", "clips"},
        "batch",
    )
    if type(document["schema_version"]) is not int or document["schema_version"] != BATCH_SCHEMA_VERSION:
        raise ValueError(f"unsupported batch schema_version: {document['schema_version']!r}")
    dataset = _nonempty_string(document["dataset"], "dataset")
    batch_name = _nonempty_string(document["batch"], "batch")
    if Path(dataset).name != dataset or Path(batch_name).name != batch_name:
        raise ValueError("dataset and batch must be simple directory names")
    manifest_relative = Path(_nonempty_string(document["manifest"], "manifest"))
    if manifest_relative.is_absolute():
        raise ValueError("manifest must be a relative path")
    manifest_path = (batch_path.parent / manifest_relative).resolve()
    manifest = load_manifest(manifest_path)
    if manifest.dataset != dataset:
        raise ValueError("batch dataset does not match manifest dataset")
    storage_prefix = _safe_relative(document["storage_prefix"], "storage_prefix")
    if storage_prefix != Path(dataset) / batch_name:
        raise ValueError("storage_prefix must equal dataset/batch")
    raw_clips = document["clips"]
    if not isinstance(raw_clips, list) or not raw_clips:
        raise ValueError("clips must be a non-empty array of TOML tables")
    manifest_by_id = {clip.clip_id: clip for clip in manifest.clips}
    clips: list[BatchClip] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_clips):
        name = f"clips[{index}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{name} must be a TOML table")
        _exact_keys(raw, {"id", "urls"}, name)
        clip_id = _nonempty_string(raw["id"], f"{name}.id")
        if clip_id in seen or clip_id not in manifest_by_id:
            raise ValueError(f"{name}.id must identify one manifest clip")
        seen.add(clip_id)
        urls = raw["urls"]
        if not isinstance(urls, list) or not urls:
            raise ValueError(f"{name}.urls must be a non-empty array")
        ordered_urls = tuple(_nonempty_string(url, f"{name}.urls") for url in urls)
        if any(not url.startswith(("http://", "https://")) for url in ordered_urls):
            raise ValueError(f"{name}.urls must be direct HTTP(S) URLs")
        source = manifest_by_id[clip_id].source
        if not source.is_relative_to(storage_prefix):
            raise ValueError(f"manifest source for {clip_id} is outside storage_prefix")
        _safe_relative(
            source.relative_to(storage_prefix).as_posix(),
            f"manifest source for {clip_id}",
        )
        clips.append(BatchClip(clip_id, ordered_urls))
    if seen != set(manifest_by_id):
        raise ValueError("batch clips must exactly match manifest clips")
    return DataBatch(batch_path.resolve(), dataset, batch_name, manifest_path, manifest, storage_prefix, tuple(clips), _fingerprint(batch_path))


def _request(url: str, *, method: str = "GET", headers: Mapping[str, str] | None = None) -> tuple[int, Mapping[str, str], bytes]:
    request = urllib.request.Request(url, method=method, headers=dict(headers or {}))
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers.items()) if error.headers else {}, error.read()


def _download_one(clip: BatchClip, target: Path, expected_sha256: str) -> None:
    failures: list[str] = []
    for url in clip.urls:
        try:
            status, _, data = _request(url)
            if not 200 <= status < 300:
                raise OSError(f"HTTP {status}")
            digest = hashlib.sha256(data).hexdigest()
            if digest != expected_sha256:
                raise ValueError(f"checksum mismatch for {clip.clip_id}: expected {expected_sha256}, got {digest}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            return
        except (OSError, urllib.error.URLError, ValueError) as error:
            failures.append(f"{url}: {error}")
    if any("checksum mismatch" in failure for failure in failures):
        raise ValueError(failures[-1])
    raise ValueError(f"all mirrors failed for {clip.clip_id}: {'; '.join(failures)}")


def _receipt(batch: DataBatch) -> dict[str, object]:
    return {"schema_version": 1, "dataset": batch.dataset, "batch": batch.batch, "storage_prefix": batch.storage_prefix.as_posix(), "descriptor_sha256": batch.fingerprint, "manifest_sha256": _fingerprint(batch.manifest_path)}


def _validate_complete(destination: Path, batch: DataBatch) -> bool:
    receipt_path = destination / RECEIPT_NAME
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if receipt != _receipt(batch):
        return False
    for clip in batch.manifest.clips:
        source = _contained(destination, destination / clip.source.relative_to(batch.storage_prefix))
        if not source.is_file() or _file_sha256(source) != clip.source_sha256:
            return False
    return True


def download_batch(path: str | Path, data_root: str | Path = "datasets") -> Path:
    batch = load_batch(path)
    root = Path(data_root)
    destination = _contained(root, root / batch.storage_prefix)
    if destination.exists():
        if _validate_complete(destination, batch):
            return destination
        raise ValueError(f"destination exists without a matching complete receipt: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{batch.batch}.{uuid.uuid4().hex}.staging"
    _contained(root, staging)
    try:
        staging.mkdir()
        manifest_by_id = {clip.clip_id: clip for clip in batch.manifest.clips}
        with concurrent.futures.ThreadPoolExecutor(max_workers=_WORKERS) as pool:
            futures = [
                pool.submit(
                    _download_one,
                    item,
                    staging / manifest_by_id[item.clip_id].source.relative_to(batch.storage_prefix),
                    manifest_by_id[item.clip_id].source_sha256,
                )
                for item in batch.clips
            ]
            for future in futures:
                future.result()
        (staging / RECEIPT_NAME).write_text(json.dumps(_receipt(batch), sort_keys=True) + "\n")
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def delete_batch(path: str | Path, data_root: str | Path = "datasets", *, yes: bool = False) -> Path:
    if not yes:
        raise ValueError("refusing deletion without --yes")
    batch = load_batch(path)
    root = Path(data_root)
    destination = _contained(root, root / batch.storage_prefix)
    if not destination.is_dir() or not _validate_complete(destination, batch):
        raise ValueError("receipt is missing, forged, or does not match this batch")
    shutil.rmtree(destination)
    return destination


def _probe_url(url: str) -> dict[str, object]:
    started = time.monotonic()
    try:
        status, headers, _ = _request(url, method="HEAD")
        if status in (405, 501):
            status, headers, _ = _request(url, headers={"Range": "bytes=0-0"})
        return {"url": url, "status": status, "latency_ms": round((time.monotonic() - started) * 1000, 3), "content_type": headers.get("Content-Type"), "ok": 200 <= status < 300}
    except (OSError, urllib.error.URLError) as error:
        return {"url": url, "status": None, "latency_ms": round((time.monotonic() - started) * 1000, 3), "content_type": None, "ok": False, "error": str(error)}


def probe_batch(path: str | Path) -> list[dict[str, object]]:
    batch = load_batch(path)
    results: list[dict[str, object]] = []
    for clip in batch.clips:
        with concurrent.futures.ThreadPoolExecutor(max_workers=_WORKERS) as pool:
            probes = list(pool.map(_probe_url, clip.urls))
        selected = next((probe["url"] for probe in probes if probe["ok"]), None)
        results.append({"clip_id": clip.clip_id, "mirrors": probes, "selected_mirror": selected})
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe, download, and safely delete manifest-bound fall-video batches")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("probe", "download", "delete"):
        command = commands.add_parser(name)
        command.add_argument("--batch", type=Path, required=True)
        command.add_argument("--data-root", type=Path, default=Path("datasets"))
        if name == "delete":
            command.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "probe":
            result: object = probe_batch(args.batch)
        elif args.command == "download":
            result = {"destination": str(download_batch(args.batch, args.data_root))}
        else:
            result = {"deleted": str(delete_batch(args.batch, args.data_root, yes=args.yes))}
        sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
    except ValueError as error:
        sys.stderr.write(f"fall-data: {error}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
