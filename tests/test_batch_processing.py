from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fall_detection import batch_processing, data_lifecycle
from fall_detection.fall_fsm import FallAlertKind, FallEvidenceLevel, FallState
from fall_detection.fall_state import FallEvent, FallIncident


def _batch(tmp_path: Path, contents: dict[str, bytes]) -> Path:
    manifest = tmp_path / "manifest.toml"
    clips = []
    entries = []
    for clip_id, content in contents.items():
        clips.extend(
            [
                "[[clips]]",
                f'id = "{clip_id}"',
                f'source = "unit/batch-a/{clip_id}.mp4"',
                f'source_sha256 = "{hashlib.sha256(content).hexdigest()}"',
                'subject = "subject"',
                'trial = "trial"',
                'camera = "camera"',
                "duration_s = 1.0",
                "",
            ]
        )
        entries.extend(["[[clips]]", f'id = "{clip_id}"', f'urls = ["https://unit/{clip_id}"]', ""])
    manifest.write_text(
        "\n".join(["schema_version = 1", 'dataset = "unit"', 'trace = "trace.jsonl"', f'trace_sha256 = "{"0" * 64}"', "", *clips])
    )
    batch = tmp_path / "batch.toml"
    batch.write_text("\n".join(["schema_version = 1", 'dataset = "unit"', 'batch = "batch-a"', 'manifest = "manifest.toml"', 'storage_prefix = "unit/batch-a"', "", *entries]))
    return batch


def _runner(monkeypatch, incidents: list[FallEvent] | None = None, failing: set[str] | None = None):
    calls: list[str] = []

    class Manager:
        def __init__(self, config):
            self.incidents = ()

        def update(self, persons, t_seconds, frame_width, frame_height):
            return incidents or []

        def forget(self, person_id):
            pass

    class Runner:
        def __init__(self, config, path, **kwargs):
            self.path = Path(path)
            self.on_frame = kwargs["on_frame"]

        def run(self):
            calls.append(self.path.stem)
            if self.path.stem in (failing or set()):
                raise RuntimeError("inference failed")
            self.on_frame([], 0.0, type("Frame", (), {"shape": (40, 80, 3)})())
            return 7

    monkeypatch.setattr(batch_processing, "FallStateManager", Manager)
    monkeypatch.setattr(batch_processing, "VideoFileRunner", Runner)
    return calls


def test_run_processes_persists_incidents_and_deletes_inputs(tmp_path, monkeypatch):
    contents = {"clip-a": b"a", "clip-b": b"b"}
    batch = _batch(tmp_path, contents)
    monkeypatch.setattr(data_lifecycle, "_request", lambda url, **kwargs: (200, {}, contents[url.rsplit("/", 1)[1]]))
    incident = FallIncident("fall-1", 0, FallAlertKind.PERSISTENT_PRONE, FallEvidenceLevel.MEDIUM, FallState.FALL_CONFIRMED, 0.0)
    events = [FallEvent(0, FallState.FALL_CONFIRMED, True, 0.0, incident=incident, incident_event="detected")]
    _runner(monkeypatch, events)
    emitted: list[dict[str, object]] = []
    log = tmp_path / "result.jsonl"

    assert batch_processing.run_batch(batch, tmp_path / "data", log, emit=emitted.append) == 0

    records = [json.loads(line) for line in log.read_text().splitlines()]
    classifications = [record for record in records if record["type"] == "classify"]
    assert len(classifications) == 2
    assert classifications[0]["incidents"][0]["incident_id"] == "fall-1"
    assert not (tmp_path / "data" / "unit" / "batch-a" / "clip-a.mp4").exists()
    assert [event["percent"] for event in emitted] == [50.0, 100.0, 50.0, 100.0, 50.0, 100.0, 100.0]


def test_run_keeps_clip_when_classification_fails(tmp_path, monkeypatch):
    contents = {"clip-a": b"a", "clip-b": b"b"}
    batch = _batch(tmp_path, contents)
    monkeypatch.setattr(data_lifecycle, "_request", lambda url, **kwargs: (200, {}, contents[url.rsplit("/", 1)[1]]))
    _runner(monkeypatch, failing={"clip-b"})
    root = tmp_path / "data"

    assert batch_processing.run_batch(batch, root, tmp_path / "result.jsonl") == 1

    destination = root / "unit" / "batch-a"
    assert not (destination / "clip-a.mp4").exists()
    assert (destination / "clip-b.mp4").read_bytes() == b"b"


def test_run_resume_retries_cleanup_without_reclassifying(tmp_path, monkeypatch):
    contents = {"clip-a": b"a"}
    batch = _batch(tmp_path, contents)
    monkeypatch.setattr(data_lifecycle, "_request", lambda url, **kwargs: (200, {}, b"a"))
    calls = _runner(monkeypatch)
    original_delete = batch_processing.delete_verified_clip
    monkeypatch.setattr(batch_processing, "delete_verified_clip", lambda *args: (_ for _ in ()).throw(OSError("busy")))
    log = tmp_path / "result.jsonl"
    root = tmp_path / "data"

    assert batch_processing.run_batch(batch, root, log) == 1
    monkeypatch.setattr(batch_processing, "delete_verified_clip", original_delete)
    assert batch_processing.run_batch(batch, root, log) == 0

    assert calls == ["clip-a"]


def test_run_rejects_malformed_log_and_log_inside_batch(tmp_path):
    batch = _batch(tmp_path, {"clip-a": b"a"})
    malformed = tmp_path / "result.jsonl"
    malformed.write_text("not json\n")

    assert batch_processing.run_batch(batch, tmp_path / "data", malformed) == 2
    assert batch_processing.run_batch(batch, tmp_path / "data", tmp_path / "data" / "unit" / "batch-a" / "result.jsonl") == 2


def test_run_rejects_a_result_log_for_a_different_batch(tmp_path):
    batch = _batch(tmp_path, {"clip-a": b"a"})
    log = tmp_path / "result.jsonl"
    log.write_text(json.dumps({"schema_version": 1, "descriptor_sha256": "0" * 64, "manifest_sha256": "0" * 64, "type": "complete"}) + "\n")

    assert batch_processing.run_batch(batch, tmp_path / "data", log) == 2


def test_run_rejects_a_source_symlink_that_escapes_data_root(tmp_path):
    batch = _batch(tmp_path, {"clip-a": b"a"})
    destination = tmp_path / "data" / "unit" / "batch-a"
    destination.mkdir(parents=True)
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"a")
    (destination / "clip-a.mp4").symlink_to(outside)

    assert batch_processing.run_batch(batch, tmp_path / "data", tmp_path / "result.jsonl") == 2
    assert outside.exists()
