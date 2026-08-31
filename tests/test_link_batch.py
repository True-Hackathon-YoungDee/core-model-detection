from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from fall_detection import link_batch
from fall_detection.fall_fsm import FallAlertKind, FallEvidenceLevel, FallState
from fall_detection.fall_state import FallEvent, FallIncident


def _descriptor(tmp_path: Path, clips: list[tuple[str, str]]) -> Path:
    path = tmp_path / "links.toml"
    lines = ['schema_version = 1', 'dataset = "unit"', 'batch = "batch-a"']
    for clip_id, label in clips:
        lines.extend(
            [
                "",
                "[[clips]]",
                f'id = "{clip_id}"',
                f'url = "https://unit/{clip_id}.mp4"',
                f'label = "{label}"',
            ]
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def _detected_event() -> FallEvent:
    incident = FallIncident(
        "fall-1",
        0,
        FallAlertKind.OBSERVED_FALL,
        FallEvidenceLevel.HIGH,
        FallState.FALL_CONFIRMED,
        0.0,
    )
    return FallEvent(
        0,
        FallState.FALL_CONFIRMED,
        True,
        0.0,
        incident=incident,
        incident_event="detected",
    )


def _runner(monkeypatch, events_per_clip: list[list[FallEvent]], *, fail: str | None = None):
    class Manager:
        def __init__(self, config):
            pass

        def update(self, persons, t_seconds, frame_width, frame_height):
            return events_per_clip.pop(0)

        def forget(self, person_id):
            pass

    class Runner:
        def __init__(self, config, path, **kwargs):
            self.path = Path(path)
            self.on_frame = kwargs["on_frame"]

        def run(self):
            if self.path.stem == fail:
                raise RuntimeError("inference failed")
            self.on_frame([], 0.0, type("Frame", (), {"shape": (40, 80, 3)})())
            return 7

    monkeypatch.setattr(link_batch, "FallStateManager", Manager)
    monkeypatch.setattr(link_batch, "VideoFileRunner", Runner)


def test_run_link_batch_writes_per_clip_outcomes_and_summary_from_jsonl(tmp_path, monkeypatch):
    """A wrong label/prediction branch or metric denominator changes the log."""
    descriptor = _descriptor(
        tmp_path,
        [("tp", "fall"), ("tn", "normal"), ("fp", "normal"), ("fn", "fall")],
    )
    downloaded: list[str] = []

    def download(url: str, target: Path) -> None:
        downloaded.append(url)
        target.write_bytes(b"video")

    monkeypatch.setattr(link_batch, "_download_url", download)
    _runner(monkeypatch, [[_detected_event()], [], [_detected_event()], []])
    log = tmp_path / "result.jsonl"

    assert link_batch.run_link_batch(descriptor, tmp_path / "data", log) == 0

    records = [json.loads(line) for line in log.read_text().splitlines()]
    results = [record for record in records if record["type"] == "clip_result"]
    assert [record["outcome"] for record in results] == ["TP", "TN", "FP", "FN"]
    assert downloaded == [
        "https://unit/tp.mp4",
        "https://unit/tn.mp4",
        "https://unit/fp.mp4",
        "https://unit/fn.mp4",
    ]
    assert records[-1]["clip_confusion"] == {"TP": 1, "TN": 1, "FP": 1, "FN": 1}
    assert records[-1]["classification_metrics"] == {
        "accuracy": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "f1_score": 0.5,
    }
    assert not list((tmp_path / "data" / "unit" / "batch-a").glob("*.mp4"))


def test_run_link_batch_keeps_failed_inference_input_and_excludes_it_from_metrics(tmp_path, monkeypatch):
    """An inference failure must not become a false negative or delete its input."""
    descriptor = _descriptor(tmp_path, [("broken", "fall"), ("normal", "normal")])
    monkeypatch.setattr(link_batch, "_download_url", lambda url, target: target.write_bytes(b"video"))
    _runner(monkeypatch, [[]], fail="broken")
    root = tmp_path / "data"
    log = tmp_path / "result.jsonl"

    assert link_batch.run_link_batch(descriptor, root, log) == 1

    records = [json.loads(line) for line in log.read_text().splitlines()]
    failed = next(record for record in records if record["clip_id"] == "broken")
    assert failed["status"] == "failed"
    assert failed["stage"] == "classify"
    assert records[-1]["clip_confusion"] == {"TP": 0, "TN": 1, "FP": 0, "FN": 0}
    assert (root / "unit" / "batch-a" / "broken.mp4").read_bytes() == b"video"


def test_run_link_batch_rejects_a_nonempty_result_log(tmp_path, monkeypatch):
    """Appending a second unpinned run would silently mix two source versions."""
    descriptor = _descriptor(tmp_path, [("clip", "fall")])
    log = tmp_path / "result.jsonl"
    log.write_text('{"previous":"run"}\n')

    assert link_batch.run_link_batch(descriptor, tmp_path / "data", log) == 2


def test_run_link_batch_does_not_replace_a_retained_failed_input(tmp_path, monkeypatch):
    """A rerun must not overwrite a clip kept for failure diagnosis."""
    descriptor = _descriptor(tmp_path, [("clip", "fall")])
    root = tmp_path / "data"
    source = root / "unit" / "batch-a" / "clip.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"keep me")
    log = tmp_path / "result.jsonl"
    monkeypatch.setattr(
        link_batch,
        "_download_url",
        lambda url, target: pytest.fail("must not replace an existing input"),
    )

    assert link_batch.run_link_batch(descriptor, root, log) == 1

    assert source.read_bytes() == b"keep me"
    record = json.loads(log.read_text().splitlines()[0])
    assert record["stage"] == "download"


def test_link_batch_rejects_https_redirects_to_http():
    """A redirect must not downgrade an unpinned video download to HTTP."""
    request = urllib.request.Request("https://unit/clip.mp4")

    with pytest.raises(urllib.error.URLError, match="non-HTTPS"):
        link_batch._HTTPSOnlyRedirectHandler().redirect_request(
            request, None, 302, "Found", {}, "http://unit/clip.mp4"
        )


def test_link_batch_rejects_duplicate_clip_ids(tmp_path):
    """Duplicate IDs would overwrite one local target and corrupt attribution."""
    descriptor = _descriptor(tmp_path, [("clip", "fall"), ("clip", "normal")])

    with pytest.raises(ValueError, match="duplicate clip id"):
        link_batch.load_link_batch(descriptor)


def test_link_batch_preserves_the_video_suffix_from_its_url(tmp_path):
    """Forcing every remote video to .mp4 can select the wrong decoder."""
    clip = link_batch.LinkClip("clip", "https://unit/clip.gif?download=1", "fall")

    assert link_batch._clip_source(tmp_path, clip) == tmp_path / "clip.gif"


@pytest.mark.parametrize(
    "replacement",
    ['url = "http://unit/clip.mp4"', 'label = "other"'],
)
def test_link_batch_rejects_non_https_urls_and_unknown_labels(tmp_path, replacement):
    """Removing descriptor validation would allow an untrusted or unscored clip."""
    descriptor = _descriptor(tmp_path, [("clip", "fall")])
    descriptor.write_text(descriptor.read_text().replace('url = "https://unit/clip.mp4"', replacement) if replacement.startswith("url") else descriptor.read_text().replace('label = "fall"', replacement))

    with pytest.raises(ValueError):
        link_batch.load_link_batch(descriptor)
