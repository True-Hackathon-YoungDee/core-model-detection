from __future__ import annotations

import importlib.util
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from fall_detection.evaluation import load_manifest, replay_trace
from fall_detection.evaluation import ManifestClip
from fall_detection.fall_config import FallConfig


SCRIPT = Path(__file__).parents[1] / "scripts" / "extract_fall_traces.py"


def _script_module():
    spec = importlib.util.spec_from_file_location("extract_fall_traces", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extractor_cli_exposes_portable_source_and_model_overrides():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--manifest" in completed.stdout
    assert "--output" in completed.stdout
    assert "--source-root" in completed.stdout
    assert "--model-path" in completed.stdout
    assert "--force" in completed.stdout


def test_existing_trace_with_different_source_checksum_requires_force(tmp_path: Path):
    output = tmp_path / "trace.jsonl"
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "clip",
                "clip_id": "clip-a",
                "source_sha256": "a" * 64,
                "duration_s": 1.0,
                "frame_width": 640,
                "frame_height": 360,
                "fps": 1.0,
                "frame_count": 1,
            }
        )
        + "\n"
    )
    expected = {"clip-a": "b" * 64}

    with pytest.raises(ValueError, match="source checksum.*--force"):
        _script_module().validate_output_compatibility(output, expected, force=False)

    _script_module().validate_output_compatibility(output, expected, force=True)
    _script_module().validate_output_compatibility(
        output, {"clip-a": "a" * 64}, force=False
    )


def test_trace_record_encoding_is_deterministic_and_rejects_non_finite_numbers():
    assert _script_module().encode_record({"z": 1, "a": 2.5}) == '{"a":2.5,"z":1}\n'

    with pytest.raises(ValueError, match="finite"):
        _script_module().encode_record({"motion_bh_s": math.nan})


@pytest.mark.parametrize(
    ("create_source", "expected_error"),
    [(False, "source file does not exist"), (True, "source checksum mismatch")],
)
def test_source_resolution_requires_present_checksum_matched_clip(
    tmp_path: Path, create_source: bool, expected_error: str
):
    source = tmp_path / "video" / "input" / "clip.mp4"
    if create_source:
        source.parent.mkdir(parents=True)
        source.write_bytes(b"not the labelled clip")
    manifest_path = tmp_path / "manifest.toml"
    manifest_path.write_text(
        "\n".join(
            [
                "schema_version = 1",
                'dataset = "unit-test"',
                'trace = "trace.jsonl"',
                f'trace_sha256 = "{"0" * 64}"',
                "[[clips]]",
                'id = "clip-a"',
                'source = "video/input/clip.mp4"',
                f'source_sha256 = "{"a" * 64}"',
                'subject = "subject-a"',
                'trial = "trial-a"',
                'camera = "camera-a"',
                'split = "test"',
                "duration_s = 1.0",
            ]
        )
        + "\n"
    )

    with pytest.raises(ValueError, match=expected_error):
        _script_module().resolve_sources(load_manifest(manifest_path), tmp_path)


def test_extract_manifest_writes_a_replayable_atomic_trace(tmp_path: Path):
    source = tmp_path / "video" / "input" / "clip.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"labelled clip fixture")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.toml"
    manifest_path.write_text(
        "\n".join(
            [
                "schema_version = 1",
                'dataset = "unit-test"',
                'trace = "trace.jsonl"',
                f'trace_sha256 = "{"0" * 64}"',
                "[[clips]]",
                'id = "clip-a"',
                'source = "video/input/clip.mp4"',
                f'source_sha256 = "{source_sha}"',
                'subject = "subject-a"',
                'trial = "trial-a"',
                'camera = "camera-a"',
                'split = "test"',
                "duration_s = 1.0",
            ]
        )
        + "\n"
    )
    output = tmp_path / "trace.jsonl"
    model = tmp_path / "pose.task"
    model.write_bytes(b"model")

    def fake_clip_records(clip, source_path, model_path):
        assert source_path == source
        assert model_path == model
        yield {
            "schema_version": 1,
            "record_type": "clip",
            "clip_id": clip.clip_id,
            "source_sha256": clip.source_sha256,
            "duration_s": 1.0,
            "frame_width": 640,
            "frame_height": 360,
            "fps": 1.0,
            "frame_count": 1,
        }
        yield {
            "schema_version": 1,
            "record_type": "observation",
            "clip_id": clip.clip_id,
            "source_sha256": clip.source_sha256,
            "frame_index": 0,
            "t_seconds": 0.0,
            "person_id": None,
            "features": None,
        }

    module = _script_module()
    manifest = load_manifest(manifest_path)
    module.extract_manifest(
        manifest,
        module.resolve_sources(manifest, tmp_path),
        model,
        output,
        force=False,
        clip_extractor=fake_clip_records,
    )

    replay = replay_trace(output, "temporal-fsm", FallConfig())
    assert replay["clips"][0]["clip_id"] == "clip-a"
    assert replay["clips"][0]["incidents"] == []
    assert list(tmp_path.glob(".trace.jsonl.*.tmp")) == []


def test_clip_extraction_validates_finite_video_metadata_before_model_start(
    tmp_path: Path, monkeypatch
):
    module = _script_module()

    class FakeCapture:
        def isOpened(self):
            return True

        def get(self, key):
            return {
                module.cv2.CAP_PROP_FRAME_WIDTH: 640,
                module.cv2.CAP_PROP_FRAME_HEIGHT: 360,
                module.cv2.CAP_PROP_FPS: 30.0,
                module.cv2.CAP_PROP_FRAME_COUNT: 30,
            }[key]

        def release(self):
            return None

    monkeypatch.setattr(module.cv2, "VideoCapture", lambda _: FakeCapture())
    clip = ManifestClip(
        clip_id="clip-a",
        source=Path("video/input/clip.mp4"),
        source_sha256="a" * 64,
        subject="subject-a",
        trial="trial-a",
        camera="camera-a",
        split="test",
        duration_s=1.0,
        events=(),
    )

    header = next(module._clip_records(clip, tmp_path / "clip.mp4", tmp_path / "model.task"))

    assert header["fps"] == 30.0
    assert header["frame_count"] == 30
