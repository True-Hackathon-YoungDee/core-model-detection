from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from fall_detection import data_lifecycle


def _manifest(root: Path, *, checksum: str) -> Path:
    manifest = root / "manifest.toml"
    manifest.write_text(
        "\n".join(
            [
                "schema_version = 1",
                'dataset = "unit"',
                'trace = "trace.jsonl"',
                'trace_sha256 = "' + "0" * 64 + '"',
                "",
                "[[clips]]",
                'id = "clip-a"',
                'source = "unit/batch-a/clip-a.mp4"',
                f'source_sha256 = "{checksum}"',
                'subject = "subject-a"',
                'trial = "trial-a"',
                'camera = "camera-a"',
                "duration_s = 1.0",
            ]
        )
        + "\n"
    )
    return manifest


def _batch(root: Path, manifest: Path, *, urls: list[str] | None = None) -> Path:
    batch = root / "batch.toml"
    batch.write_text(
        "\n".join(
            [
                "schema_version = 1",
                'dataset = "unit"',
                'batch = "batch-a"',
                f'manifest = "{manifest.name}"',
                'storage_prefix = "unit/batch-a"',
                "",
                "[[clips]]",
                'id = "clip-a"',
                "urls = [" + ", ".join(f'\"{url}\"' for url in (urls or ["https://one/clip", "https://two/clip"])) + "]",
            ]
        )
        + "\n"
    )
    return batch


def test_batch_descriptor_rejects_unowned_manifest_source(tmp_path: Path):
    manifest = _manifest(tmp_path, checksum="a" * 64)
    batch = _batch(tmp_path, manifest)
    batch.write_text(batch.read_text().replace('storage_prefix = "unit/batch-a"', 'storage_prefix = "unit/other"'))

    with pytest.raises(ValueError, match="storage_prefix"):
        data_lifecycle.load_batch(batch)


def test_batch_descriptor_rejects_manifest_source_traversal(tmp_path: Path):
    manifest = _manifest(tmp_path, checksum="a" * 64)
    manifest.write_text(
        manifest.read_text().replace(
            "unit/batch-a/clip-a.mp4", "unit/batch-a/../escape.mp4"
        )
    )
    batch = _batch(tmp_path, manifest)

    with pytest.raises(ValueError, match="contained relative path"):
        data_lifecycle.load_batch(batch)


def test_download_falls_back_verifies_and_atomically_promotes(tmp_path: Path, monkeypatch):
    content = b"approved video bytes"
    manifest = _manifest(tmp_path, checksum=hashlib.sha256(content).hexdigest())
    batch = _batch(tmp_path, manifest)
    calls: list[str] = []

    def fetch(url: str, *, method: str = "GET", headers=None):
        calls.append(url)
        if url == "https://one/clip":
            raise OSError("unavailable")
        return 200, {"Content-Type": "video/mp4"}, content

    monkeypatch.setattr(data_lifecycle, "_request", fetch)
    root = tmp_path / "datasets"

    destination = data_lifecycle.download_batch(batch, root)

    assert destination == root / "unit" / "batch-a"
    assert (destination / "clip-a.mp4").read_bytes() == content
    assert (destination / data_lifecycle.RECEIPT_NAME).is_file()
    assert calls == ["https://one/clip", "https://two/clip"]
    assert not list((root / "unit").glob(".batch-a.*.staging"))
    assert data_lifecycle.download_batch(batch, root) == destination


def test_download_rejects_bad_checksum_and_cleans_staging(tmp_path: Path, monkeypatch):
    manifest = _manifest(tmp_path, checksum="a" * 64)
    batch = _batch(tmp_path, manifest, urls=["https://one/clip"])
    monkeypatch.setattr(
        data_lifecycle,
        "_request",
        lambda url, *, method="GET", headers=None: (200, {}, b"wrong"),
    )
    root = tmp_path / "datasets"

    with pytest.raises(ValueError, match="checksum mismatch"):
        data_lifecycle.download_batch(batch, root)

    assert not (root / "unit" / "batch-a").exists()
    assert not list((root / "unit").glob(".batch-a.*.staging"))


def test_probe_uses_ranged_get_when_a_mirror_rejects_head(tmp_path: Path, monkeypatch):
    manifest = _manifest(tmp_path, checksum="a" * 64)
    batch = _batch(tmp_path, manifest)
    requests: list[tuple[str, str, object]] = []

    def request(url: str, *, method: str = "GET", headers=None):
        requests.append((url, method, headers))
        if url == "https://one/clip" and method == "HEAD":
            return 405, {}, b""
        if url == "https://one/clip":
            return 206, {"Content-Type": "video/mp4"}, b"x"
        return 200, {"Content-Type": "video/mp4"}, b""

    monkeypatch.setattr(data_lifecycle, "_request", request)

    report = data_lifecycle.probe_batch(batch)

    assert report[0]["selected_mirror"] == "https://one/clip"
    assert report[0]["mirrors"][0]["content_type"] == "video/mp4"
    assert ("https://one/clip", "GET", {"Range": "bytes=0-0"}) in requests


def test_committed_dataset_batch_templates_link_their_manifests():
    batches = Path(__file__).parents[1] / "evaluation" / "batches"

    for name in ("urfd.example.toml", "up-fall.example.toml"):
        assert data_lifecycle.load_batch(batches / name).clips


def test_delete_requires_matching_receipt_and_stays_in_data_root(tmp_path: Path, monkeypatch):
    content = b"approved video bytes"
    manifest = _manifest(tmp_path, checksum=hashlib.sha256(content).hexdigest())
    batch = _batch(tmp_path, manifest)
    monkeypatch.setattr(
        data_lifecycle,
        "_request",
        lambda url, *, method="GET", headers=None: (200, {}, content),
    )
    root = tmp_path / "datasets"
    destination = data_lifecycle.download_batch(batch, root)

    with pytest.raises(ValueError, match="--yes"):
        data_lifecycle.delete_batch(batch, root, yes=False)
    assert destination.exists()

    (destination / data_lifecycle.RECEIPT_NAME).write_text('{"storage_prefix":"unit/other"}')
    with pytest.raises(ValueError, match="receipt"):
        data_lifecycle.delete_batch(batch, root, yes=True)
    assert destination.exists()
