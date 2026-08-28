"""Behavioral regression snapshot over the synthetic ADL/fall corpus.

These scenarios are authored ``FallFeatures`` streams (see
``fall_detection.synthetic_traces``), not recordings of real subjects. The
load-bearing assertions here are the exact per-clip outcomes: this catches
any threshold drift in ``fall_fsm.py`` / ``fall_evidence.py`` that flips a
clip's incident count, kind, evidence level, or detection time -- something
an aggregate accuracy number can miss entirely if two clips flip in
opposite directions. The accuracy-floor assertions below describe how the
FSM currently responds to these authored inputs; they are not a measured
system accuracy claim (see the module docstring on ``synthetic_traces.py``
and the README "Replay regression" section for that distinction).
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from fall_detection.evaluation import evaluate_manifest
from fall_detection.fall_config import FallConfig

_REPO_ROOT = Path(__file__).parents[1]
_MANIFEST = _REPO_ROOT / "evaluation" / "manifests" / "synthetic-adl.toml"

# clip_id -> (evidence_level, detected_at, recovered_at) for every clip that
# must produce exactly one incident.
_EXPECTED_FALLS: dict[str, tuple[str, float, float | None]] = {
    "synth-fall-forward-fast": ("HIGH", 1.75, None),
    "synth-fall-backward-fast": ("HIGH", 2.05, None),
    "synth-fall-lateral-recovered": ("HIGH", 1.7, 3.05),
    "synth-fall-slow-slump": ("HIGH", 2.95, None),
    "synth-fall-low-visibility": ("HIGH", 2.0, None),
    "synth-fall-after-long-quiet": ("HIGH", 9.25, None),
}

# ADL hard negatives and degenerate inputs that must produce zero incidents.
_EXPECTED_NEGATIVES = frozenset(
    {
        "synth-adl-fast-sit",
        "synth-adl-lie-on-bed-brief",
        "synth-adl-bend-pick-up",
        "synth-adl-squat",
        "synth-adl-jump",
        "synth-adl-brisk-walk",
        "synth-adl-sit-on-floor",
        "synth-adl-kneel",
        "synth-degenerate-all-invalid",
        "synth-degenerate-single-observation",
    }
)


def test_synthetic_corpus_pinned_per_clip_outcomes():
    # Construct the config explicitly (not via load_fall_config(None, None))
    # so this pin does not silently track whatever the default profile
    # happens to be in the future.
    report = evaluate_manifest(_MANIFEST, "temporal-fsm", FallConfig())

    assert {clip["clip_id"] for clip in report["clips"]} == (
        set(_EXPECTED_FALLS) | _EXPECTED_NEGATIVES
    )

    for clip in report["clips"]:
        clip_id = clip["clip_id"]
        if clip_id in _EXPECTED_NEGATIVES:
            assert clip["incidents"] == [], clip_id
            assert clip["false_positives"] == 0, clip_id
            continue
        assert len(clip["incidents"]) == 1, clip_id
        incident = clip["incidents"][0]
        evidence_level, detected_at, recovered_at = _EXPECTED_FALLS[clip_id]
        assert incident["kind"] == "OBSERVED_FALL", clip_id
        assert incident["evidence_level"] == evidence_level, clip_id
        assert incident["detected_at"] == pytest.approx(detected_at), clip_id
        assert incident["recovered_at"] == pytest.approx(recovered_at), clip_id


def test_synthetic_corpus_event_counts_and_false_alerts():
    report = evaluate_manifest(_MANIFEST, "temporal-fsm", FallConfig())

    # clip_confusion is per-clip (bool(events) vs bool(incidents)), so a
    # clip with one real fall plus spurious extra alerts would still score
    # a clean TP there -- event_counts / false_alerts_per_hour are what
    # actually expose event-level false-positive pressure.
    assert report["event_counts"] == {
        "labelled": len(_EXPECTED_FALLS),
        "detected": len(_EXPECTED_FALLS),
        "true_positive": len(_EXPECTED_FALLS),
        "false_positive": 0,
        "missed": 0,
    }
    assert report["metrics"]["false_alerts_per_hour"] == pytest.approx(0.0)


def test_synthetic_corpus_meets_accuracy_floors():
    """Not a system-accuracy claim -- see module docstring. This pins that
    the FSM currently clears the requested floors on these authored inputs
    and will fail loudly the day it stops clearing them."""
    report = evaluate_manifest(_MANIFEST, "temporal-fsm", FallConfig())

    metrics = report["classification_metrics"]
    assert metrics["recall"] >= 0.99
    assert metrics["precision"] >= 0.95
    assert metrics["accuracy"] >= 0.98
    assert report["clip_confusion"]["FN"] == 0


def test_synthetic_corpus_beats_naive_vote_baselines():
    """Sanity check that the scenarios are actually discriminating: the
    real temporal FSM should do at least as well as the naive AND/OR vote
    baselines replay also supports. If it doesn't, the scenarios aren't
    testing anything the simplest strategy wouldn't already pass."""
    fsm_metrics = evaluate_manifest(_MANIFEST, "temporal-fsm", FallConfig())[
        "classification_metrics"
    ]
    and_metrics = evaluate_manifest(_MANIFEST, "legacy-and", FallConfig())[
        "classification_metrics"
    ]
    or_metrics = evaluate_manifest(_MANIFEST, "relaxed-or", FallConfig())[
        "classification_metrics"
    ]

    assert fsm_metrics["f1_score"] >= and_metrics["f1_score"]
    assert fsm_metrics["f1_score"] >= or_metrics["f1_score"]


def test_synthetic_generator_is_reproducible(tmp_path: Path):
    # Matching the committed basenames matters: the manifest embeds the
    # trace's filename as a relative path, so a differently-named output
    # would legitimately produce a differently-named manifest line.
    trace_path = tmp_path / "synthetic-adl-v1.jsonl"
    manifest_path = tmp_path / "synthetic-adl.toml"
    subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "generate_synthetic_traces.py"),
            "--output",
            str(trace_path),
            "--manifest-output",
            str(manifest_path),
        ],
        check=True,
        cwd=_REPO_ROOT,
    )

    committed_trace = _REPO_ROOT / "evaluation" / "traces" / "synthetic-adl-v1.jsonl"
    committed_manifest = _REPO_ROOT / "evaluation" / "manifests" / "synthetic-adl.toml"

    assert (
        hashlib.sha256(trace_path.read_bytes()).hexdigest()
        == hashlib.sha256(committed_trace.read_bytes()).hexdigest()
    )
    assert (
        hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        == hashlib.sha256(committed_manifest.read_bytes()).hexdigest()
    )
