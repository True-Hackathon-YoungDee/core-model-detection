# Fall Detection Engine

> **"Most AI fall detectors show pretty boxes on stage. This system saves lives in production when the network fails, frames drop, and black-box models hallucinate."**

[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-red.svg?style=for-the-badge)](#)
[![Architecture: Clean/Hexagonal](https://img.shields.io/badge/Architecture-Partial_Hexagonal-blueviolet?style=for-the-badge)](#)
[![Clinical Grade: FSM Deterministic](https://img.shields.io/badge/Logic-Deterministic_FSM-success?style=for-the-badge)](#)

## Learning this codebase

Onboarding guide for a new contributor. Not a reference doc — those exist
already:

- **[README.md](README.md)** — install, run, CLI flags, strategy benchmarks,
  layout table, programmatic use.
- **[docs/fall-detection.md](docs/fall-detection.md)** — the fall-detection
  operating guide: FSM transitions, config, telemetry, replay evaluation,
  limitations.
- **[CLAUDE.md](CLAUDE.md)** — dev conventions and repo-specific gotchas.

This doc is the missing piece: a reading order and a mental model, so you
don't have to re-derive the architecture by reading 20 files cold.

## What this is

Fall detection on top of MediaPipe Pose (Tasks API): 33 body landmarks per
person from a webcam or video file, one or several people. A fall decision
sits on top of that keypoint layer — pixel-corrected RGB geometry and
duration-weighted evidence feed a deterministic temporal state machine.
`world_landmarks` remain useful for anatomical display, but their hip-centred
coordinates are not treated as room motion or floor height.

## Data flow / mental model

```
cli.py:main()
  -> PoseConfig
  -> VideoFileRunner / LiveStreamRunner        (runner.py)
       -> engine.py:build_engine picks:
            NativeEngine    (pose.py)
            CascadeEngine   (cascade.py + person_detector.py)
       -> PosePipeline:
            tracking.py:IdentityTracker        (stable ids across frames)
            smoothing.py:LandmarkSmoother      (One-Euro filter)
       -> per-frame list[PersonPose]
       -> fall_state.py:FallStateManager       (one tracker per stable id)
            -> fall_evidence.py:ImageEvidenceExtractor / classify_evidence
            -> fall_fsm.py:PersonFallFSM.step()
            -> FallEvent / FallIncident
       -> drawing.py overlay, VideoWriter, telemetry JSONL
```

Two concerns are layered here: **pose estimation** (getting landmarks
reliably, one or many people) and **fall detection** (turning a stream of
landmarks into a state-machine decision). They're separable — you can run
pose-only with `--no-fall-detection`.

## Suggested reading order

1. **`CLAUDE.md`** — conventions and gotchas first, before any source.
2. **`tests/conftest.py`** + **`tests/synthetic_falls.py`** — learn the data
   shapes (`PersonPose`, the axis convention) through fixtures before reading
   the source that produces them. `conftest.py` has basic builders
   (`make_person`, `standing_pose`); `synthetic_falls.py` builds a full
   keyframed standing→lying sequence for FSM/orchestrator tests.
3. **`src/fall_detection/pose.py`** — the core wrapper: `PoseConfig`,
   `PersonPose`, `PoseDetector`. Everything downstream consumes `PersonPose`.
4. **`src/fall_detection/engine.py`** → **`cascade.py`** +
   **`person_detector.py`** — why there are two detection strategies (plain
   full-frame landmarker vs. detect-persons-then-landmark-each-crop) and how
   `Strategy.AUTO` picks between them. See README's
   [Choosing a strategy](README.md#choosing-a-strategy) for the measured
   tradeoffs.
5. **`src/fall_detection/runner.py`** — the orchestration loop:
   `VideoFileRunner` (synchronous per-frame) vs `LiveStreamRunner`
   (producer thread + bounded queue + stall-timeout supervisor). Read
   README's [Notes that bite](README.md#notes-that-bite) before touching
   this file — timestamps, off-thread results, and self-healing live runs
   are all non-obvious.
6. **`src/fall_detection/fall_evidence.py`** →
   **`fall_fsm.py`** → **`fall_state.py`** — the fall-detection layer.
   `fall_evidence.py` computes per-frame `FallFeatures` from pixel landmarks
   (torso angle, bbox aspect ratio, downward speed, etc.) and turns them into
   boolean `FallEvidence` gates. `fall_fsm.py:PersonFallFSM` drives the
   7-state machine per person. `fall_state.py:FallStateManager` owns one
   tracker per stable `person_id` and aggregates events into durable
   incidents. Full transition table lives in
   [docs/fall-detection.md](docs/fall-detection.md) — don't re-derive it from
   source, read that doc.
7. **`src/fall_detection/cli.py`** — how everything above gets wired
   together and exposed as flags.

**Not on the hot path:** `biomechanics.py`, `discriminators.py`, and
`FloorEstimator` are standalone physics utilities (De Leva center-of-mass,
postural instability index, ADL discriminators). They consume
`world_landmarks` and are unit-tested (`test_biomechanics.py`,
`test_discriminators.py`), but nothing in
`cli.py → runner.py → fall_state.py → fall_fsm.py → fall_evidence.py` calls
into them. Treat them as a physics reference / future-use module, not active
code.

## The one gotcha that will bite you

MediaPipe's `world_landmarks` (metric, hip-centered) are **not** flipped to
a standard y-up convention — `y` increases **downward**, same as normalized
image landmarks. So "vertical"/"up" in any physics code is the `-y`
direction, and the ground plane is `(x, z)`. This is verified against a real
MediaPipe inference and documented in `biomechanics.py:8-11` and
`tests/conftest.py:3-6`. Standard 3D-graphics axis intuition is backwards
here — don't assume it.

Two more concrete ones, since they're easy to trip on:

- **Entry point ordering**: `fall-detection` points at `fall_detection:main`,
  not `cli.py` directly. `main()` (`src/fall_detection/__init__.py`) lazily
  imports `cli.py` inside the function body so mediapipe's native log
  quieting runs before mediapipe is imported anywhere. Don't import `cli`
  (or anything importing `mediapipe`) at module top-level in `__init__.py`.
- **Cascade `z` rescale**: crop-local `z` shares `x`'s scale, so a value
  copied straight out of a crop reads far too deep — `cascade.py` rescales
  it. `world_landmarks` are left alone; metric and hip-centred, they're
  already correct per crop.

## Dev workflow

- Package manager is **uv**: `uv sync` to install, `uv run <cmd>` to run
  anything.
- `uv run pytest` — tests are TDD-developed: write the failing test first,
  watch it fail, then implement.
- `tests/` has no `__init__.py` on purpose (pytest prepend import mode puts
  `tests/` on `sys.path`) — import fixtures with a bare
  `from conftest import ...`, not `from tests.conftest import ...`. Don't add
  `tests/__init__.py`.
- No `print()` anywhere — every module uses
  `logger = logging.getLogger(__name__)`.
- No lint/format/type-check tooling and no CI configured. Don't invent
  commands for these.
- `uv run fall-detection --source 0` to run the CLI against a webcam; see
  README for the full flag table.

## Where to go deeper

| Topic                                                             | Doc                                                         |
| ----------------------------------------------------------------- | ----------------------------------------------------------- |
| Install, run, CLI flags, strategy benchmarks, module layout       | [README.md](README.md)                                      |
| Fall-detection FSM, config profiles, telemetry, replay evaluation | [docs/fall-detection.md](docs/fall-detection.md)            |
| Dev/agent conventions                                             | [CLAUDE.md](CLAUDE.md)                                      |
| Regression replay tooling                                         | `src/fall_detection/evaluation.py`, `evaluation/manifests/` |
