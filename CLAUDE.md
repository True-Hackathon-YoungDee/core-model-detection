# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Physics-informed fall-detection pipeline on top of MediaPipe Pose (Tasks API, `mediapipe>=1.0.1`). Package: `src/fall_detection/`.

## Docs map

- `README.md` — install, run, CLI flags, strategy benchmarks, onboarding quickstart.
- `docs.md` — guided, file-by-file reading order for a new contributor.
- `docs/fall-detection.md` — the fall-detection operating guide: FSM transitions, config profiles, telemetry, replay evaluation, limitations.

Point edits at whichever of these already owns the topic; don't duplicate content across them.

## Commands

- Package manager is **uv**. Install/sync: `uv sync`. Run anything: `uv run <cmd>`.
- Run tests: `uv run pytest` (testpaths already scoped to `tests/` via pyproject.toml).
- Run the CLI: `uv run fall-detection --source 0 [flags...]` (or a video file path instead of `0`).
- Two console scripts are defined in `pyproject.toml`: `fall-detection` (`fall_detection:main`) and `fall-evaluate` (`fall_detection.evaluation:main`, replay regression — see README's [Replay regression](README.md#replay-regression)).
- `--fall-config` takes a TOML file; `config/fall_detection.example.toml` is the annotated template to copy.
- **No lint, format, or type-check tooling is configured** (no ruff/black/mypy/pre-commit). Don't invent commands for these — there's nothing to run.
- **No CI** exists yet.

## Test conventions

- `tests/` has no `__init__.py` on purpose, so pytest's prepend import mode puts `tests/` on `sys.path`. Test files import shared fixtures with a **bare** import: `from conftest import make_person, standing_pose`, not `from tests.conftest import ...`. Follow this pattern in new test files; do not add `tests/__init__.py` (it would break the existing imports).
- Shared synthetic `PersonPose` fixtures live in `tests/conftest.py` (basic builders) and `tests/synthetic_falls.py` (full fall-sequence generator for orchestrator-level tests).
- This is TDD-developed code: write the failing test first, watch it fail, then implement.

## Key gotcha: world-landmark axis convention

MediaPipe's `world_landmarks` (metric, hip-centered) are **not** flipped to a standard "y-up" convention — `y` increases **downward**, same as the normalized image landmarks. So "vertical"/"up" in any physics code is the `-y` direction, and the ground plane is `(x, z)`. This is verified against a real MediaPipe inference and documented in both `src/fall_detection/biomechanics.py` and `tests/conftest.py` — easy to get backwards, don't assume standard 3D-graphics/physics axis conventions here.

## Entry point ordering

The `fall-detection` console script points at `fall_detection:main` (in `__init__.py`), not directly at `cli.py`. `main()` lazily imports `cli.py` inside the function body so that mediapipe's native log quieting (`logging_config.quiet_native_logs()`) runs *before* `mediapipe` gets imported anywhere. Don't import `cli` (or anything that imports `mediapipe`) at module top-level in `__init__.py`.

## Logging

No `print()` anywhere in the package by design — use `logger = logging.getLogger(__name__)` per module, matching the existing convention in every file.

## CLI output

`--output <path>` writes annotated frames to a video file (file sources only, rejected for live sources). Frame annotation only happens when display or output writer is enabled.

## Offline modules

`biomechanics.py`, `discriminators.py`, and `kalman.py` are unit-tested (`test_biomechanics.py`, `test_discriminators.py`, `test_kalman.py`) but nothing in `cli.py → runner.py → fall_state.py → fall_fsm.py → fall_evidence.py` calls into them — they're a physics reference / future-use layer, not active code on the CLI path. Don't assume an edit there changes runtime behavior, and don't delete them as dead code. `geometry.py` is the one exception reached at runtime: `biomechanics.py` imports its convex-hull helpers, so it isn't fully isolated.
