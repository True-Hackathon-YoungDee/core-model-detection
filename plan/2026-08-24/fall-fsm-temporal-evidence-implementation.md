# RGB Temporal-Evidence Fall Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the invalid hip-centred world-coordinate fall gates with a configurable RGB temporal-evidence FSM that confirms the four labelled fall clips, distinguishes observed falls from persistent prone detections, records recovery, and cannot oscillate indefinitely in `SLUMPING`.

**Architecture:** A validated TOML configuration selects one of three seed profiles. A per-person image-evidence extractor converts normalized landmarks to pixels and derives body-relative motion; the seven-state FSM latches dynamic cues and integrates time-weighted postural evidence. A state manager owns durable incidents independently from disposable identity trackers, while CLI telemetry and an offline replay evaluator expose every decision.

**Tech Stack:** Python 3.13, NumPy, OpenCV, MediaPipe Tasks, standard-library `tomllib`, pytest, uv.

**Spec:** `plan/2026-08-24/fall-fsm-temporal-evidence-spec.md`

## Global Constraints

- Preserve `FallState` numeric values exactly: `UPRIGHT=0`, `DESCENDING=1`, `IMPACT=2`, `SLUMPING=3`, `POST_STABILITY_EVALUATION=4`, `FALL_CONFIRMED=5`, `BED_REST=6`.
- Use RGB image landmarks converted with positive frame width and height for global movement; never use hip-centred MediaPipe world coordinates as room translation, floor clearance, impact force, kinetic energy, or ground contact.
- Alert kinds are exactly `OBSERVED_FALL`, `PERSISTENT_PRONE`, and `BED_REST`; evidence levels are exactly `HIGH` and `MEDIUM`.
- Use elapsed seconds for every dwell/window/timeout; temporal evidence requires at least `0.80` coverage and ignores adjacent gaps over `0.5` seconds.
- CLI profile precedence is explicit `--fall-profile`, then TOML `profile`, then `balanced`; explicit TOML fields override selected profile defaults.
- With no configured furniture polygon, `BED_REST` must be unreachable.
- Incidents survive identity loss and runtime reset; sustained upright recovery updates `recovered_at` and allows a later distinct incident.
- Keep `--body-mass-kg` temporarily accepted, emit a deprecation warning, and do not let it affect decisions.
- Write production changes through strict TDD: focused failing test observed first, minimal implementation, focused green run, full suite before each task commit.
- Use logging rather than `print` in the package. Do not commit videos, downloaded model bundles, generated annotated videos, or external dataset contents.

---

### Task 1: Validated profiles and furniture ROI configuration

**Files:**
- Create: `src/fall_detection/fall_config.py`
- Create: `tests/test_fall_config.py`
- Create: `config/fall_detection.example.toml`

**Interfaces:**
- Produces: `FallProfile`, `FurnitureROI`, `FallConfig`, and `load_fall_config(path: Path | None = None, profile: FallProfile | str | None = None) -> FallConfig`.
- `FurnitureROI.contains(point: tuple[float, float]) -> bool` uses normalized image coordinates and includes boundary points.
- `FallConfig` exposes the exact profile and common values in the specification as validated immutable fields, plus `furniture_rois: tuple[FurnitureROI, ...]`.

- [ ] **Step 1: Write failing profile and precedence tests**

  Add literal assertions for all balanced seed values, all differing sensitive
  and precision values, explicit CLI-profile precedence over a TOML profile,
  and TOML field overrides after profile selection. The change these tests
  catch is a wrong seed value or reversed precedence.

- [ ] **Step 2: Run the focused tests and verify RED**

  Run `uv run pytest tests/test_fall_config.py -q`. Confirm collection fails
  because `fall_detection.fall_config` does not exist.

- [ ] **Step 3: Implement immutable configuration and TOML loading**

  Define the enum and dataclasses, a literal profile-default table, recursive
  known-key validation for `[dynamic]`, `[posture]`, `[timing]`, `[quality]`,
  and `[[furniture_rois]]`, and `dataclasses.replace`-based overrides. Validate
  finite/ranged numbers, polygon points in `[0, 1]`, at least three distinct
  polygon vertices, and unique non-empty ROI names.

- [ ] **Step 4: Add invalid-input and polygon behavior tests**

  Test unknown keys, NaN/Infinity, fractions outside `[0,1]`, zero durations,
  malformed polygons, boundary containment, and an interior/exterior point.

- [ ] **Step 5: Verify GREEN and commit**

  Run `uv run pytest tests/test_fall_config.py -q`, then `uv run pytest -q`.
  Commit `feat: add validated fall detection profiles`.

### Task 2: Pixel-corrected RGB evidence extraction

**Files:**
- Create: `src/fall_detection/fall_evidence.py`
- Create: `tests/test_fall_evidence.py`
- Modify: `src/fall_detection/kalman.py`
- Modify: `tests/test_kalman.py`

**Interfaces:**
- Consumes: `FallConfig` from Task 1 and `PersonPose`/`PoseLandmark`.
- Produces: immutable `FallFeatures`, immutable `FallEvidence`,
  `ImageEvidenceExtractor(config: FallConfig)`,
  `ImageEvidenceExtractor.update(person, t_seconds, frame_width, frame_height) -> FallFeatures`,
  and `classify_evidence(features, config) -> FallEvidence`.
- `FallFeatures` contains `t_seconds`, `valid`, `torso_angle_deg`,
  `bbox_aspect_ratio`, `hip_downward_speed_bh_s`,
  `bbox_downward_speed_bh_s`, `torso_rotation_deg_s`,
  `height_collapse_fraction`, `motion_bh_s`, `visibility_quality`,
  `torso_centroid`, `furniture_roi`, and `scale_source`.
- `FallEvidence` contains the individual dynamic/posture/stillness/quality
  gates and a `dynamic_cue_count` property.

- [ ] **Step 1: Write failing geometry tests**

  Build real `PersonPose` fixtures and assert an upright torso is near zero
  degrees, a horizontal torso is near 90 degrees, and identical normalized
  geometry produces the same pixel-corrected angle/aspect behavior in 16:9,
  9:16, and square frames. Assert non-positive dimensions raise `ValueError`.

- [ ] **Step 2: Verify geometry RED**

  Run `uv run pytest tests/test_fall_evidence.py -q`; confirm the missing
  extractor is the reason for failure.

- [ ] **Step 3: Implement current-frame geometry and quality handling**

  Convert landmark deltas with the supplied dimensions before angles,
  distances, and aspect ratios. Require finite shoulder/hip coordinates and
  minimum visibility for a valid observation. Return an invalid feature object
  for short/malformed landmark lists rather than indexing or throwing.

- [ ] **Step 4: Write failing temporal feature tests**

  Use hand-calculated timestamps and pixel positions to assert body-height
  downward speed, rotation rate, upright rolling-median height collapse,
  diagonal fallback before baseline, no fabricated collapse before baseline,
  gap reset beyond 0.5 seconds, ROI matching, and finite outputs.

- [ ] **Step 5: Implement temporal extraction and evidence gates**

  Keep only the history required by configured windows. Update the upright
  height baseline only for good-quality poses at or below the 35-degree
  recovery angle. Normalize motion by baseline height or current diagonal and
  classify the three dynamic cues plus posture/stillness gates.

- [ ] **Step 6: Cap prediction-only Kalman output**

  First add a failing test showing an invisible landmark can be predicted at
  0.4 seconds but not at 0.6 seconds. Add a `max_prediction_gap_s=0.5`
  constructor option and make stale prediction return unavailable data without
  treating it as a fresh observation.

- [ ] **Step 7: Verify GREEN and commit**

  Run `uv run pytest tests/test_fall_evidence.py tests/test_kalman.py -q`, then
  `uv run pytest -q`. Commit `feat: extract RGB temporal fall evidence`.

### Task 3: Temporal seven-state FSM and recovery decisions

**Files:**
- Rewrite: `src/fall_detection/fall_fsm.py`
- Rewrite: `tests/test_fall_fsm.py`

**Interfaces:**
- Consumes: `FallConfig`, `FallFeatures`, and `FallEvidence` from Tasks 1-2.
- Produces: stable `FallState`, `FallAlertKind`, `FallEvidenceLevel`, immutable
  `FallDecision`, and `PersonFallFSM(config: FallConfig)`.
- `PersonFallFSM.step(features: FallFeatures) -> FallDecision` is the only
  observed-frame transition entry point.
- `PersonFallFSM.observe_gap(t_seconds: float) -> FallDecision` advances
  timeouts without inventing a valid pose.
- `FallDecision` exposes `state`, `previous_state`, `state_changed`, `evidence`,
  `evidence_fraction`, `coverage_fraction`, `evidence_elapsed_s`,
  `evidence_required_s`, `alert_kind`, `evidence_level`, and `recovered`.

- [ ] **Step 1: Preserve the public enum and write dynamic-latching RED tests**

  Assert the exact integer enum mapping. Feed literal feature sequences proving
  one cue does not leave `UPRIGHT`, two distinct cues inside 0.75 seconds enter
  `DESCENDING`, stale cues do not combine, and dynamic torso angle advances to
  `IMPACT`. Confirm `IMPACT` means evidence peak in its docstring/API.

- [ ] **Step 2: Implement cue latching and candidate timeout**

  Store one recent timestamp per dynamic cue, prune by the dynamic window, and
  clear all latches on rejection/reset. Use explicit `None` checks so a valid
  timestamp of `0.0` works. Apply the 0.5-second rejection cooldown.

- [ ] **Step 3: Write time-weighted postural evidence RED tests**

  Test identical behavior at 5, 15, and 30 observations/second; exact 80%
  coverage; a gap over 0.5 seconds that contributes no coverage; exact profile
  fraction boundaries; observed-fall high confirmation; all-three-cues plus
  lost-observation medium confirmation; and rejection at timeout.

- [ ] **Step 4: Implement observed-fall progression**

  Make `IMPACT` transition to `SLUMPING`, integrate piecewise-constant valid
  observation intervals, advance through `POST_STABILITY_EVALUATION`, and emit
  exactly one alert decision on the following step. Never use frame counts.

- [ ] **Step 5: Write persistent-prone, ROI, loop, and recovery RED tests**

  Assert a prone-from-first-observation subject emits `PERSISTENT_PRONE` after
  the configured dwell; a brief prone pose does not; a qualifying ROI emits
  `BED_REST`; no ROI cannot; failed evaluation remains `UPRIGHT` through
  cooldown instead of re-entering the old 30/1/15 oscillation; terminal state
  records a recovery decision only after two upright seconds; and a later fall
  can create a new alert decision.

- [ ] **Step 6: Implement persistent posture, furniture, and recovery paths**

  Use the same duration-weighted accumulator with the profile's persistent
  dwell. Track furniture-positive duration only inside valid posture evidence.
  Keep terminal state active while accumulating upright recovery intervals and
  return to `UPRIGHT` without destroying the prior alert decision.

- [ ] **Step 7: Verify GREEN and commit**

  Run `uv run pytest tests/test_fall_fsm.py -q`, then `uv run pytest -q`.
  Commit `feat: replace frame voting with temporal fall FSM`.

### Task 4: Durable incidents, safe manager integration, and time-based identity expiry

**Files:**
- Rewrite: `src/fall_detection/fall_state.py`
- Modify: `src/fall_detection/tracking.py`
- Rewrite: `tests/test_fall_state.py`
- Modify: `tests/test_runner_hooks.py`
- Create: `tests/test_tracking_timeouts.py`

**Interfaces:**
- Consumes: extractor/FSM/config from Tasks 1-3.
- Produces: immutable `FallIncident`, expanded immutable `FallEvent`, and
  `FallStateManager(config: FallConfig | None = None, body_mass_kg: float | None = None)`.
- `FallStateManager.update(persons, t_seconds, frame_width, frame_height) -> list[FallEvent]`
  requires positive integer dimensions.
- `FallStateManager.incidents -> tuple[FallIncident, ...]` is chronological;
  `forget(person_id)` discards only transient tracking state;
  `reset(preserve_incidents: bool = True)` preserves incidents by default;
  `clear_incidents()` explicitly removes them.
- `FallEvent.incident_event` is `"detected"`, `"recovered"`, or `None`.
- `IdentityTracker` expires by `max_unseen_s` and never increments a newly
  created track as missed on its creation observation.

- [ ] **Step 1: Write manager validation and malformed-input RED tests**

  Assert omitted/non-positive dimensions fail, empty/short world landmark lists
  are irrelevant, short image landmark lists return a safe invalid event, and
  all decisions use pixel geometry.

- [ ] **Step 2: Implement the RGB-only per-person manager path**

  Remove `FloorEstimator`, centre-of-mass derivatives, kinetic energy, energy
  dissipation, and ground-bound calls from the decision path. Retain only
  diagnostics that do not affect decisions. Tick absent known trackers through
  `observe_gap` so timeouts and medium-evidence rules can complete safely.

- [ ] **Step 3: Write durable-incident RED tests**

  Drive an observed fall and a persistent-prone event, assert unique stable
  incident IDs and exact kinds/levels, verify only one detected event per
  incident, call `forget` and default `reset` and assert history survives, then
  drive upright recovery and assert `recovered_at` plus one recovery event.

- [ ] **Step 4: Implement incident ownership in the manager**

  Create incidents from FSM alert decisions, keep active/history maps separate
  from trackers, replace immutable incidents on recovery, and prevent duplicate
  creation while an incident is active. Use an incrementing manager-local
  sequence in IDs so later falls are distinct.

- [ ] **Step 5: Write and implement wall-clock identity expiry tests**

  Assert a track survives many result calls within `max_unseen_s`, expires on
  the first call beyond elapsed timeout, a creation observation has zero missed
  age, and `on_lost` fires once. Adapt runner hook tests to the seconds API.

- [ ] **Step 6: Verify GREEN and commit**

  Run `uv run pytest tests/test_fall_state.py tests/test_tracking_timeouts.py tests/test_runner_hooks.py -q`,
  then `uv run pytest -q`. Commit `feat: preserve fall incidents across tracking loss`.

### Task 5: CLI configuration, versioned telemetry, alerts, and debug overlay

**Files:**
- Create: `src/fall_detection/fall_telemetry.py`
- Create: `tests/test_fall_telemetry.py`
- Modify: `src/fall_detection/cli.py`
- Modify: `src/fall_detection/drawing.py`
- Modify: `tests/test_cli.py`
- Rewrite: `tests/test_cli_alerts.py`
- Modify: `tests/test_drawing.py`

**Interfaces:**
- Consumes: `load_fall_config`, `FallStateManager`, and expanded `FallEvent`.
- Produces: `event_record(event, event_type) -> dict[str, object]`,
  `telemetry_record(event) -> dict[str, object]`, and append-only UTF-8 JSONL
  writing with `schema_version: 1`.
- CLI adds `--fall-config`, `--fall-profile`, `--fall-telemetry-log`, and
  `--fall-debug-overlay` exactly.

- [ ] **Step 1: Write CLI parsing/configuration RED tests**

  Assert all new options parse, explicit profile reaches `load_fall_config`,
  invalid TOML exits with code 2 and a useful log message, positive frame
  dimensions reach the manager in file and live callbacks, and body mass emits
  one deprecation warning without changing constructed fall configuration.

- [ ] **Step 2: Implement CLI configuration wiring**

  Load and validate configuration before runner construction, instantiate the
  manager with it, and keep file handles managed for the run rather than opening
  an alert file separately for every event.

- [ ] **Step 3: Write telemetry and alert-schema RED tests**

  Assert literal dictionaries for detected and recovered incident lines and a
  telemetry line containing raw feature fields, individual gates, evidence
  fraction/coverage/timing, state before/after, identity, observation age, and
  incident event. Assert JSON encoding rejects non-finite values.

- [ ] **Step 4: Implement JSONL serialization and lifecycle**

  Emit telemetry for every manager event and alert records only for detected or
  recovered incident events. Flush each incident line so a crash cannot hide an
  alert. Preserve the legacy `person_id`, `state`, and `t_seconds` keys.

- [ ] **Step 5: Write and implement debug-overlay tests**

  Assert non-debug output remains the compact person/state label. In debug mode
  render incident kind/level, evidence percent, elapsed/required seconds,
  coverage percent, and stale observation age; update stale age from monotonic
  wall time between live inference results.

- [ ] **Step 6: Verify GREEN and commit**

  Run `uv run pytest tests/test_cli.py tests/test_cli_alerts.py tests/test_fall_telemetry.py tests/test_drawing.py -q`,
  then `uv run pytest -q`. Commit `feat: expose fall profiles and decision telemetry`.

### Task 6: Replay evaluation, labelled clip regression, and operator documentation

**Files:**
- Create: `src/fall_detection/evaluation.py`
- Create: `tests/test_evaluation.py`
- Create: `evaluation/manifests/local-falls.toml`
- Create: `evaluation/manifests/up-fall.example.toml`
- Create: `evaluation/manifests/urfd.example.toml`
- Create: `evaluation/manifests/le2i.example.toml`
- Create: `scripts/extract_fall_traces.py`
- Create: `evaluation/traces/local-falls-v1.jsonl`
- Create: `docs/fall-detection.md`
- Modify: `README.md`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: serialized `FallFeatures`, `FallConfig`, and `PersonFallFSM`.
- Produces: `load_manifest(path)`, `replay_trace(path, strategy, config)`, and
  `evaluate_manifest(path, strategy, config)` plus a command-line entry in
  `pyproject.toml` named `fall-evaluate`.
- Supported strategy names are exactly `legacy-and`, `relaxed-or`, `k-of-n`,
  and `temporal-fsm`.
- Metrics contain event sensitivity, precision, false alerts/hour, miss rate,
  per-event alert latency, median latency, recovery timing, and state dwell.

- [ ] **Step 1: Write manifest and metric RED tests**

  Use tiny literal trace fixtures to prove subject/trial/camera group keys are
  mandatory, related groups cannot appear in multiple declared splits, missing
  files/checksum mismatches fail clearly, and hand-calculated event counts,
  false-alert hours, latency median, recovery timing, and dwell totals match.

- [ ] **Step 2: Implement trace replay and event-level metrics**

  Parse strict TOML manifests, validate SHA-256, replay timestamps through the
  four named strategies, and aggregate by labelled event rather than frame.
  Emit JSON to stdout only from the command-line wrapper; package code logs
  diagnostics through `logging`.

- [ ] **Step 3: Build deterministic local trace extraction**

  Make the script run MediaPipe over a manifest clip, serialize only finite
  numeric/image evidence plus source checksum and extractor schema version, and
  refuse to overwrite a trace whose source checksum differs unless an explicit
  `--force` is supplied. Run it against the four local fall clips using the
  full model and commit the resulting numerical JSONL, not the MP4s.

- [ ] **Step 4: Add labelled clip acceptance tests and tune only from traces**

  Replay `local-falls-v1.jsonl` with the balanced profile and assert exactly one
  `OBSERVED_FALL` for each of clips 1-4, no recovery for clips 1-3, and recovery
  for clip 4. Run the no-person sample videos through extraction and assert zero
  incidents. Record any seed adjustment in the spec and keep all three profile
  orderings monotonic; do not introduce clip-name branches.

- [ ] **Step 5: Document operation and research basis**

  Explain the hip-centred coordinate root cause, every FSM transition, profile
  precedence, ROI polygon format, telemetry fields, incident/recovery semantics,
  public-dataset manifest population, leakage-safe splitting, benchmark commands,
  and the held-out release gate. Cite the official MediaPipe documentation and
  the primary-source DOIs listed in the specification research notes.

- [ ] **Step 6: Verify complete branch and commit**

  Run `uv run pytest -q`, `uv run fall-evaluate --manifest evaluation/manifests/local-falls.toml --strategy temporal-fsm`,
  and `git diff --check`. Confirm no nonterminal fresh-observation trace exceeds
  the configured timeout and no video/model file is staged. Commit
  `feat: add fall replay evaluation and operating guide`.

