# Fall detection operating guide

This detector makes event decisions from RGB image geometry over time. Its
thresholds are transparent regression seeds for this implementation, not
clinical cutoffs, medical-device claims, or proof of deployment fitness.

## Why the old coordinate model failed

MediaPipe Pose Landmarker returns normalized image landmarks and world
landmarks. The [official coordinate contract](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python)
defines world coordinates in metres with the hip midpoint as origin. That
origin is re-established for each pose observation; it is not a fixed point in
the room. Differentiating it as scene position therefore cannot measure global
centre-of-mass descent, and retaining an ankle coordinate cannot establish a
room floor. Those assumptions caused the old prone vote to fail and the FSM to
oscillate through slumping, evaluation, and upright states.

The active decision path instead converts normalized landmarks to pixels using
the exact observation width and height. It derives torso angle from pixel
vertical, hip and bounding-box downward speed, torso rotation rate, collapse
against a rolling upright-height median, pixel width/height ratio, image-space
motion, torso visibility, and normalized torso centroid. Speeds use body
heights per second. A gap over 0.5 seconds breaks derivative and coverage
adjacency. `motion_available=false` distinguishes an unavailable first/post-gap
derivative from measured zero motion, so unavailable motion cannot satisfy the
stillness diagnostic or the legacy strict-AND comparator.

## States, transitions, and incidents

`IMPACT` names the peak of combined image evidence; it does not claim a force
measurement. Public numeric state values remain stable.

| From | To | Condition |
| --- | --- | --- |
| `UPRIGHT` | `DESCENDING` | At least two distinct downward-speed, torso-rotation, and height-collapse cues latch within 0.75 s. |
| `UPRIGHT` | `SLUMPING` | With no dynamic candidate, strict torso-plus-aspect posture reaches the persistent-prone dwell and coverage/fraction thresholds. |
| `DESCENDING` | `IMPACT` | Dynamic torso angle joins the latched candidate before its 2.0 s timeout. |
| `DESCENDING` | `UPRIGHT` | Candidate times out; evidence clears and a 0.5 s rejection cooldown starts. |
| `IMPACT` | `SLUMPING` | The next valid observation begins duration-weighted postural accumulation. |
| `SLUMPING` | `POST_STABILITY_EVALUATION` | Normal observed-fall path: 1.0 s window, at least 0.80 coverage, and profile posture fraction qualifies `HIGH`. Empirical fallback: all three dynamic cues, the same covered window, and torso-OR-aspect support for at least half the strict posture fraction qualify `MEDIUM`. Persistent-prone qualification stays strict torso plus aspect and is `HIGH`. |
| candidate state | `POST_STABILITY_EVALUATION` | If observations disappear, all three cues were latched, and at least one valid post-impact sample had torso-or-aspect support, timeout may queue `MEDIUM`; otherwise timeout rejects to `UPRIGHT`. |
| `POST_STABILITY_EVALUATION` | `FALL_CONFIRMED` / `BED_REST` | The next step creates one incident. A qualifying configured furniture ROI produces `BED_REST`; otherwise the observed/persistent kind is retained. |
| terminal state | `UPRIGHT` | Sustained, quality-valid torso angle at or below 35° reaches the profile recovery dwell. |

The normal `HIGH` path is always evaluated before the `MEDIUM` fallback.
Continuous-coverage fallback was selected because local clips 3 and 4 latch all
three dynamic cues but have strict torso-plus-aspect fractions of 0.266 and
0.0. Requiring partial torso-or-aspect support rejects an all-three-cue impulse
that immediately returns upright while preserving those two regressions.
Weakening static posture would also weaken persistent-prone specificity. This
trace-driven rule is a narrower regression calibration, not a universal fall
model.

One terminal confirmation creates one durable `FallIncident`. Losing a track
or resetting runtime trackers does not erase history; only
`clear_incidents()` does. A loss/reset detaches the unresolved person-ID binding
so a reused tracker ID can create a later incident; the unresolved historical
record remains immutable. Recovery adds `recovered_at` to an incident while its
identity remains active. It does not retract or suppress the alert that was
already emitted. A later fall creates a new identifier. Identity retention is
the configured candidate timeout plus maximum observation gap.

## Profiles and configuration

Profile selection precedence is:

1. explicit `--fall-profile`;
2. the TOML document's `profile`;
3. `balanced`.

Explicit TOML fields then override the selected profile. Unknown keys,
non-finite values, invalid fractions/ranges, and non-positive durations fail
with `ValueError`.

| Seed | sensitive | balanced | precision |
| --- | ---: | ---: | ---: |
| dynamic torso angle | 40° | 45° | 55° |
| downward speed | 0.40 BH/s | 0.50 BH/s | 0.70 BH/s |
| torso rotation | 45°/s | 60°/s | 75°/s |
| height collapse | 0.10 | 0.15 | 0.25 |
| posture torso angle | 45° | 50° | 60° |
| posture aspect ratio | 0.90 | 1.00 | 1.20 |
| posture fraction | 0.50 | 0.60 | 0.75 |
| persistent-prone dwell | 1.5 s | 2.0 s | 3.0 s |
| recovery dwell | 0.50 s | 0.70 s | 1.00 s |

The recovery seeds are monotonic empirical calibration. The original 2.0 s
seed was impossible for the labelled fourth three-second clip, which contains
about 1.2 s of final upright evidence. Balanced replay closes that incident at
2.933 s without changing the alert or the strict persistent-prone gate.

Furniture regions are optional normalized image polygons:

```toml
[[furniture_rois]]
name = "bed"
points = [[0.10, 0.25], [0.90, 0.25], [0.90, 0.95], [0.10, 0.95]]
```

Coordinates must be finite and within `[0, 1]`; at least three distinct points
must enclose non-zero area without self-intersection. Boundary points count as
inside. A repeated closing vertex is accepted and normalized away. With no
configured ROI, furniture contact and `BED_REST` are never claimed.

## Telemetry and alert JSONL

`--fall-telemetry-log PATH` writes one schema-v1 decision record per person and
processing step. Each line contains:

- `person_id`, `t_seconds`, `previous_state`, `state`, `state_changed`, and
  `observation_age_s`;
- raw `features`: validity, pixel torso angle/aspect, two downward speeds,
  rotation, collapse, motion, explicit motion availability, visibility,
  centroid, ROI, and scale source;
- Boolean evidence gates for dynamic torso, downward motion, rotation,
  collapse, strict torso/aspect posture, stillness, and quality;
- duration-weighted `evidence_fraction`, `coverage_fraction`, elapsed/required
  seconds, alert kind/level, incident ID, and incident event.

`--fall-alert-log PATH` is the smaller incident stream. A `detected` or
`recovered` record includes schema version, incident ID, original and current
person IDs, terminal/current state, timestamp, kind, evidence level,
`detected_at`, and `recovered_at`. Package code uses `logging`; only the
`fall-evaluate` command writes its JSON report to stdout.

## Trace extraction and replay

The extractor validates source SHA-256 before MediaPipe starts, serializes only
finite numeric/image evidence, and atomically replaces its output. Every clip
header records the source and pose-model SHA-256 plus a canonical fingerprint
of the exact `FallConfig` used to derive features, including profile,
thresholds, and furniture ROIs. Replay loads configuration with production
precedence (explicit profile, then file profile, then `balanced`) and rejects a
different fingerprint with a re-extraction instruction. If an existing trace
names different source checksums, replacement also requires `--force`.
Trace schema v2 requires `motion_available`. The reader still accepts strict
schema-v1 traces and conservatively treats their unlabelled motion derivative
as unavailable; regenerate v1 traces before comparing the stillness-dependent
legacy strategy.
The fingerprint is SHA-256 over compact, key-sorted JSON containing
`schema_version = 1` and every `FallConfig` dataclass field; enum profiles use
their string value and ROIs use ordered names and normalized point pairs.

```bash
uv run python scripts/extract_fall_traces.py \
  --manifest evaluation/manifests/local-falls.toml \
  --output evaluation/traces/local-regression-v2.jsonl \
  --source-root /path/to/repository-root \
  --model-path /path/to/pose_landmarker_full.task \
  --fall-profile balanced --force

uv run fall-evaluate \
  --manifest evaluation/manifests/local-falls.toml \
  --strategy temporal-fsm
```

Four replay strategies are supported, exactly:

- `legacy-and`: all five dynamic/posture votes plus stillness on one sample;
- `relaxed-or`: any dynamic/posture vote on one sample;
- `k-of-n`: at least three of the five votes on one sample;
- `temporal-fsm`: the production latching, duration, gap, evidence-level, and
  recovery behavior.

Reports contain labelled/detected/true-positive/false-positive/missed event
counts, event sensitivity, precision, false alerts per monitored hour, miss
rate, each matched alert latency and its median, detection-to-recovery timing,
and state dwell. Every label has an explicit `match_end_s`; only a same-kind
incident detected in the inclusive interval from `onset_s` through
`match_end_s` is a true positive. A later incident is a false positive and the
unmatched label is missed. Overlapping label windows use deterministic
maximum-cardinality one-to-one matching; no timestamp tolerance widens either
boundary. Replay expires disappeared identities at the same configured age as
the runtime, so state dwell does not extend a stale FSM to clip end. Metrics
are event-level, never frame-level.

The local balanced-profile regression labels 10 `OBSERVED_FALL` clips: 8 are
detected (clips 1, 2, 5, 7, 8, 10 are `HIGH`; clips 3 and 4 are `MEDIUM`, and
only clip 4 recovers), and 2 (clips 6 and 9) are genuine misses — both are
short (~1.9s) staged falls where the labelled event never reaches
`FALL_CONFIRMED` before the clip ends. `tests/test_evaluation.py` pins these
outcomes, including the misses; don't "fix" the test by relabelling those
clips. The committed trace also contains both zero-event `sample_1.mp4` and
`sample_2.mp4` negatives, each with zero emitted incidents. These fixtures
guard behavior; they are too small and too staged to establish production
performance.

`evaluation/manifests/synthetic-adl.toml` is a second, separate replay set:
16 hand-authored `FallFeatures` streams (6 falls, 8 ADL hard negatives such
as fast sit / brief lie-down / bend / squat / kneel / jump / brisk walk /
deliberate floor-sit, and 2 degenerate inputs), generated deterministically
by `scripts/generate_synthetic_traces.py` from
`src/fall_detection/synthetic_traces.py`. These are not recordings of a real
subject — they exist to pin exact FSM behavior against ADL motions the small
real negative set doesn't cover. Its accuracy metrics describe how the FSM
responds to authored inputs, not measured system accuracy; see that module's
docstring and the README "Replay regression" section before quoting them as
the latter.

## Public datasets and leakage-safe splits

Start from `evaluation/manifests/up-fall.example.toml`,
`urfd.example.toml`, or `le2i.example.toml`. Replace every zero checksum and
illustrative path/time with values from your licensed local copy. Each clip
requires a repo-relative source, SHA-256, positive duration, subject, trial,
camera, optional split, and zero or more ordered event labels (`kind`, `onset_s`,
`match_end_s`, and whether recovery is expected). `match_end_s` must be after
the onset and no later than clip duration; the examples use a two-second
post-onset association horizon, clamped to the clip end.

Trace validation requires at least one observation record for every decoded
frame index from zero through `frame_count - 1`. Multiple identities may share
a frame timestamp, but `(frame_index, person_id)` keys must be unique and frame
timestamps must be consistent, nondecreasing, and inside the clip duration.

Subject, trial, and camera keys are mandatory. The loader rejects a subject
assigned to more than one declared split, which keeps all trials, frames, and
camera views of that person on one side of the evaluation boundary. Allocate
subjects first, then place all their trials/views in that split; never split
frames or cameras independently. A local manifest may omit `split` from every
clip. A manifest with multiple declared splits must select one explicitly in
the API or with `--split`; aggregation across train/test is rejected.

For each populated manifest, extract once and compare all strategies against
the same immutable trace:

```bash
for strategy in legacy-and relaxed-or k-of-n temporal-fsm; do
  uv run fall-evaluate --manifest evaluation/manifests/my-dataset.toml \
    --strategy "$strategy" --split test
done
```

Release requires a separate held-out, deployment-like set with at least 95%
event sensitivity, at most 1 false alert per monitored hour, and median alert
latency at most 2 seconds. Tune on training/validation subjects only, freeze
configuration and code, then run the held-out gate once.

## Research basis and limits

The implementation borrows measurable patterns, not universal thresholds:

- [MediaPipe Pose Landmarker documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python)
  defines image/world output coordinates and motivates the pixel-space choice.
- [Chen et al. (2020), DOI 10.3390/sym12050744](https://doi.org/10.3390/sym12050744)
  studies torso angle and bounding-box sequences.
- [Fan et al. (2017), DOI 10.1177/1550147717707418](https://doi.org/10.1177/1550147717707418)
  motivates temporal lying evidence followed by immobility.
- [Debard et al. (2011), DOI 10.3233/978-1-60750-795-6-441](https://doi.org/10.3233/978-1-60750-795-6-441)
  evaluates combinations of features on real home falls.
- [Bosch et al. (2014), DOI 10.1016/j.eswa.2014.06.045](https://doi.org/10.1016/j.eswa.2014.06.045)
  uses calibrated gravity-aware camera geometry; this uncalibrated RGB path
  deliberately does not imitate its floor claims.
- [Yang and Tian (2015), DOI 10.3390/s150923004](https://doi.org/10.3390/s150923004)
  uses calibrated depth/floor-plane clearance, evidence unavailable here.
- [Martinez-Villasenor et al. (2019), DOI 10.3390/s19091988](https://doi.org/10.3390/s19091988)
  introduces UP-Fall and documents staged multimodal data.
- [Bagala et al. (2012), DOI 10.1371/journal.pone.0037062](https://doi.org/10.1371/journal.pone.0037062)
  warns that staged-fall performance does not guarantee real-fall
  generalization.

RGB-only pose estimates can be occluded, perspective-dependent, and wrong.
Furniture occupancy is a configured image region, not physical contact.
`MEDIUM` means a complete dynamic sequence with adequate temporal coverage but
insufficient strict posture support; it is not a confidence probability.
