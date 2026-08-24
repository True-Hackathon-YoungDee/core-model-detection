# RGB Temporal-Evidence Fall Detection Specification

## Problem

The current fall detector differentiates MediaPipe pose-world coordinates as
though they were scene coordinates. MediaPipe pose-world landmarks are metric
but are re-centred at the hip midpoint for each observation, so their derived
centre-of-mass velocity and retained ankle "floor" do not represent whole-body
translation or a room floor. This makes the floor vote fail on real prone
subjects and produces a repeating `SLUMPING -> POST_STABILITY_EVALUATION ->
UPRIGHT -> SLUMPING` cycle.

The replacement must use only RGB image evidence, retain the published numeric
`FallState` values, alert for both an observed fall and a person first seen
persistently prone, and preserve alert history after recovery or tracking loss.

## Public state and incident model

The existing enum values are stable API and must not change:

```python
class FallState(IntEnum):
    UPRIGHT = 0
    DESCENDING = 1
    IMPACT = 2
    SLUMPING = 3
    POST_STABILITY_EVALUATION = 4
    FALL_CONFIRMED = 5
    BED_REST = 6
```

`IMPACT` means the peak of combined image evidence. It does not claim that a
physical impact force was measured.

Alerts have a separate semantic type:

```python
class FallAlertKind(StrEnum):
    OBSERVED_FALL = "OBSERVED_FALL"
    PERSISTENT_PRONE = "PERSISTENT_PRONE"
    BED_REST = "BED_REST"

class FallEvidenceLevel(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
```

Each confirmation creates one `FallIncident`. An incident remains in the
manager's incident history when a track disappears or the runtime restarts.
The selected profile's sustained-upright recovery dwell ends the active
incident, returns the FSM to `UPRIGHT`, and records `recovered_at`; it does not
delete the incident. A later fall creates a new incident.

## RGB feature model

All global movement and geometry used for decisions comes from normalized image
landmarks converted to pixels with the exact frame width and height associated
with the observation. `FallStateManager.update` therefore requires positive
`frame_width` and `frame_height`; there is no normalized-coordinate fallback.

For each observation extract:

- torso angle from vertical in pixel-corrected image coordinates;
- bounding-box and hip-centre downward speed;
- absolute torso rotation rate;
- bounding-box height collapse from a rolling upright-height baseline;
- pixel bounding-box width/height ratio;
- image-space motion magnitude plus an explicit availability flag used as
  stillness evidence;
- mean torso landmark visibility and an observation-valid flag;
- normalized torso centroid and matching furniture ROI, if any.

Motion is expressed in body-heights per second. A rolling median of recent
upright bounding-box heights is the preferred scale. Before an upright baseline
exists, the current bounding-box diagonal is the scale and height collapse is
unavailable. Missing or malformed landmarks produce an invalid observation,
not an exception or fabricated zero. First and post-gap motion derivatives are
marked unavailable rather than treated as measured stillness. Adjacent
observations separated by more than 0.5 seconds do not contribute an evidence
interval. Any retained Kalman
prediction is capped at 0.5 seconds and is not used as fresh evidence.

MediaPipe world landmarks may remain available for relative anatomical display
or diagnostics, but floor clearance, global CoM derivatives, kinetic energy,
energy dissipation, and ground-bound decisions are removed from the alert path.
`--body-mass-kg` remains temporarily accepted with a deprecation warning and
has no effect on RGB fall decisions.

## Profiles and configuration

Configuration is TOML. Profile selection precedence is:

1. an explicitly supplied `--fall-profile`;
2. the TOML document's `profile` value;
3. `balanced`.

After selecting profile defaults, explicitly present TOML fields override the
selected profile. Unknown sections/keys, non-finite numbers, invalid ranges,
non-positive dimensions/durations, and malformed polygons fail with a clear
`ValueError`.

The seed profiles are:

| Parameter | sensitive | balanced | precision |
|---|---:|---:|---:|
| dynamic torso angle | 40 deg | 45 deg | 55 deg |
| downward speed | 0.40 BH/s | 0.50 BH/s | 0.70 BH/s |
| torso rotation | 45 deg/s | 60 deg/s | 75 deg/s |
| height collapse | 0.10 | 0.15 | 0.25 |
| posture torso angle | 45 deg | 50 deg | 60 deg |
| posture aspect ratio | 0.90 | 1.00 | 1.20 |
| posture evidence fraction | 0.50 | 0.60 | 0.75 |
| persistent-prone dwell | 1.5 s | 2.0 s | 3.0 s |
| recovery dwell | 0.50 s | 0.70 s | 1.00 s |

All profiles use:

- dynamic cue window: 0.75 s;
- observed-fall postural window: 1.0 s;
- candidate timeout: 2.0 s;
- maximum observation gap: 0.5 s;
- minimum temporal coverage: 0.80;
- rejection cooldown: 0.5 s;
- minimum torso visibility: 0.50;
- recovery torso angle: 35 degrees;
- furniture occupancy fraction: 0.60.

These are transparent seed values for validation, not claims of universal
clinical optimality. The recovery dwell values are empirical regression
calibration: 2.0 seconds could not express the labelled recovery in the
three-second fourth repository clip, while the monotonic 0.50/0.70/1.00-second
seeds retain progressively stricter closure behavior.

Furniture regions are optional normalized polygons. `BED_REST` is reachable
only when a configured polygon contains the torso centroid for the required
fraction of valid postural evidence. With no ROI configured, the system never
claims furniture contact.

## FSM transition contract

Dynamic evidence is temporal rather than a simultaneous Boolean gate:

1. In `UPRIGHT`, latch timestamps for downward motion, rapid torso rotation,
   and height collapse. Enter `DESCENDING` when at least two distinct cues
   occur within 0.75 seconds.
2. Enter `IMPACT` when the configured dynamic torso angle joins the latched
   dynamic cues before the 2.0-second candidate timeout.
3. On the next valid processing step enter `SLUMPING` and accumulate
   duration-weighted postural evidence.
4. Enter `POST_STABILITY_EVALUATION` after a 1.0-second observed-fall window
   has at least 80% observation coverage and meets the profile's strict
   torso-plus-aspect posture fraction, producing `HIGH` evidence. If that
   normal path fails but all three distinct dynamic cues were latched, the
   window still has at least 80% observation coverage, and torso-OR-aspect
   support occupies at least half the strict posture fraction, allow a `MEDIUM`
   observed-fall confirmation. If all three cues were latched but observations
   instead disappear before postural confirmation, allow the same `MEDIUM`
   confirmation at timeout only after at least one valid post-impact sample had
   torso-or-aspect support. Otherwise reject at timeout, clear all candidate
   evidence, and apply the 0.5-second cooldown. The fallback applies only to an
   observed dynamic fall; persistent-prone detection remains strict
   torso-plus-aspect evidence.
5. On the next step, create exactly one alert and enter `FALL_CONFIRMED`, or
   `BED_REST` when furniture occupancy qualifies.
6. Independently, a person with no dynamic transition whose prone posture
   meets the profile fraction and coverage for the persistent-prone dwell goes
   through `SLUMPING` and `POST_STABILITY_EVALUATION`, then emits one
   `PERSISTENT_PRONE` incident.
7. A terminal alert state stays active until the selected profile's sustained
   upright recovery dwell (0.50/0.70/1.00 seconds). Recovery returns the FSM to
   `UPRIGHT` and updates the existing incident rather than erasing it.

The continuous-coverage `MEDIUM` fallback and recovery seeds were calibrated
from committed numerical regression traces, not clip identifiers: clips 1 and
2 retain normal `HIGH` confirmation, while clips 3 and 4 latch all three
dynamic cues but reach only 0.266 and 0.0 strict posture fractions. Lowering
aspect or persistent-posture thresholds would broaden static prone alerts. The
partial-support guard rejects an all-three-cue impulse that immediately returns
upright while preserving the trace-backed fallback. Balanced replay confirms
clips 3 and 4 at 2.166 and 2.200 seconds and closes clip 4 at 2.933 seconds.

No fresh, nonterminal track may remain in a candidate state longer than the
configured timeout plus one transition step. Failed evaluation must not retain
enough static evidence to recreate the old oscillation loop immediately.

## Runtime contract

- Identity expiry uses elapsed seconds, not numbers of inference results, and
  equals candidate timeout plus maximum observation gap.
- A new identity is marked seen on its creation observation.
- Short observation gaps preserve candidate state; invalid intervals do not
  inflate coverage.
- Incidents survive tracker `forget` and manager runtime reset, while unresolved
  active person-ID bindings are detached so a reused ID can create a new
  incident. An explicit `clear_incidents` operation is the only destructive
  incident reset.
- JSONL telemetry records schema version, timestamps, identity, raw features,
  gate results, evidence duration/fraction/coverage, state before/after,
  observation age, and incident event.
- The overlay shows current state, evidence fraction, elapsed/required seconds,
  coverage, incident kind/level, and observation age when debug mode is on.
- Alert JSON remains one JSON object per line and adds a schema version,
  incident identifier, event (`detected` or `recovered`), kind, evidence level,
  and recovery fields while retaining `person_id`, `state`, and `t_seconds`.

CLI additions:

```text
--fall-config PATH
--fall-profile {sensitive,balanced,precision}
--fall-telemetry-log PATH
--fall-debug-overlay
```

## Evaluation and release contract

Commit only numerical feature traces, checksums, manifests, and labels—not
video files or model bundles. The evaluator accepts TOML manifests for local
clips and user-supplied UP-Fall, URFD, and Le2i layouts. Splits are grouped by
subject, trial, and camera so related frames/views never leak across folds.
Local manifests may omit split from every clip; manifests with multiple
declared splits require explicit API/CLI filtering and cannot aggregate
train/test together.

Report event-level sensitivity, precision, false alerts per hour, missed-fall
rate, alert latency, recovery timing, and state dwell. Support replay
comparisons for the legacy strict-AND vote, relaxed OR vote, k-of-N vote, and
the temporal FSM.

Each event label declares `onset_s` and a required `match_end_s` with
`onset_s < match_end_s <= duration_s`. Only a same-kind incident in that exact
inclusive association interval is matched. Overlapping windows use
deterministic maximum-cardinality one-to-one matching; a later incident is a
false positive while the label remains missed. Repository labels use a
two-second post-onset window, clamped to clip duration.

Because image evidence depends on `FallConfig`, extraction records a canonical
exact-config fingerprint (including furniture ROIs) and pose-model SHA-256 in
every clip header. Replay requires consistent per-trace provenance and the same
config fingerprint. Trace schema v2 requires explicit motion availability;
strict schema-v1 traces remain readable with motion conservatively unavailable.
A trace is complete only when observation records cover
every frame index in `0..frame_count-1`; duplicate `(frame_index, person_id)`
keys, inconsistent same-frame times, nonmonotonic frame times, and times outside
the clip duration are invalid.

Repository acceptance fixtures are labelled as follows:

| Clip | Expected incident | Recovery |
|---|---|---|
| `fall_example_1.mp4` | exactly one `OBSERVED_FALL` | absent |
| `fall_example_2.mp4` | exactly one `OBSERVED_FALL` | absent |
| `fall_example_3.mp4` | exactly one `OBSERVED_FALL` | absent |
| `fall_example_4.mp4` | exactly one `OBSERVED_FALL` | present |
| `sample_1.mp4` | no incident | n/a |
| `sample_2.mp4` | no incident | n/a |

No-person samples must create no alerts. Unit/integration coverage must include
rapid sitting, bending, crouching, controlled lying, getting up, variable FPS,
dropped observations, malformed landmarks, identity loss, ROI/no-ROI behavior,
incident preservation, and every transition edge.

The production release gate, evaluated on held-out deployment-like data, is at
least 95% event sensitivity, at most 1 false alert per monitored hour, and
median alert latency at most 2 seconds. The six repository clips are regression
fixtures only and cannot establish those production metrics.

## Primary research sources

- MediaPipe Pose Landmarker coordinate contract:
  <https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/python>
- Chen et al. (2020), torso angle and bounding-box sequence:
  <https://doi.org/10.3390/sym12050744>
- Fan et al. (2017), temporal lying vote followed by immobility:
  <https://doi.org/10.1177/1550147717707418>
- Debard et al. (2011), feature combinations on real home falls:
  <https://doi.org/10.3233/978-1-60750-795-6-441>
- Bosch et al. (2014), calibrated gravity-aware camera geometry:
  <https://doi.org/10.1016/j.eswa.2014.06.045>
- Yang and Tian (2015), calibrated depth/floor-plane clearance:
  <https://doi.org/10.3390/s150923004>
- Martinez-Villasenor et al. (2019), UP-Fall dataset and staged-data limits:
  <https://doi.org/10.3390/s19091988>
- Bagala et al. (2012), staged versus real-fall generalization warning:
  <https://doi.org/10.1371/journal.pone.0037062>
