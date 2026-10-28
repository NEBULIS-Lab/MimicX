# Portable SONIC Motion Study

This runner prepares and evaluates a released SONIC controller against audited
motion-study inputs. It does not train, download, select checkpoints, repair
references, or import the historical launchers. A successful diagnostic pilot
does not establish a study-level success rate or a matched comparative result.

## Configuration First

Use an existing Python environment with NumPy and MuJoCo for preparation/scoring.
Simulation additionally needs the installed upstream `gear_sonic`, PyYAML,
Unitree SDK, CycloneDDS, matching deploy executable, observation configuration,
encoder/decoder and their runtime libraries. No optional simulator is imported
by help, command inspection or manifest selection.

Create a private runtime JSON. Every path resolves relative to that JSON, not the
current directory. Paths below are examples, not an installation prescription:

```json
{
  "sonic_root": "third_party/GR00T-WholeBodyControl",
  "policy_parameters": "third_party/GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp",
  "canonical_model": "tracking/src/assets/robots/unitree_g1/xmls/g1.xml",
  "simulator_config": "third_party/GR00T-WholeBodyControl/gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml",
  "simulator_model": "third_party/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml",
  "deploy_binary": "third_party/GR00T-WholeBodyControl/gear_sonic_deploy/target/release/g1_deploy_onnx_ref",
  "decoder": "third_party/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx",
  "encoder": "third_party/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx",
  "observation_config": "third_party/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/observation_config.yaml",
  "sim_python": "third_party/GR00T-WholeBodyControl/.venv_sim/bin/python",
  "environment": {},
  "interface": "lo",
  "dds_domain": 0,
  "dds_isolation": "parent_serialized_domain_zero",
  "sim_frequency": 200,
  "startup_timeout_s": 330,
  "startup_mode": "reference_phase_zero",
  "stand_settle_s": 0,
  "release_settle_s": 0,
  "reference_mode": "reference_to_deployment_fk",
  "protocol_file": "protocol.json",
  "protocol_sha256": "fa10de204d7be9f9bc3fdc12c16ecb7ee7eae4c7b08babc56d4ceeba7e227804",
  "protocol": {
    "root_height_m": 0.25,
    "relative_ee_height_m": 0.25,
    "root_orientation_rad": 1.0,
    "persistence_frames": 5,
    "critical_bodies": ["left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link"],
    "common_prefix_seconds": 1.0,
    "clock_tolerance_s": 0.03,
    "state_gap_s": 0.045
  }
}
```

The public Apache-2.0 module `mimicx/experiments/sonic_helpers.py` vendors the two
lightweight corrected release/continuous-recording helpers, with upstream
attribution. No private corrected-source checkout is needed. A per-run
`code/entry.py` invokes frozen copies of the runner and these helpers. Optional
`corrected_source` can identify a historical provenance directory containing
`run_sonic_sim_record.py`, `direct_policy_comparison.py`, `sonic_release.py` or
`sonic_recording.py`; existing files are copied as evidence only, never executed
or imported. Original/archived sources are never modified. Upstream robot assets
and deploy code still come from the separately licensed official
[GR00T-WholeBodyControl repository](https://github.com/NVlabs/GR00T-WholeBodyControl).
`protocol_file` contains the complete prospectively frozen common protocol, not
just the abbreviated runtime threshold block. Its canonical sorted compact JSON
hash must equal `protocol_sha256`; the eight runtime thresholds/body/clock keys
must agree exactly. With a protocol-bearing manifest, its full object and every
task's protocol ID/hash are also checked. A prior numerical-audit manifest can
instead bind the explicit sidecar without changing its inputs or eligibility.
Preparation stores the complete protocol and hash in every run; threshold drift
or unsupported alignment/persistence semantics fail closed.

Supply library search paths explicitly through `environment`, for example
`LD_LIBRARY_PATH`. The upstream environment-setup shell script is not sourced:
it can perform downloads and unrelated setup. The caller's device visibility is
preserved; runtime JSON cannot override `CUDA_VISIBLE_DEVICES` or `PYTHONPATH`.
ONNX files are copied into each run's `weights/` directory. The deploy binary
writes its GPU-specific TensorRT engine caches beside those copies, never beside
shared original weights. Cold engine compilation is part of startup cost; set a
prospective startup timeout adequate for that environment.

The released binary initializes DDS **logical domain 0**. Setting a domain
environment variable alone does not change that. A parent can arrange a dedicated
network namespace and set `dds_isolation: "dedicated_network_namespace"`, selecting
its interface and any reviewed CycloneDDS settings through `environment`.
Namespace creation and transport isolation verification are external prerequisites.
Alternatively, `parent_serialized_domain_zero` explicitly means the parent has
checked that no other controller/simulator shares this transport. The per-run
duplicate guard does not serialize different experiments or establish isolation.
Do not launch concurrent jobs on the same unisolated domain/interface. No
transport isolation has been demonstrated by the CPU tests.

## CPU Preparation

```bash
python scripts/experiments/sonic_study.py preflight \
  --runtime /path/to/runtime.json --output /path/to/new-preflight.json
```

This inspects `ldd` and imports CPU numerical packages in the configured Python,
with GPU visibility disabled. It never executes the deploy binary. Missing
libraries and symbol-version errors, including Ubuntu 22.04 GLIBC incompatibility,
produce a blocked report and exit 2. Passing only means CPU preflight passed;
TensorRT/CUDA inference and transport remain unverified. Prepare at the actual
destination after transfer. A remote loader failure does not invalidate an
independently verified local environment; use a separate local run directory.

Create an explicit perturbation JSON. `{"mode": "nominal"}` means **zero noise**;
it is appropriate for an initialization diagnostic, not the study's matched
nominal cohort. The paired study distributions are:

| Cohort | Seeds | Physical joint-position noise | Root linear-velocity delta |
| --- | --- | --- | --- |
| Development selection | 1001, 1002, 1003 | independent uniform +/-0.005 rad | 0 |
| Final nominal holdouts | 2001, 2002, 2003 | independent uniform +/-0.005 rad | 0 |
| Final perturbed holdouts | 3001, 3002, 3003 | independent uniform +/-0.01 rad | 0 |

For selection/final nominal use:

```json
{"mode": "release_state_uniform", "joint_position_rad": 0.005, "root_velocity_m_s": 0.0}
```

For final perturbed holdouts use:

```json
{"mode": "release_state_uniform", "joint_position_rad": 0.01, "root_velocity_m_s": 0.0}
```

These distributions act on the same immutable phase-zero physical state for both
controllers. Reusing a seed with `mode: nominal` does not reproduce the native
study's noisy nominal initialization. Check recorded realized deltas as well as
seed/distribution labels when pairing controllers. Nonzero root-velocity noise
is supported for a separately declared study, but is not part of this protocol.
SONIC records the NumPy PCG64 generator and realized deltas against all 29 joint
names, independent of source array storage order. Matching seed labels across
different simulator RNG implementations is not a claim of bit-identical noise
draws; paired distributions, named application and physical initialization must
agree. Exact common-random-number coupling requires separately supplying and
auditing the same realized named vectors for both controllers.

```bash
python scripts/experiments/sonic_study.py prepare \
  --manifest /path/to/data/manifest.json --runtime /path/to/runtime.json \
  --output /path/to/new-study-output --task AUDITED_TASK_ID \
  --seed 1001 --perturbation /path/to/perturbation.json
```

Repeat `--task` for an explicit cohort, or omit it to request every manifest task.
No task names or four-task restriction are built in. Every selected task must
have both quality fields exactly `passed`, a matching prepared-motion SHA-256,
unique safe identity and a contained manifest-relative path. Blocked tasks are
not silently dropped; the full source cohort and statuses remain in preparation
records. An explicitly selected eligible pilot can proceed independently.
Use a new output root for each seed/protocol/attempt; existing runs are retained.

Before producing any executable run records, preparation validates the selected
cohort's named 29-joint/30-body references at 50 Hz and checks nine spread-out
frames against canonical FK. Position error over **2 mm**, rotation error over
0.002 rad, nonfinite arrays, joint-limit excess and malformed names fail closed.
`reference_to_deployment_fk` then derives body control targets with the released
deployment asset at exactly the same named joints, root pose, velocity and time.
Its within-asset FK roundtrip uses the same 2 mm/0.002 rad gates. These are not
cross-XML body-origin equality checks. No fitted offsets, heading fits, retiming,
joint/root repair, PD change or physical-model change is applied. Deployment
requires one unconstrained floating base and no equality/mocap support.
The simulator repeats the deployment FK audit in its own numerical environment
before opening DDS; scoring also requires that runtime audit and states-only,
no-reset, matching-timestep evidence.

Upstream joint permutations are parsed and checked as inverse bijections from
the configured policy-parameters source. CSV joints use its IsaacLab order;
state/target NPZs carry explicit hardware-order joint names and all 30 canonical
body names. Only the controller CSVs use the released 14-body subset; common
world-body, canonical local-FK and velocity metrics always use all 30 bodies for
both native controllers and SONIC. A 14-body-only common metric is rejected.
The released metadata body
IDs are paired with their names, not mistaken for CSV column indices.
Reference duration is `(frames-1)/50`, including the final sample.

`preflight.json` records feasibility. `preparation.json` records passed/blocked
input audits, including per-body deployment FK discrepancies. A blocked FK gate
creates no executable run records. A loader-blocked environment can still have
CPU-prepared inputs, but `evaluate` rechecks and refuses it before launch.

### Asset Frame Bridge

A read-only CPU audit of a passed walk-start input found a 10.208 mm maximum
`torso_link` discrepancy between the canonical tracking asset and the released
43-DoF SONIC scene. Pelvis/leg bodies agreed; arm-body discrepancies were at most
1.892 mm. The XMLs have different waist/torso/shoulder offsets; the discrepancy
was posture-dependent in the checked frames, not a verified constant local
translation. This is an explicit asset-frame difference, not a reason to repair
the immutable joint/root motion or silently relax a within-asset gate.

The approved bridge preserves `motion.npz` bytes and canonical `target.npz`, and
writes separate deployment-FK `control_target.npz` and 14-body control CSVs.
Every run records `reference_to_deployment_fk`, original motion identity,
canonical/deployment model hashes, protocol hash, per-body origin differences,
unchanged root/joint/time declarations and unchanged controller/physics. The
simulator repeats its within-deployment-asset gate against `control_target.npz`.
Cross-asset origin differences are reported diagnostics, not failed mappings.

For common scoring, **both** observed and reference trajectories are recomputed
using the same canonical asset from their measured/immutable pelvis pose, 29
named joints, joint velocities and root world velocities. All 30 canonical body
positions, orientations and origin velocities, including the failure predicates,
use that representation. Raw deployment body frames are retained as evidence
but cannot change the common metric simply by having a different torso origin.

## Evaluate And Inspect

Each prepared task/seed has `run.json`. Inspect exact arguments before execution:

```bash
python scripts/experiments/sonic_study.py commands \
  --run /path/to/new-study-output/AUDITED_TASK_ID/seed_1001/run.json
python scripts/experiments/sonic_study.py evaluate \
  --run /path/to/new-study-output/AUDITED_TASK_ID/seed_1001/run.json
```

The caller arranges resource/process placement and transport isolation. Ordinary
inherited visibility such as `CUDA_VISIBLE_DEVICES=0` is supported; session
management is optional and no session-management executable is called. This
command acquires no resources, starts no remote connection, and never changes
visibility. `attempt.json` records coordinator and child PIDs, raw inherited
device visibility, optional nullable session context, network namespace, commands, logs,
elapsed wall time and exit/error. Device compute time/energy remain null without
an external collector. No elapsed-allocation time is labelled GPU compute time.

### Phase-Zero Initialization

The default `reference_phase_zero` startup waits for the released SDK's `Init Done`
homing completion **without entering CONTROL**. It then disables startup support,
clears only that band's residual wrench, and initializes physical state exactly
once: immutable reference frame-zero pelvis position/quaternion, all 29 named
joint positions/velocities, and world pelvis-link linear/angular velocities,
plus the declared seeded position/linear-velocity noise. MuJoCo's free-joint
angular velocity is converted from world to root-local coordinates. Extra hand
positions are retained; their initial velocities are zero. `mj_forward` updates
derived quantities; no physics step or simulation clock change is inserted.
There is no reset/teleport after CONTROL or during scoring. Released gains,
policy, collision geometry and physical model remain unchanged.

DDS receipt is demonstrated before CONTROL, not assumed after a fixed sleep.
The simulator temporarily changes the `mode_machine` LowState header, observes
the released SDK echo it in LowCmd, restores the original header, and requires a
second echo with unchanged motor q/dq/kp/kd/torque commands. The SDK stores the
complete LowState before echoing that header; the simulated bridge does not use
the header in its torque calculation. This transport-only handshake has a fixed
timeout and is not a policy/PD intervention. Release ACK is published only after
the restored echo and records initialization, noise, receipt and physical clocks.
It must report zero band, generalized external force, applied wrench and root
constraints. Physical integration continues throughout, including the handshake.

After ACK the runner immediately sends official `I]` (heading reinitialization
and CONTROL), waits for the first CONTROL heading log, then immediately sends
`T`. The new controller's history contains no earlier suspended CONTROL frames.
There is no three-second unsupported hold of a dynamic reference frame. ACK to
CONTROL request is limited to 30 ms, initialization to first CONTROL to 250 ms,
and first CONTROL to playback request to 30 ms. Actual first CONTROL joint/base
and torso IMU logs must agree with measured post-reset physics; stale state or
heading fails closed. `startup_events.json`, `control_request.json`, release ACK,
`play_request.json` and `startup_audit.json` preserve causal clocks and the first
scored state. The separate startup contract/hash binds this sequence, source,
assets, weights/configuration, declared noise and unchanged common protocol.

`legacy_paused_reference` is an explicit historical diagnostic mode, with its
former settling delays. Such runs report initialization incompatibility and
must not enter the phase-zero comparative cohort. Historical results and frozen
attempts are not rewritten when the new startup contract is adopted.

Only the runner's `mode: nominal` has zero deltas; the paired study's **nominal
cohort** uses the nonzero joint-noise distribution above. Changing a zero-noise
diagnostic's seed does not create a matched repeat or robustness experiment.
Perturbations never change references, gains or physics.

Recording is states-only, at 50 Hz, with native physics substeps and actuator
torques. The no-reset fall monitor replaces the interactive reset path. All
post-release physics steps reject external forces/support. Only measured
generalized states are copied on the live clock. All 30 post-integration body
transforms and velocities are calculated from those snapshots **after capture**,
with no additional physics steps and with the subtree-COM velocity correction.
This keeps FK/graphics work off the control clock. No graphics context or MP4 is created. Measured clocks,
named joints/bodies, support and native fall telemetry are preserved; raw physical
states remain available for later independently audited rendering.
For earlier recordings containing only 14 body channels but complete physical
`qpos`/`qvel`, scoring can reconstruct all 30 body transforms and origin velocities
from the configured deployment scene. Named joints must agree with that physical
state. Reconstruction is recorded explicitly; it never uses targets or fitted
offsets. Missing physical state fails closed rather than silently scoring 14.

Failed startup/playback/instrumentation attempts retain their logs and cost
record. Existing attempts cannot be overwritten or silently resumed. The runner
stops and waits for both owned child process groups on exit; it does not stop
external processes. `evaluated` means the physical recording passed protocol
audits, not that the controller completed the motion successfully.

## Clock And Common Metrics

`score-recording` is CPU-only and re-audits saved evidence without inference:

```bash
python scripts/experiments/sonic_study.py score-recording \
  --run /path/to/new-study-output/AUDITED_TASK_ID/seed_1001/run.json
```

It refuses an existing result. It needs the run's physical state NPZ, synchronized
controller `q.csv`, `base_quat.csv`, `torso_quat.csv` and `motion_playing.csv`,
target log, release ACK and startup/control/playback evidence. Controller row
IDs/timestamps must agree. Exactly one uninterrupted
playing span must contain every reference frame; all 36 logged root/joint target
columns must match the frozen input. State-logger joints are already restored to
hardware order and offsets; they are not normalized a second time.
The SDK target CSV's single empty trailing field is treated as CSV syntax, not
as a 37th data channel; any extra numerical channel or nonfinite value is rejected.

Onset is the first measured playing-event timestamp. Scoring uses
`onset + arange(frames)/50`, not error minimization or per-tick time warping.
The physical simulator must bracket the full grid and measured event span.
Endpoint holds, truncation, reset clocks and missing samples are rejected.
The 45 ms sample-gap gate covers physical reset/release through the scored
endpoint, including both interpolation brackets. Cold-start gaps before that
causal interval remain reported diagnostics, not failures of a later correctly
captured trajectory. Timestamps must still be finite and strictly increasing
throughout the original capture; this scope rule never repairs, drops or retimes
samples. Gaps during reset/receipt, playback or endpoint brackets fail closed.
Controller and simulation clock ratios must be within 0.97-1.03 and local drift
within the frozen tolerance. Named simulator/controller joint agreement must
have RMSE at most 0.1 rad. Release ordering and force-free interpolation brackets
are checked. A startup fall after release counts as a failure, not a reason to
restart and select another rollout.

Outputs are `clock_audit.json`, `paired_states.npz`, and evidence-hashed
`result.json`. The common metric vector contains world-root, heading,
world-body, canonical identity-root local FK, joint RMSE, critical end-effector,
joint-velocity, body-linear-velocity and body-angular-velocity errors. Each has
time mean and time p95 of its per-frame statistic. Full-horizon errors retain
post-failure physical frames; a prospectively configured common prefix is
reported separately, as are pre-failure diagnostics. Completion and normalized
first-failure horizon cannot improve by censoring difficult frames.
Zero adaptation updates refer to this evaluation, not to SONIC's pretraining cost.

The frozen failure predicate is absolute root-height error, maximum end-effector
height error relative to its root, or root quaternion geodesic error, exceeding
their thresholds for the configured consecutive frames. Root-height is relative
to the reference, not a universal standing-height cutoff. The separate native
fall flag is diagnostic during playback. Ground-contact tasks still require the
input agent's shared semantic audit; this runner does not override blocked inputs.
Contact-conditioned slip remains explicitly null/unsupported: raw physical
contact labels alone do not supply a common reference support-phase definition.

`common_metrics(actual, target, *, model_xml, joint_names, body_names, protocol,
reference_frames=None, native_failure_frame=None, valid_mask=None)` is reusable
by the native study. Inputs are the named fixed-clock arrays used above, and
targets must come from the original world-coordinate motion, not a controller's
yaw/XY-aligned reward-space targets. An explicit frame vector must be phase-zero
`arange(N)`. Native failure is incorporated in the first-failure outcome. A
validity mask must be a contiguous prefix; post-reset rows cannot reenter scoring.
When a validity tail is missing, full-horizon errors are null, and a common-prefix
error is available only if that entire prefix was physically observed. Missing
tails are never padded or labelled complete.

## Reproducibility And Tests

Prepared runs freeze the runner and corrected helper sources, target arrays,
motion bytes, reference CSVs and hashes. Configured models, their XML includes and
file-backed assets, weights, observation/parameter/config files, numerical
interpreter and simulator Python sources are hash-checked again before use.
Changes require new preparation. Runtime JSON and generated runs can contain
private paths; keep them outside source control. Neither proprietary weights nor
licensed input data are bundled or granted redistribution rights here.
The preparation index hashes every run contract. CLI evaluation, scoring and
command inspection delegate to the frozen per-run code, so later checkout edits
do not silently change an existing run's orchestration or metric definitions.

An explicit CPU-only rescoring command uses the **current** scorer and writes to
a new directory outside the immutable capture:

```bash
python scripts/experiments/sonic_study.py rescore-recording \
  --run /path/to/captured-task/seed_1001/run.json \
  --output /path/to/new-rescoring-directory \
  --reason "Approved scorer-only sampling-gap scope correction"
```

This is not another execution or an implicit replacement of the original
result/failed attempt. `rescoring.json` binds the reason, original attempt,
complete capture hashes, old frozen scorer, new copied scorer and unchanged
protocol. Source hashes are checked before and after scoring. Raw states,
clocks, references, controller behavior and old outputs are never rewritten;
no simulator, deploy process or DDS participant is started. Existing rescoring
destinations and active/unfinished physical attempts are rejected. A zero-noise
pilot remains a diagnostic after rescoring, not a formal noisy nominal repeat.

```bash
python -m unittest discover -s tests -p test_sonic_study.py -v
```

The CPU suite covers strict FK, names, manifest eligibility/integrity, immutable
preparation, explicit perturbations, commands, duplicate-run exclusion, GLIBC
failure recording, release forces, event alignment, clock resets/gaps, full
coverage, phase-zero initialization without integration/clock changes, rotated
world-link velocities, stale DDS receipt/state/heading rejection, cold-start
versus causal/scored/bracket gaps, immutable new-directory rescoring, failure
outcomes and native validity masks. Physics tests skip
explicitly without MuJoCo. Synthetic tests do not establish actual DDS,
TensorRT, local inference, remote portability, or successful study evaluation.
