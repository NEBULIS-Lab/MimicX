# Recording and Contact Replay

These tools replay recorded robot states and optionally integrate a dynamic ball.
They do not train an object-aware policy, modify an existing checkpoint, replace
paper trials, or change the published model/metric claims. Ball initial-state
fitting is a display reconstruction operation, not controller optimization.

## Dependencies

- Base preparation, recording guards, camera math, fitting utilities and numerical
  audits use the existing NumPy/SciPy dependencies.
- Physics requires MuJoCo; CPU integration was checked with MuJoCo 3.5.0.
- Video export also requires `imageio-ffmpeg` (provided by the visualization extra)
  and a working, explicitly selected MuJoCo graphics backend.
- OpenCV, Torch, external controller weights and licensed visual assets are not
  imported by the CPU helpers. No external weights or assets are bundled here.

## Baseline Recording Adapter

`mimicx.visualization.baseline_recording` provides:

- `release_if_requested(sim_env, request_path)`: disables the startup elastic band,
  clears its body's residual applied wrench, then atomically acknowledges the
  request. A stale acknowledgement with renewed support raises an error. Use a
  fresh request filename for every run; other bodies' forces are not erased.
- `install_no_reset_fall_monitor(sim_env)`: records falls without resetting state.
- `require_forward_time(previous, current)`: rejects clock stalls/resets.
- `named_joint_addresses(model, names)` and `joint_permutation(source, target)`:
  validate complete scalar-joint mappings instead of assuming backend DOF order.
- `StateRecorder(model, joint_names, body_names=())`: captures actual state,
  simulator and monotonic clocks, named joints, startup support and optional body
  telemetry. World-frame body linear velocities refer to body origins, with the
  subtree-COM offset corrected explicitly. `append(...)` copies arrays;
  `save(path)` writes a numeric NPZ.
- `validate_playback(states, playback_wall, clock_audit=None)`: checks physical
  coverage, clocks and startup forces, including interpolation boundary samples.
  A supplied legacy clock/named-joint audit is validated as additional evidence.

These are **integration APIs**, not a replacement SONIC executable. The upstream
runtime, transport, configuration, controller and weights must be obtained
separately. This migration does not launch or patch an installed controller.
In the external simulator loop, install the no-reset monitor once, process release
requests before stepping, reject non-forward simulator time, and call the recorder
after forward kinematics with `time.monotonic()` and the actual band's enabled
state. The controller must wait for the release acknowledgement before playback.
Keep its existing control loop and reference protocol unchanged.

```bash
python scripts/visualization/audit_recording.py \
  --states inputs/sim_state.npz --playback-wall inputs/playback_wall.npy \
  --clock-audit inputs/clock_audit.json --output outputs/clock.json
```

`playback_wall.npy` is a strictly increasing vector of measured monotonic seconds,
not frame indices. The state archive must contain `time`, `monotonic_time`,
`elastic_band_enabled` and nonnegative `external_wrench_max`. The audit derives
elapsed playback time from simulator timestamps; it never assumes a 50 Hz logger.
Named-joint agreement with a separate controller log requires that additional
audit; clock/force validation alone does not establish it.

## Source-Camera Replay

`mimicx.visualization.source_camera` exposes `resample`, `frame_times`,
`prepare_source_replay`, `project_points`, `set_camera`, `stabilize`,
`protect_robot_pixels`, `composite_ghost` and `export_source_video`.

Input replay directories use the existing exact-rollout package fields:
`timestamps.npy`, `robot_root_pose.npy`, `robot_joint_positions.npy`,
`reference_root_pose.npy`, `reference_joint_positions.npy` and `joint_names.json`.
Optional `reference_timestamps.npy` supplies a distinct reference clock. Poses are
XYZ plus unit **wxyz** quaternion; joint columns follow the names file. The replay,
reference and camera must already share a world frame. No heading correction,
per-frame alignment, translation, looping or duration stretching is applied.

```bash
python scripts/visualization/source_camera.py prepare \
  --replay inputs/replay --frames 301 --fps 30 --output outputs/poses.npz
python scripts/visualization/source_camera.py stabilize \
  --calibration inputs/camera.json --background inputs/background.json \
  --output outputs/stable_camera.json
```

The camera JSON has per-frame `camera_origin_world_m` (N x 3),
`world_from_camera_linear` (N x 3 x 3) and `intrinsics` (N x 3 x 3).
The linear basis must be a proper rotation with positive uniform scale. Intrinsics
are zero-skew pinhole matrices. A supplied source-background audit has
`valid_fraction`, `step_p95_px`, `accumulated_max_px` and `source_width`.
Near-static backgrounds lock the display camera; otherwise translation,
intrinsics and rotation in a shared tangent frame are smoothed. Robot trajectories
never enter this decision. Background optical-flow extraction is not included;
the caller supplies measured source evidence rather than a robot-following camera.

Explicit rendering command, to run only in an appropriately configured graphics
environment:

```bash
python scripts/visualization/source_camera.py render \
  --replay inputs/replay --frames 301 --fps 30 \
  --calibration outputs/stable_camera.json --model inputs/robot.xml \
  --source-size 812 720 --output-size 812 720 --backend egl \
  --ghost-rgba 0.23 0.47 0.66 0.23 --output outputs/replay.mp4
```

The exporter preserves actual-robot pixels under the translucent reference,
including a one-pixel silhouette guard. It replays poses with forward kinematics,
never simulation steps. Full decode verifies frame count, dimensions and FPS
before publishing the MP4. All source frames, including the tail, are requested.
At most 0.04 seconds of endpoint hold is allowed by default; set
`--max-endpoint-hold 0` for strict coverage. The receipt reports endpoint holds,
identity coordinate transform and input hashes. Output pixel protection is exact
before lossy video encoding, not a claim about decoded pixel equality.
Source and output sizes must have the same aspect ratio: only uniform resizing is
supported, with no implicit stretch, crop or letterboxing. All source-camera CLI
commands refuse existing outputs; rendering also refuses an existing JSON receipt.
Video encoding uses a unique temporary directory and publishes the decoded video
atomically without replacing a destination created by another export.

## Dynamic Ball Replay

`BallReplay(xml, timestamps, root_pose, joint_positions, joint_names,
config=BallConfig(...), assets=None)` accepts a caller-owned robot/scene MJCF and
numeric recordings. There must be exactly one free-root robot with scalar joints.
The caller supplies collision geometry, including a racket when needed; existing
visual-only geoms remain non-colliding. An optional plane is controlled by
`floor_z`; use `null` to rely entirely on supplied scene geometry.

`simulate(initial, start=None, end=None)` integrates one ball continuously. The
initial vector is XYZ plus linear velocity, optionally followed by angular
velocity. Substep robot poses and velocities are interpolated from the recording;
large robot armature suppresses contact-induced robot response, and exact recorded
poses are restored each step. The ball is never set to observed positions after
release. This is prescribed robot motion with dynamic object contact, not coupled
closed-loop robot control. The exact final timestamp is retained even when it
requires a shorter last integration step.

Minimal configuration (`inputs/contact.json`, paths relative to this JSON):

```json
{
  "model": "robot_collision.xml",
  "robot": "robot.npz",
  "initial": [0, 0, 1.2, 0, 0, -1],
  "physics": {
    "dt": 0.0008333333333333334,
    "mass": 0.057,
    "radius": 0.0335,
    "damping": 0.18,
    "friction": 0.35,
    "stiffness": 1000000,
    "gravity": [0, 0, -9.81],
    "floor_z": 0
  }
}
```

`robot.npz` contains `timestamps`, `robot_root_pose`,
`robot_joint_positions`, and a Unicode `joint_names` array. No object arrays or
pickle loading are needed. The CLI accepts flattened MJCF with external mesh and
texture paths relative to the model; XML include expansion is not implemented.
The Python API additionally accepts MuJoCo's in-memory asset mapping.

```bash
python scripts/visualization/contact_replay.py simulate \
  --config inputs/contact.json --output outputs/contact
python scripts/visualization/contact_replay.py audit --input outputs/contact
```

Outputs are `ball.npz` (time, position, velocity, quaternion, constraint force and
contact counts), `contacts.json` (contact times, normals, positions and forces),
and `manifest.json` (inputs, physics, initial contacts and grouped impulses).
Validation checks file integrity, Euler position integration, force balance,
unit quaternions, contact timing/duration and world force direction. Every non-ballistic velocity change must have matching contacts.
Initial overlaps are reported, not automatically rejected. Collision geometry and
penetration quality still need task-specific checking. Input hashes cover the
recording/config/model files, not transitive external assets.

`ballistic_initial(...)` computes a release velocity toward a target.
`fit_initial_conditions(...)` uses a seeded, bounded CEM search over six initial
state values; the objective is observed world-position RMSE. To use the `fit`
CLI, add a `fit` object with `observations` (NPZ with `time` and `position`),
six-element `lower`/`upper` bounds, and optional `rounds`, `population`, `seed`.
Run `contact_replay.py fit --config inputs/contact.json --output outputs/fit.json`,
then explicitly use the returned `initial` for simulation. Task-specific sports
selection heuristics, held-ball phases, multi-ball coupling and new policy metrics
are not implemented by this generic fitter.

## Verification Boundary

```bash
python -m pytest tests/test_recording_release.py tests/test_contact_replay.py -q
python -m pytest tests/test_contact_replay_mujoco.py -q
```

The second command needs MuJoCo and exercises CPU simulation only. Rendering is
separate and opt-in: `MIMICX_RENDER_TESTS=1 python -m pytest
tests/test_recording_release_render.py -q`, with the graphics backend already
configured. Separate graphics validation passed both opt-in tests (2 passed in
0.70 seconds). Full sports recordings and external SONIC deployment remain
unverified by this migration; no external SONIC runtime was launched.
The passing opt-in tests cover both off-center pinhole projection and a four-frame
free-root, scalar-joint robot export: uniform downscaling, source FPS, full decode,
moving silhouettes and exact robot-pixel protection before encoding.

## Implementation

The recording utilities cover source-camera projection, camera stabilization,
clock coverage, named-joint sampling, startup-support checks and translucent
references. Contact replay uses MuJoCo integration, bounded CEM fitting and
checks for unsupported changes in ball velocity. These utilities operate on
recorded robot states; external model weights and scene assets are supplied
separately. See the APIs and commands above for each workflow.
