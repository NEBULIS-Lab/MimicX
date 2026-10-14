# Parkour Actor + MPC

This is a separate controller entry point, not a new PPO checkpoint. A frozen
actor supplies the approach; state-feedback MPC takes over at frame 190. There
are **zero additional training updates**. The controller sends joint-position
targets through the existing physical actuators, without root forces or live
pose writes after reset.

The released selected recording (reset 29003) completes 269 control steps and
all seven motion checks. The six recorded validation resets yield **4/6 horizon
completions, 3/6 terminal passes, and 1/6 all-motion passes**. The selected run was
chosen after validation, not an independent unseen success. Its maximum sampled
world-contact penetration is 0.05344 m. The locally retimed reference is 5.38 s.

## Inputs and Dependencies

Use matching Assets and Policies releases. Asset resolution
and download are separate from controller execution; no private paths, weights,
or third-party robot meshes are included in this code.

```bash
ASSETS=/path/to/Assets-snapshot
POLICIES=/path/to/Policies-snapshot
RELEASE="$ASSETS/releases/2026-10-14-demonstrations"
P="$RELEASE/controllers/parkour"
RUNTIME="$RELEASE/runtime"
MOTION="$ASSETS/releases/2026-10-11-scenes/tasks/parkour/motion.npz"
SCENE="$ASSETS/releases/2026-10-11-scenes/tasks/parkour/scene.json"
TRACE="$P/recorded_states/seed29003/traces.npz"
NAMES="$P/recorded_states/seed29003/names.json"
```

`controller.json` in `$P` supplies the matching checkpoint, motion and scene files. Resolve checkpoint paths relative to `$POLICIES`, other paths relative
to `$ASSETS`; do not treat the archival `source-artifact://` entries in
`config.json` as filenames. Keep `names.json`, `planner.json`, `traces.npz`, and
`validation/seed*.json` paired. Controller index 0 is frozen PPO, index 1 is the
previous shooting controller, and index 2 is the selected actor+MPC run.

Lightweight tests and `--help` need only the base package. CPU replay/planning
also requires MuJoCo; the native build and numerical checks used Python 3.11,
MuJoCo 3.5.0 and SciPy 1.17.1. The full native runtime additionally needs the
compatible MjLab/backend installation and separately licensed G1 robot assets.
An installed base package alone does not provide that runtime.

## Build the Model on CPU

The HF controller bundle does not supply `runtime_model.mjb`. Build the actual
native model from the installed robot configuration, scene and reference:

```bash
python scripts/parkour/hybrid.py prepare-model --help
PYTHONPATH="$RUNTIME${PYTHONPATH:+:$PYTHONPATH}" \
  python scripts/parkour/hybrid.py prepare-model \
  --scene "$SCENE" --motion "$MOTION" --output work/parkour-model
```

Use a new output directory. This invokes the native play configuration and
`mimicx.parkour.environment.SceneBuilder`, constructs `Scene(..., device="cpu")`,
compiles it, and applies the native MuJoCo options. It does **not** initialize a
device-backed environment or render. Outputs:

- `model.mjb`: locally compiled model, including native collision/actuator data.
- `affine.json`: named action scales and offsets in action-space order.
- `names.json`: joint order, full-body reference order and tracked-body subset.
- `model_manifest.json`: input files, MuJoCo version and dimensions/timing.

No meshes are copied by this command. A compiled model can contain third-party
geometry; do not redistribute the output without the relevant permissions.

All CPU execution commands also accept `--model-xml FILE` instead of
`--model-mjb FILE`. An XML must describe the **fully assembled** native robot,
servos and scene, with its licensed mesh dependencies resolvable. A bare robot
XML or visual scene is not a substitute. MJB compatibility is version-specific;
rebuild with the matching runtime when necessary.

The preserved settings include 0.005 s physics steps, four substeps per 0.02 s
control frame, implicit-fast integration, native actuator gains/force limits,
ground friction `[0.8, 0.005, 0.0001]`, per-box sliding friction, and unchanged
contact solver parameters. Scene yaw and all poses use **wxyz** quaternions.
MuJoCo free-root angular velocity is local, while reference angular velocity is
world-frame; the implementation converts explicitly. Reference world heights
are never ground-aligned or shifted for display.

## Replay Recorded States

```bash
python scripts/parkour/hybrid.py state-replay \
  --model-mjb work/parkour-model/model.mjb --names "$NAMES" \
  --motion "$MOTION" --traces "$TRACE" --index 2 --output work/state-replay
```

This reconstructs the selected root/joint trajectory and recomputes FK,
completion and contact metrics. It produces numerical states, not a video. Its
report explicitly says `kinematic_state_replay` and
`physical_execution_verified: false`. Reset states after an episode terminates
are excluded. The selected recording contains 270 states for 269 actions.

## Replay Physical Servo Actions

```bash
python scripts/parkour/hybrid.py servo-replay \
  --model-mjb work/parkour-model/model.mjb --affine work/parkour-model/affine.json \
  --names "$NAMES" --motion "$MOTION" --traces "$TRACE" --index 2 \
  --output work/servo-replay
```

Only the initial pose is set; subsequent states result from `mj_step`. The old
trace lacks root velocities, so true-start initialization uses its saved pose
and joint velocities plus the reference's unperturbed frame-zero root velocity,
matching the native play reset contract. `--initial-state FILE.npz` can supply
explicit `qpos` and `qvel` vectors instead, also at frame zero. Do not infer an
exact arbitrary mid-episode state from poses alone.

Recorded actions are open-loop controls, **not actor inference**. CPU numerical
drift can accumulate at contacts. In the current check this run stopped after
83 valid steps with `anchor_pos`, rather than reproducing the selected GPU
trajectory. This result is reported, not replaced with saved poses or resets.

## Recover and Run the Planner

The original recorded nominal prior is not a standalone file in the controller
bundle. Recover its used tail from the selected actions and planner decisions:

```bash
python scripts/parkour/hybrid.py recover-nominal \
  --model-mjb work/parkour-model/model.mjb --affine work/parkour-model/affine.json \
  --names "$NAMES" --traces "$TRACE" --index 2 \
  --planner "$P/recorded_states/seed29003/planner.json" \
  --output work/recovered-prior
python scripts/parkour/hybrid.py plan \
  --model-mjb work/parkour-model/model.mjb --affine work/parkour-model/affine.json \
  --names "$NAMES" --motion "$MOTION" --traces "$TRACE" --index 2 \
  --nominal-actions work/recovered-prior/nominal.npz --output work/cpu-mpc
```

Recovery subtracts each *executed* segment of the three-knot residual from the
saved target, accounting for actuator/action permutations. Only frames 190..268
recover the original nominal prior; the retained prefix is not that prior and
must not be used to claim its reconstruction. Float32 action recording limits
precision: the checked tail differs from the original by at most `6.6e-7` in
normalized action units. It is not byte-identical to the missing source.

`plan` performs real differential-evolution optimization against current CPU
physics state. It uses recorded approach actions before handover and therefore
reports `recorded_prefix_cpu_mpc`, not actor+MPC. It stops on native failure
thresholds; a failed open-loop approach is not advanced to frame 190 by resetting.

Optional `--config planner.json` accepts only `PlannerConfig` fields. The
approved defaults are equivalent to:

```json
{
  "start": 190, "total_steps": 269, "horizon": 45, "replan_every": 3,
  "population": 96, "generations": 24, "threads": 6,
  "optimizer_seed": 28001, "control_dt": 0.02, "substeps": 4,
  "momentum": true, "nominal": "recorded",
  "contact_weight": 0.0, "penetration_tolerance": 0.01
}
```

The objective preserves pelvis orientation/position, joints and hip/knee error,
torso orientation, all-body position, termination-margin guards, world-frame
root angular-velocity tracking, and residual regularization. Both average and peak violations count.
Each plan uses 15 coupled joint channels at three knots, the previous plan as a
warm start, and a deterministic optimizer RNG independent of reset seeds.

Contact-refinement variants set `contact_weight` to 5000 or 20000 and penalize
penetration exceeding 0.01 m using exact world contacts. These are **optional
experimental variants**, not the selected incumbent. Contact physics is never
modified. Likewise, the warm-start variant can set `horizon: 79` and use the
selected trace as `--nominal-actions "$TRACE" --nominal-index 2`; that changes
the prior and is not the original 45-frame controller. `nominal: "reference"`
uses reference joint position plus damping-compensating velocity lead.

## Frozen Actor and Six Resets

With the native runtime and checkpoint installed, use the dedicated entry point:

```bash
CHECKPOINT="$POLICIES/$(python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"]["path"])' \
  "$P/controller.json")"
PYTHONPATH="$RUNTIME${PYTHONPATH:+:$PYTHONPATH}" \
  python scripts/parkour/hybrid.py actor-mpc \
  --scene "$SCENE" --motion "$MOTION" --checkpoint "$CHECKPOINT" \
  --names "$NAMES" --nominal-actions work/recovered-prior/nominal.npz \
  --seed 29003 --device cuda:0 --output work/actor-mpc
```

The evaluator builds the live model itself, refreshes the initial relative
reference cache, loads actor weights only, and creates a fresh planner. It
captures no auto-reset state and ends the episode at its first termination.
`--ppo-only` runs the frozen PPO baseline without MPC and without requiring a
nominal file. Neither mode trains. Device provisioning is the caller's
responsibility. Do not use the PPO-only output as reproduction of the MPC video.

`--six-resets` replaces the single seed with
`28002,28003,28004,29001,29002,29003`, creating a fresh environment and planner
per reset. Full online search is computationally substantial; this is a release
validation hook, not part of the quick check. Fresh single-environment resets
need not be bit-identical to the original three-controller batched evaluator.
For integration with that evaluator, use `Planner.callback(..., index=...)`.

Aggregate the **already recorded** validation without launching simulations:

```bash
python scripts/parkour/hybrid.py summarize --reports "$P/validation" \
  --index 2 --output work/recorded-validation.json
```

Horizon completion, terminal checks and all seven motion checks are separate
fields. Physical run reports also include action magnitude/rate, peak actuator
force and sampled penetration. The native missing-state penalty prevents early
failures from receiving a deceptively low objective.

## Verification Scope

Checked on CPU: native model compilation; numerical agreement of the model's
geometry/contact/inertia/actuator arrays with the original; exact selected
root/joint state reconstruction; recomputed seven-check score and contact
depth; prior-tail recovery; existing six-report aggregation; and a five-candidate,
zero-generation, three-frame planner smoke run from the true initial state.
The latter changes `start` to 0 and `horizon` to 3 for a cheap mechanics check,
not a Parkour performance claim.

Fresh full actor+MPC execution, its six-reset repeat and GPU numerical parity
remain unverified by this port. The checked full CPU open-loop replay fails as
described above. None of these checks establishes collision-free execution or
general success across resets.

```bash
python -m pytest tests/test_hybrid_controller.py tests/test_hybrid_controller_runtime.py -q
```

Runtime tests skip when MuJoCo is absent. To also test the actual native build
and selected recording, set `MIMICX_PARKOUR_SCENE`, `MIMICX_PARKOUR_MOTION` and
`MIMICX_PARKOUR_RECORDING` (the directory containing `traces.npz`/`names.json`),
with the runtime on `PYTHONPATH`. Tests never initialize a GPU or run full search.

Public APIs live in `mimicx.parkour.servo` (`ActionAffine`, `load_model`,
`prepare_model`), `mimicx.parkour.predictive` (`PlannerConfig`, `Planner`,
`recover_nominal`), and `mimicx.parkour.hybrid` (`replay_states`, `run_cpu`,
`evaluate_actor`, `evaluate_six_resets`, `completion_report`). No package-level
exports or PPO loading interfaces are replaced.
