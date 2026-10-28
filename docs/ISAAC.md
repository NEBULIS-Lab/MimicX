# Optional Isaac Lab Backend (Experimental)

Start with the public [pinned source/environment installation instructions](ISAAC_INSTALL.md).

MuJoCo/MjLab remains the default. Core imports, conversion and command
preparation do not import Isaac, Torch or start Kit. This integration uses
BeyondMimic's `whole_body_tracking` G1 task, robot, observations, action scales,
PD gains and PhysX settings, with project-owned training/rollout entry points.
No W&B account, artifact registry, or remote checkpoint lookup is used.

## Runtime Boundary

The upstream reference is Isaac Lab 2.1 / Isaac Sim 4.5 with RSL-RL 2.x.
The experimental port targets Isaac Sim 6.0, Isaac Lab v3.0.0-beta2, Python 3.12,
Torch >=2.10 and RSL-RL 5.0.1. These are distinct environments, not interchangeable
checkpoint formats. RSL 5 uses separate actor/critic MLPs and TensorDict
observations; RSL 2 uses ActorCritic and the older wrapper contract. Other RSL
major versions are explicitly rejected. Lab main is not a substitute for beta2.

The adapter installs a process-local compatibility alias for the upstream
`AdditiveUniformNoiseCfg` import when Lab3 only exposes `UniformNoiseCfg`
(same bounds and default additive operation). It runs after AppLauncher and
before upstream's eager task registration. Shared upstream files are not edited.
Standalone upstream robot probes must call
`mimicx.backends.isaaclab.compat.prepare_upstream_import()` at that boundary too.

The beta2 adapter explicitly initializes `sim.physics` with PhysX while applying
the upstream buffer settings, unwraps asset ProxyArrays only for the upstream
Torch command implementation, and converts canonical WXYZ motion quaternions
once into Lab3's native XYZW convention. Lab3's native math and observations stay
XYZW; state exports return to WXYZ. Body velocities are selected at link origins,
not COMs. Root resets translate link-origin velocity to COM velocity explicitly
before calling the PhysX COM setter, avoiding beta2's link-velocity setter buffer
mismatch. Native Lab2 paths retain WXYZ and their legacy link-state setter.

Lab3's nested URDF also needs full-tree contact reporters. A project-owned
spawn wrapper enables reporters on each rigid link; a PhysX sensor subclass
creates exact env-major link-path views rather than assuming flat siblings.
Native force/history kernels are retained. Only unfiltered net forces are
supported. The runner rejects incomplete sensor coverage and records the
contact-body order separately from articulation order.

The Sim6 URDF importer ignores the older `replace_cylinders_with_capsules`
option. Baseline and MimicX must share the same actual Sim6 asset; exact Sim4.5
collider/solver parity has not been established. Local motion/checkpoint inputs
remove registry access, but simulator material/ground assets can still cause
startup cache/network lookups. A fully network-disconnected installation has
not been certified.

Prepare the optional runtime and upstream G1 assets using [ISAAC_INSTALL.md](ISAAC_INSTALL.md). The primary
Sim 6 / Lab3 beta2 pairing has passed the bounded runtime checks below; the
Lab2 / Sim4.5 fallback has not been run in this integration session. CPU tests
also exercise mapping, routing, patch translation, telemetry and the closed-loop
protocol with fake processes; those tests are separate from PhysX evidence.

## Reference Conversion

Input NPZ fields: `fps`, `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`,
`body_lin_vel_w`, `body_ang_vel_w`. Joint arrays are `[T,J]`; body arrays are
`[T,B,3]` (quaternions `[T,B,4]`). All bodies of the articulation must be present,
not only the 14 tracked bodies. Supply named arrays `joint_names`, `body_names`
or explicit source names in the mapping. Never infer order from 29/30 columns.

Mapping JSON contains `target_joints`, `target_bodies`, optionally
`source_joints`, `source_bodies`, and `quaternion_order` (`wxyz` or `xyzw`,
describing the input). All name values are ordered string lists. Obtain target
order from the actual Isaac articulation and source order from the producer.
Defaults are SI units, world-frame angular velocities, link-origin body linear
velocities, and 50 Hz. `linear_velocity_reference` must be `link_origin`, not
`com`. Existing convention metadata cannot contradict the mapping declaration.
Other units
or velocity frames must be converted at the source. Conversion reorders existing
FK arrays; it does not compute FK or certify correspondence to a different asset.

```bash
python scripts/isaac/convert_motion.py source.npz isaac_motion.npz --mapping mapping.json
```

Conversion never overwrites an input or existing output. It rejects mismatched
names, non-unit quaternions, nonfinite values, malformed shapes and irregular
timestamps. It writes explicit SI/world/wxyz metadata. Training checks the
complete named order against the live articulation and FPS against control dt.
Different-rate data must be resampled consistently before this adapter; editing
`fps` alone is not resampling. Sampled FK, collider/inertia and asset identity
checks remain runtime validation gates.

## Local Training, Resume And Rollout

Run inside the separately prepared simulator environment with `whole_body_tracking`
importable. Paths below are illustrative, relative to the checkout.

```bash
python scripts/isaac/train.py --motion-file isaac_motion.npz \
  --output-root outputs/isaac --run-name initial --iterations 200 --num-envs 1024 --headless
python scripts/isaac/train.py --motion-file isaac_motion.npz \
  --checkpoint-file outputs/isaac/initial/model_199.pt \
  --output-root outputs/isaac --run-name continuation --iterations 200 \
  --learning-rate 2e-6 --patch-file candidate.json --headless
python scripts/isaac/rollout.py --motion-file isaac_motion.npz \
  --checkpoint-file outputs/isaac/continuation/model_398.pt --steps 150 \
  --metrics-output outputs/eval/metrics.json \
  --step-metrics-output outputs/eval/metrics_steps.csv --seed 11 --headless
```

Use the actual checkpoint filename emitted by the runner, not an assumed step
number. `--iterations` means additional PPO updates after loading. Existing run
directories are rejected. Local TensorBoard logs, resolved environment/runner
YAML, input identities and candidate patch are saved. An explicit learning rate
is reapplied to optimizer groups after loading (subsequent adaptive PPO updates
may change it). Training saves and reloads the final checkpoint and compares
deterministic actions on the identical observation; only an executed passing
check produces `reload_check.json`.
The output root also receives `<run-name>.completed.json` only after this check;
AutoRefine requires that artifact so a Kit shutdown exit code cannot disguise a
partially failed training run. Exceptions are printed before simulator shutdown.
RSL-RL checkpoint suffixes are iteration indices, not a cumulative interaction
counter. The example filenames follow the pinned RSL 5 runner's zero-based loop.
GitPython defaults to `GIT_PYTHON_REFRESH=quiet` so a missing container git binary
does not prevent training; explicit overrides are respected. The source manifest
hash and declared upstream identities are recorded independently in run metadata.

`--task BeyondMimic-Tracking-Flat-G1-v0` selects the unpatched upstream baseline;
it rejects patch input. The default `MimicX-Tracking-Flat-G1-v0` uses upstream
sampling when no patch is supplied. Patches translate the existing bounded
reward families, body selections and start/window sampling. The legacy
`motion_global_root_pos/ori` keys map to **anchor** position/orientation equations,
not a new pelvis objective. Window proposals mix the diagnosed window with
phase-zero replay. Training changes do not alter evaluation thresholds.

Rollout starts at phase zero, removes pose/velocity/joint reset perturbations
and policy observation noise, disables pushes, and retains upstream physical
termination thresholds. One environment and `0 < steps < frame_count` are
required. Terminal state is captured before auto-reset. Reference-end wrap cannot
teleport a successful trajectory back to its start. JSON and per-step CSV include
failures and body/anchor/joint errors; mean metrics are over all requested steps,
including post-failure reset episodes, not just successful prefixes. A separate
`.states.npz` stores physical joint/body states and velocities, reference positions,
actions and contact forces when the contact sensor is present. The first reset's
relative-body targets are initialized after FK without advancing the reference.
`--video-output DIR` optionally records the physical policy rollout with cameras.
This is not a ghost-only replay or a claim of matching another simulator's scorer.

### Reference Clock and Terminal Semantics

The adapter preserves upstream ordering: apply the action, advance physics,
compute termination and reward against the **action-command reference frame**,
then auto-reset failed environments and advance the command for the next
observation. Thus the first post-physics sample is at `dt` but compares against
command frame 0. These metrics are not an endpoint-time FK scorer; the final
reference frame is prepared for the next observation rather than scored as the
last action-command frame. With T frames, at most T-1 physical steps are allowed.

The initial episode starts at frame 0. After a failure, upstream resets the
robot to frame 0 and advances the command to frame 1 before the next action.
This inherited behavior is retained, not silently redefined during integration.
`motion_frame`, `sample_time_s`, `command_time_s`, `episode_id`, `episode_step`
and `episode_time_s` make the physical and reference clocks inspectable.

Telemetry captures states and per-term flags immediately after termination
computation, before reset can overwrite them. `terminated_any` and
`truncated_any` are separate; the existing gate conservatively treats **any**
episode end as a failure via `done_any`. Strict rollout sets normal episode
expiry beyond the requested horizon, so an expected timeout is not counted as
successful completion. Post-failure episodes are retained in whole-rollout
means; they do not lengthen the first-failure horizon. Reference exhaustion is
not allowed to teleport the robot during strict rollout.

Telemetry schema `mimicx.isaac-rollout.v2` commits JSON, CSV and finite state NPZ
before atomically creating `<metrics-stem>.completed.json` with artifact hashes.
AutoRefine requires all four files; missing state export is not success even if
Kit exits 0. Output-path collisions, hard-link aliases and attempts to overwrite
the input motion/checkpoint are rejected before simulator startup. New code
does not treat legacy three-file exports as transaction-complete commands;
those records remain readable for analysis and explicit replay. Use frozen code
for existing runs, and new output directories for a new protocol version.

## AutoRefine Routing

Existing manifests remain unchanged. For Isaac set `runtime.backend: isaaclab`,
point `runtime.train_script` and `runtime.rollout_script` at these entry points,
set `runtime.dynamic_task: MimicX-Tracking-Flat-G1-v0` and
`task.strict_task: MimicX-Tracking-Flat-G1-Strict-v0`. Use a converted named motion
and a native Isaac checkpoint. `runtime.checkpoint_root` is passed as output root;
`**/{run_name}/model_*.pt` remains a valid checkpoint glob. Existing diagnosis,
repeated validation, precision guards and incumbent protection are reused.
The proposal CLI's `--runtime-backend` is distinct from its existing `--backend`
(heuristic/llm/hybrid). Neither routing nor script construction launches training.

MuJoCo checkpoints are not claimed transferable. One-motion runtime verification
does not establish sim-to-sim transfer or replace the planned multi-motion,
multi-training-seed evaluation.

## Runtime Verification (2026-10-28)

Validated primary environment: Isaac Sim 6.0, Lab v3.0.0-beta2, Python 3.12.13,
Torch 2.11.0+cu128 and RSL-RL 5.0.1. Inputs and code were frozen for each run.
The real-motion pilot was the provenance-audited BONES walk-start reference,
278 frames at 50 Hz, reordered to the actual 29-joint/30-body articulation.

| Check | Actual Runtime Result |
|---|---|
| Nine sampled reference states through the adapter reset | Position error <=8.3e-7 m, orientation error <=2.6e-6 rad, all-link velocity error <=1.3e-6 m/s |
| Full-body contact scope | Eight environments x 30 named rigid links; physical ankle/body forces present |
| Local-file train and resume | Two PPO updates each, eight environments; native checkpoints saved and loaded |
| Identical-observation action reload | Maximum absolute action difference 0 for warmstart and resume |
| Strict physical rollout | 277 steps; finite state/action/contact arrays and terminal reasons exported before reset |
| Actual AutoRefine iteration | Two incumbent evaluations, real failure mining/proposal, one two-update candidate continuation, two candidate evaluations, guarded acceptance |
| Incumbent protection | CPU replay of the same four physical evaluation artifacts with roles reversed and unchanged guards rejected the previous baseline; no additional physical iteration |

The full-contact run's `selection.json` accepted the candidate and the loop
ended as `budget_exhausted`, not `strict_success`. Its policies still terminated
during the reference. This is an end-to-end integration smoke test with a tiny
training budget, not evidence of trained-policy quality or method superiority.
Earlier pelvis-only contact runs are not full-contact validation evidence.

The full-contact refinement smoke above predates the v2 export-transaction/clock
metadata hardening. A subsequent actual physical rollout (v11), using the public
v2 exporter and the completed 100-update profile checkpoint, verified all 277
steps, the JSON/CSV/state-NPZ completion transaction and all three artifact
hashes, finite named state/contact arrays, episode/reference clocks, and terminal
flags. Independent CPU auditing confirmed those artifacts and episode boundaries.
The exporter also has CPU regression coverage for masked-exit and partial-write
failures.

The v11 checkpoint had only 100 training updates. Its rollout recorded 21
terminations, the first at step 13, and zero truncations: this verifies physical
export and terminal logging, **not policy success**. The longer warmstart and
research comparison remain separate work; neither this export check nor the
bounded refinement smoke establishes method superiority.

## CPU Verification

```bash
python -m pytest tests/isaac tests/autorefine tests/runtime -q
```

Upstream source: [BeyondMimic whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking).
Simulator and robot assets retain their upstream licenses and acquisition terms.
