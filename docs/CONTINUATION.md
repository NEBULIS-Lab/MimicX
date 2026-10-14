# Recovery and Terrain Continuation

These entry points migrate the recorded recovery PPO workers and execution gates.
They do not replace the original paper matrix, task defaults, references, or
checkpoints. Published demonstration evaluation and additional training remain
separate operations. Obtain the matching weights, references, scenes, profiles,
and frozen runtime through the versioned demonstration release.

## Runtime Configuration

The lightweight package supports CPU imports and metric tests without Torch,
MuJoCo, or mjlab. Training and physical evaluation require the separate simulation
environment. The published runtime records Python 3.11, Torch 2.7.0, mjlab 1.2.0,
MuJoCo and mujoco-warp 3.5.0, warp-lang 1.12.0, rsl-rl-lib 5.0.1, NumPy 2.4.6,
and tensordict 0.13.0. Use its complete `requirements.txt` and `environment.json`;
the lightweight package installation does not provide this stack.

JSON paths resolve relative to the configuration file, not the working directory.
`backend_root` contains the configured `src/tasks` and robot `src/assets` tree.
`native_root` contains `scripts/rollout_mjlab_motion_smoke.py`; it selects the
rollout script without replacing the imported `mimicx` package. `rollout_script`
can explicitly select another compatible script. `--backend-root` and
`--native-root` override their corresponding JSON values. An older configuration
with `snapshot` resolves its `backend/` and `native/` subdirectories.

Minimal core evaluation configuration, with the task's recorded values:

```json
{
  "backend_root": "backend",
  "native_root": "runtime",
  "runtime_kind": "core_recovery",
  "route": "world_fidelity",
  "motion": "assets/reference.npz",
  "horizon": 518,
  "joint_init_noise": 0.01
}
```

```bash
python scripts/policy_recovery/worker.py \
  --config configs/recovery.json --mode evaluate \
  --checkpoint checkpoints/selected.pt --output runs/evaluation-seed4004 \
  --seed 4004 --device cuda:0
```

Choose a device visible to the launched process. The worker does not modify device
visibility, and rejects inherited `MIMICX_DEVICE_ID` during evaluation so the
exporter cannot silently override `--device`. Fixed-reference evaluation also
rejects `MIMICX_AUTOREFINE_PATCH_FILE`.

Evaluation requires a new output directory and always exports `metrics.json`,
`steps.csv`, and `replay/`. `--video` adds 1920-by-1080, 50-fps video and sampled
frames; `--no-ghost` hides the reference in that video. The exporter uses the
latest release's ghost alpha of 0.16. Terminations, reference heights, initial
reference-cache refresh, control horizon, and configured joint noise remain
active. The evaluator's output can contain terminal and post-reset samples;
selection metrics never count them as continued execution.

## Terrain Evaluation

Use the same arguments with `scripts/parkour/worker.py` and one of:

| `runtime_kind` | Additional configuration | Registration |
| --- | --- | --- |
| `scene_v1` | `scene`, `motion`, `horizon` | Recorded Track, Stairs, Forest, and Platform scene-v1 evaluation |
| `terrain_profile` | `scene`, `profile`, `motion`, `horizon` | Recorded profile-driven evaluation, including Descent |
| `terrain_recovery` or omitted | `scene`, `motion`, `horizon`, `route`, `case`, `events` | Existing terrain continuation environment |

Scene-v1 and profile registrations are evaluation-only in this worker and retain
their recorded joint noise of 0.01. Use the asset paths and horizon from the paired
release, not a profile inferred from another task. The original fixed-reference
and contact-aware parkour registrations also remain available through
`mimicx.parkour.environment.register`.

## Training and Resume

Core training adds `num_envs`, `train_seed`, `learning_rate`, and
`updates_per_block` to the configuration. The source campaign used 1024
environments, seed 101, 500 updates per block, and base learning rates of `1e-4`
for execution recovery or `3e-5` for world fidelity. These are recorded settings,
not new defaults imposed on existing paper runs.

```bash
python scripts/policy_recovery/worker.py \
  --config configs/recovery.json --mode train \
  --checkpoint checkpoints/incumbent.pt --output runs/block00/train \
  --block 0 --window 60 100
```

Terrain continuation uses `scripts/parkour/worker.py`, `training_seed` instead of
`train_seed`, and optionally `--practice practice.json` containing
`{"windows": [[60, 100, 0.3], [0, 240, 0.2]]}`. Preserve the source route and event
configuration. Existing optional phase and tail objectives are imported as
`mimicx.parkour.phase_dynamics` and `mimicx.parkour.tail_objectives`; they remain
off unless the configuration enables them.

Core learning rates decay by `0.85 ** block`; terrain rates by `0.9 ** block`.
The PPO schedule stays fixed. After loading, both the algorithm learning rate
and every optimizer parameter-group learning rate are set explicitly. A fresh
block zero resets optimizer state; subsequent blocks and interrupted-block
resumes restore it. Save intervals remain 250 updates for core recovery and 50
for terrain recovery.

`effective_training.json` fixes the inherited iteration, block budget, practice
windows, configuration, inputs, and transition budget. `attempts/` records
each invocation. Changed contracts and checkpoints without original metadata
are rejected. Re-run the same training command to resume:

- A saved `iter` represents the last completed update; resume starts at `iter + 1`.
- Fresh blocks retain the original worker's inherited-index convention.
- Update and transition budgets count only the requested additional training.
- A fully completed block does not create another environment or train again.
- An interrupted finalization promotes the last complete periodic checkpoint.
- Writes use temporary files and atomic replacement. Corrupt newest checkpoints
  fall back to valid older checkpoints; an entirely invalid history stops.

Resume restores counters and optimizer state, not simulator episodes or random
generator state. It is not a bitwise rollout continuation. Checkpoints must be
trusted: the simulator's native checkpoint loader uses Torch serialization.
Use one writer per training directory.

## Recovery Selection

`python scripts/policy_recovery/campaign.py --config configs/campaign.json` keeps
the source train-diagnose-select loop. Add `task`, `checkpoint`, `output`,
`blocks`, `selection_seeds`, and `holdout_seeds`. The original campaign used
selection seeds `[1001, 2002, 3003]` and holdout seeds `[4004, 5005, 6006]`.
`checkpoint_sha256`, `motion_sha256`, and a relative-path-to-hash `source_hashes`
manifest can verify frozen inputs. Older `snapshot` configurations retain their
`source_hashes.json` verification. `campaign` optionally selects the media
collection directory.

The incumbent remains protected by the original common-prefix tracking guard,
completion ranking, quality gate, failure-window selection, and separate
holdouts. Selection uses noise 0.005; holdouts use 0.01. Completed stages are
reused only when commands, configurations, checkpoints, inputs and outputs
match the saved run.
Evaluation artifacts live below each stage's `rollout/`; interrupted
outputs are retained under `incomplete_rollout_*` before a fresh evaluation.
Terrain's separate paired-prefix/per-seed gates and practice-window logic are
available in `mimicx.parkour.metrics`; these are not replaced by the core gate.

## Implementation

The environment and scene modules match the bundled demonstration runtime:

| Module | Role |
| --- | --- |
| `mimicx/policy_recovery/environment.py` | Main-task continuation environment |
| `mimicx/parkour/environment.py` | Contact-aware Parkour environment |
| `mimicx/parkour/scene_geometry.py` | Collision-scene geometry |
| `mimicx/parkour/terrain_environment.py` | Terrain tracking environment |

Related modules:

| Destination | Source | Adaptation |
| --- | --- | --- |
| `policy_recovery/metrics.py` | Native `mimicx/policy_recovery/metrics.py` | Exact copy |
| `parkour/registered_contacts.py` | Native `mimicx/parkour/registered_contacts.py` | Exact copy |
| `policy_recovery/worker.py` and both worker entry points | Native `scripts/policy_recovery/worker.py`; `terrain_recovery_worker_20261010.py` | Portable paths, shared stage implementation, resume, immutable evaluation outputs |
| `policy_recovery/resume.py` | `terrain_resume_20261010.py` | Same update accounting; atomic cleanup, invalid-filename filtering, final-budget check |
| `scripts/policy_recovery/campaign.py` | Native `scripts/policy_recovery/campaign.py` | Portable paths and process inheritance; preserved selection loop |
| `parkour/recovery_environment.py` | `terrain_recovery_environment_20261010.py` | Only two package-relative objective imports changed |
| `parkour/phase_dynamics.py` | `parkour_roll_objectives_20261010.py` | Exact copy |
| `parkour/tail_objectives.py` | `parkour_tail_fidelity_20261011.py` | Only three package-relative imports changed |
| `parkour/metrics.py` | `terrain_recovery_20261010.py`, `failed` through `configured_decision` | Extracted CPU gates; package-relative optional imports |
| `parkour/release.py` | `releases/2026-10-11-scenes/run_scene.py::register_evaluation` | Exact function body, normal package module |

The original paper backend overlay remains unchanged by these modules. For a
demonstration-only backend, use the frozen tracking tree and its matching robot
assets via `python scripts/setup_dependencies.py unitree_rl_mjlab --profile demonstrations`.
Use its separate `third_party/unitree_rl_mjlab_demonstrations` checkout as
`--backend-root`. Compared with the paper overlay, the frozen tree also supplies
`mdp/metrics.py`, `mdp/observations.py`, `rl/__init__.py`, `rl/autorefine_lora.py`,
and package initializers. Its G1 configuration includes additional AMASS and
LoRA registrations. Do not silently apply those registrations to paper runs.

## Verification Boundary

```bash
python -m pytest -q tests/test_policy_recovery_metrics.py \
  tests/test_policy_recovery_worker.py tests/test_terrain_release_environment.py
```

CPU coverage includes lightweight imports, source identity, geometry, collision
configuration, play-mode noise, initial observation refresh, execution gates,
path resolution, immutable outputs, and interruption/resume budget accounting.
Simulator dependencies are substituted at the worker boundary in the CPU
resume test. Matching-weight/seed full-horizon physical rollout comparisons and
actual optimizer/checkpoint integration require the separate simulation runtime.
