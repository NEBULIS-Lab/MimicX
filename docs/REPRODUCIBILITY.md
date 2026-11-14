# Reproducing MimicX

This repository implements **MimicX: Policy-in-the-Loop Supervision Refinement
for Video-Driven Humanoid Motion Tracking**. Runtime configuration and input
identity are separate from measured-result archives and pretrained weights.
Use the same project entry points with the input and evaluation protocol that
matches your experiment. Preserve recorded inputs and gate decisions when
reproducing an existing run.

## Choose an Entry Point

| Starting point | Workflow | Evaluation conditions |
|---|---|---|
| Human video | [Reconstruct, retarget, train and refine](QUICKSTART.md) | Declare reference timing, horizon and repeated seeds before training |
| Prepared robot motion | [Validate inputs](MOTION_STUDY.md), then [run the matched-budget comparison](CONFIRMATION.md) | Shared warmstarts; BeyondMimic/MjLab, Static Search and MimicX; nominal and perturbed holdouts |
| Released SONIC controller | [Common-clock evaluation](SONIC_STUDY.md) | Released controller without per-task PPO; match reference, physical protocol and clock |
| Native Isaac simulation | [Install](ISAAC_INSTALL.md), [convert, train and refine](ISAAC.md) | Native Isaac assets, state conventions and checkpoints |
| Existing policy | [Paired downloads](DEMONSTRATIONS.md) and [continuation](CONTINUATION.md) | Pair each checkpoint with its actual reference and robot configuration |
| Component comparison | [Four-task input bundle](#exact-core-input-bundle) | Recorded task references, candidate budgets and three-seed repetitions |
| Feedback execution cost | [HLoop](HLOOP.md) | Identical dependency graph, inputs and selection evidence across executors |

These are experiment configurations within one framework. The completed
12-motion study used MjLab; native Isaac has a one-motion physical validation.
Do not interchange simulator checkpoints or combine metrics with different
body sets, clocks or success criteria. Unified HF `main` separates the nine
demonstration actors (`policies/<task>/policy.pt`) from the 12-motion comparison
policies (`comparisons/<task>/<method>/seed-101/policy.pt`). Their current
`policy_index.json` uses `mimicx.policy-index.v2`, not the historical `trials`
schema. BONES raw motions are not redistributed: acquire authorized upstream
inputs and follow `benchmarks/motion-tracking/references.json` for the source,
window, conversion recipe and expected hashes. The original four-task core
archive is retained at `archive-before-unified-20261114` in both HF repositories.

## Available Components

| Component | Released material | Required separately |
|---|---|---|
| Video reconstruction | `scripts/reconstruct_video.py`, pinned GVHMR source | Authorized input video, GVHMR weights, licensed SMPL-X models |
| Retargeting | GMR wrapper, motion conversion, pinned GMR source | Reconstruction outputs and body models |
| Tracking and continuation | Pinned Unitree/MjLab backend setup, `dependencies/backend_overlay/`, PPO configuration | GPU runtime and task warmstart, or train one using the quickstart |
| Task-aware refinement | Diagnosis, proposals, coordinator, objective/curriculum overlay | Motion and current policy |
| Repeated execution gate | Verification, accept/protect persistence and resume tests | Recorded or newly generated rollout telemetry |
| Four-task controlled study | `mimicx/configs/paper/core_inputs.json`, four method YAMLs, matrix/configuration scripts | Exact registered input bundle described below |
| HLoop workload | Real fixed-policy benchmark generator, three executors, report selector | Four registered fixed policies from the same input bundle |
| SONIC comparison | Motion bridge, rollout audits and common metric adapters | Upstream SONIC runtime/weights and deployment records |
| Matched-budget motion comparison | Frozen protocol, adaptive proposals, shared-warmstart coordinator and repeated local/global gate | Authorized named robot motions and installed MjLab runtime |
| Isaac backend | Named motion conversion, native train/resume/rollout and AutoRefine routing | Pinned Isaac environment, upstream G1 assets and native checkpoints |
| Additional video and motion breadth | General task/comparison entry points and website evidence | Authorized source clips/motions and task warmstarts; these are not part of the core input bundle |
| Visualization | Rollout/reference exports and state/replay utilities | Simulation runtime; presentation scenes have independent asset terms |

## Exact Core Input Bundle

`mimicx/configs/paper/core_inputs.json` lists 22 inputs: four original
motions, two pre-registered repaired motions, four warmstarts, eight historical
parameter dumps, and four fixed-policy HLoop checkpoints. The bundle has been
assembled and verified by the authors. It is now available through the paired
[MimicX-Policies](https://huggingface.co/Shuaijun/MimicX-Policies) and
[MimicX-Assets](https://huggingface.co/datasets/Shuaijun/MimicX-Assets)
repositories at the historical tag **`archive-before-unified-20261114`**,
rather than stored in Git. Download both complete tagged snapshots into
dedicated directories; unified `main` no longer contains the old hash-addressed
core archive. Its v2 policy index must not be passed to historical `prepare-eval`.
The tagged index retains `mimicx.hf-policy-index.v1` and its `trials` records;
see the [archived artifact usage guide](https://huggingface.co/Shuaijun/MimicX-Policies/blob/archive-before-unified-20261114/USAGE.md).

```bash
python -m pip install -e '.[artifacts]'
python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download

revision = 'archive-before-unified-20261114'
root = Path('artifacts') / revision
snapshot_download('Shuaijun/MimicX-Policies', revision=revision,
                  local_dir=root / 'MimicX-Policies')
snapshot_download('Shuaijun/MimicX-Assets', repo_type='dataset', revision=revision,
                  local_dir=root / 'MimicX-Assets')
PY
python scripts/reproduction/hf_artifacts.py core-inputs \
  --policies artifacts/archive-before-unified-20261114/MimicX-Policies \
  --assets artifacts/archive-before-unified-20261114/MimicX-Assets \
  --output artifacts/paper-core-inputs
```

The CPU-only `core-inputs` command checks the complete archived `FILES.json`
inventories and reconstructs the 22-file bundle. `verify`, `prepare-eval` and
`verify-gates` in the same script also target these historical snapshots, not
the new main index. `prepare-eval --trial` uses an exact ID from the tagged
`trials` list and only prepares commands; it does not execute a rollout.
Public parameter dumps use portable paths; use the
emitted `manifest.json` with `--spec`, as shown below. Checkpoint and motion
bytes are unchanged. Do not substitute a newly trained
warmstart while calling it an exact reproduction of the reported continuation.

Raw video reconstruction uses separately obtained source videos/models. The
core continuation protocol starts from the registered robot references, not
from a repeated stochastic reconstruction of the original video.

After obtaining the bundle and installing the pinned backend:

```bash
python scripts/reproduction/configure_core.py \
  --assets artifacts/paper-core-inputs \
  --spec artifacts/paper-core-inputs/manifest.json \
  --backend third_party/unitree_rl_mjlab \
  --runtime-python /path/to/tracking-environment/bin/python \
  --output runs/paper-config
python scripts/autorefine/launch_paper_matrix.py --prepare \
  --matrix runs/paper-config/matrix.yaml --run-dir runs/paper-core --gpus 0
```

The first command verifies every asset hash, installs the warmstarts at the
backend's expected run paths, and generates relocatable task/method settings.
It expands to 48 trials: four tasks, four methods, three continuation seeds.
The second prepares runnable jobs; it requires the tracking Python environment
for checkpoint inspection. Run those jobs only inside your compute allocation.
See `launch_paper_matrix.py --help` and [HLoop](HLOOP.md) for execution.

| Method name | Configuration file | Supervision |
|---|---|---|
| Fixed Reference | `m0_open_loop.yaml` | Fixed tracking objective |
| Failure Curriculum | `m1_policy_window.yaml` | Diagnosed failure-window curriculum |
| Task-Aware Refinement | `m2_task_hierarchy.yaml` | Task/body-aware objective plus curriculum |
| MimicX | `m3_full_mimicx.yaml` | Bounded proposals, continuation and repeated execution gate |

Internal IDs are retained for joining historical records; presentation uses
method names. Three static methods use 250 continuation iterations. MimicX
uses one search round with three candidates, each with the same 250-iteration
continuation budget. Report candidate budget as well as per-candidate budget.
Verification seeds are fixed to 1001, 2002 and 3003. The registered reference
does not change within an automated loop.

For Dance and Kung Fu, MimicX uses a pre-registered repaired reference while
the three component methods use the original reference. Tennis and Football
share the original reference across methods. The configuration preserves this
paper protocol; body errors are relative to each method's selected reference.

## Inputs, Outputs and Verification

For new authorized videos, follow [the quickstart](QUICKSTART.md) to reconstruct,
retarget, train a warmstart and run the loop. This reproduces the method on new
inputs without depending on the authors' exact checkpoint bundle.

Keep the run configuration, selected checkpoint, candidate decisions, repeated
rollout telemetry and actual reference together. Report shared warmstart cost
once, all attempted candidate training, selection overhead and final evaluation
cost separately. SONIC evaluation cost does not estimate its upstream pretraining.

CPU tests verify control flow, acceptance/protection, scheduler semantics and
configuration contracts. Physical execution is verified by saved simulator
rollouts and completion records. Retain unsuccessful and protected outcomes;
distinguish a failed motion from incomplete or invalid telemetry. Full-motion
errors and common-prefix errors have different denominators and should keep
their respective labels. The release notes describe completed runtime coverage.
