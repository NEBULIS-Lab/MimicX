# Paper Reproduction Inventory

This repository implements **MimicX: Policy-in-the-Loop Supervision Refinement
for Video-Driven Humanoid Motion Tracking**. Runtime configuration and input
identity are separate from measured results, which live in the website only.

## Available Components

| Paper component | Released material | Required separately |
|---|---|---|
| Video reconstruction | `scripts/reconstruct_video.py`, pinned GVHMR source | Authorized input video, GVHMR weights, licensed SMPL-X models |
| Retargeting | GMR wrapper, motion conversion, pinned GMR source | Reconstruction outputs and body models |
| Tracking and continuation | Pinned Unitree/MjLab backend setup, `dependencies/backend_overlay/`, PPO configuration | GPU runtime and task warmstart, or train one using the quickstart |
| Task-aware refinement | Diagnosis, proposals, coordinator, objective/curriculum overlay | Motion and current policy |
| Repeated execution gate | Verification, accept/protect persistence and resume tests | Recorded or newly generated rollout telemetry |
| Four-task controlled study | `mimicx/configs/paper/core_inputs.json`, four method YAMLs, matrix/configuration scripts | Exact registered input bundle described below |
| HLoop workload | Real fixed-policy benchmark generator, three executors, report selector | Four registered fixed policies from the same input bundle |
| SONIC comparison | Motion-format bridge and common metric adapters | Upstream SONIC runtime/weights and deployment records; its machine-specific launcher is not released |
| Additional video and motion breadth | General task/comparison entry points and website evidence | Authorized source clips/motions and task warmstarts; these are not part of the core input bundle |
| Visualization | Rollout/reference exports and state/replay utilities | Simulation runtime; presentation scenes have independent asset terms |

## Exact Core Input Bundle

`mimicx/configs/paper/core_inputs.json` records checksums for 22 assets: four original
motions, two pre-registered repaired motions, four warmstarts, eight historical
parameter dumps, and four fixed-policy HLoop checkpoints. The bundle has been
assembled and verified by the authors. It is now available through the paired
[MimicX-Policies](https://huggingface.co/Shuaijun/MimicX-Policies) and
[MimicX-Assets](https://huggingface.co/datasets/Shuaijun/MimicX-Assets)
repositories, rather than stored in Git. The bundled `prepare.py core-inputs`
command reconstructs the 22-file input bundle; see the
[artifact usage guide](https://huggingface.co/Shuaijun/MimicX-Policies/blob/main/USAGE.md).
Public parameter dumps have sanitized paths and updated checksums; use the
emitted `manifest.json` with `--spec`, as shown below. Checkpoint and motion
bytes are unchanged. Do not substitute a newly trained
warmstart while calling it an exact reproduction of the reported continuation.

Raw video reconstruction uses separately obtained source videos/models. The
core continuation protocol starts from the registered robot references, not
from a repeated stochastic reconstruction of the original video.

After obtaining the bundle and installing the pinned backend:

```bash
python scripts/reproduction/configure_core.py \
  --assets /path/to/paper-core-inputs \
  --spec /path/to/paper-core-inputs/manifest.json \
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

## New Inputs and Verification

For new authorized videos, follow [the quickstart](QUICKSTART.md) to reconstruct,
retarget, train a warmstart and run the loop. This reproduces the method on new
inputs without depending on the authors' exact checkpoint bundle.

CPU tests verify control flow, acceptance/protection, scheduler semantics and
configuration contracts; they do not replace physical-policy experiments.
The release preparation additionally checks the actual input bundle and
materializes the paper matrix and fixed HLoop plan without starting training.
Fresh end-to-end GPU installation and rerunning all supplementary experiments
are outside this packaging verification. The website links all retained
numerical evidence, including protected and unsuccessful outcomes.
