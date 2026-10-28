<div align="center">

<a href="https://nebulis-lab.github.io/MimicX/">
  <img src="docs/assets/branding/mimicx-full-logo.svg" alt="MimicX" width="680">
</a>

<h1>MimicX: Policy-in-the-Loop Supervision Refinement<br>for Video-Driven Humanoid Motion Tracking</h1>

<p>
  <a href="https://nebulis-lab.github.io/MimicX/"><img src="https://img.shields.io/badge/Project-Website-D45B4C?style=flat-square" alt="Project website" height="23"></a>
  <a href="https://arxiv.org/abs/2610.09055"><img src="https://img.shields.io/badge/arXiv-2610.09055-D45B4C?style=flat-square&amp;logo=arxiv&amp;logoColor=white" alt="arXiv paper" height="23"></a>
  <a href="https://huggingface.co/Shuaijun/MimicX-Policies"><img src="https://img.shields.io/badge/Hugging_Face-Policies-D45B4C?style=flat-square&amp;logo=huggingface&amp;logoColor=white" alt="Hugging Face policies" height="23"></a>
  <a href="https://huggingface.co/datasets/Shuaijun/MimicX-Assets"><img src="https://img.shields.io/badge/Hugging_Face-Assets-3B78A8?style=flat-square&amp;logo=huggingface&amp;logoColor=white" alt="Hugging Face assets" height="23"></a>
  <a href="https://github.com/NEBULIS-Lab/shuaijun-ICLR-paper-MimicX"><img src="https://img.shields.io/badge/Paper-Source-7A7F87?style=flat-square&amp;logo=github&amp;logoColor=white" alt="Paper source" height="23"></a>
</p>
<p>
  <a href="https://github.com/NEBULIS-Lab/MimicX/actions/workflows/tests.yml"><img src="https://github.com/NEBULIS-Lab/MimicX/actions/workflows/tests.yml/badge.svg?branch=main" alt="CPU tests" height="20"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10%2B-3B78A8?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python 3.10 or newer" height="20"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-23866B?style=flat-square" alt="Code license: Apache-2.0" height="20"></a>
</p>

**[Overview](#overview) &middot; [Quickstart](#quickstart) &middot; [Policies & Assets](#policies-and-assets) &middot; [Documentation](#documentation)**

<a href="https://nebulis-lab.github.io/MimicX/#policies">
  <img src="docs/assets/media/readme-tennis-motion.png" alt="Successive poses of the recorded MimicX G1 tennis policy, arranged across a tennis-court presentation scene" width="1000">
</a>
<p><sub>Recorded G1 tennis policy motion. Successive poses are arranged spatially for visualization.</sub></p>

</div>

## Overview

**MimicX** learns humanoid tracking policies from video-derived motion and
uses policy execution feedback to refine their training supervision. Rollout
diagnosis identifies difficult time windows and body regions; bounded
objective and curriculum candidates guide policy continuation. Repeated
execution checks determine whether to accept a candidate or retain the
current policy.

<p align="center">
  <strong>Skill construction:</strong> Human video &middot; Human reconstruction &middot; Robot reference &middot; Tracking policy<br>
  <strong>Policy feedback:</strong> Policy rollout &middot; Failure diagnosis &middot; Supervision refinement &middot; Verified continuation
</p>

| Component | What it does |
| :--- | :--- |
| **Video-to-policy pipeline** | Connects GVHMR / SMPL-X reconstruction, GMR retargeting and PPO tracking in MuJoCo / MjLab, with an optional [Isaac Lab backend](docs/ISAAC.md). |
| **Task-aware AutoRefine** | Uses failure windows and body-error channels to propose targeted objectives and curricula. |
| **Repeated execution gate** | Checks execution improvements and tracking guards before replacing the current policy; records acceptance or protection. |
| **MimicX-HLoop** | Overlaps GPU rollout and CPU diagnosis while preserving policy-selection decisions. |

Within each automated loop, the registered reference stays fixed while
training objectives and curricula are refined. Input reference preparation
is a separate stage. The [method documentation](docs/METHOD.md) describes
the training and selection procedure; the [project website](https://nebulis-lab.github.io/MimicX/)
contains the visual comparisons and numerical results.

## Quickstart

### Install the core package

```bash
git clone https://github.com/NEBULIS-Lab/MimicX.git
cd MimicX
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,visualization]'
python -m pytest -q
```

The CPU test suite includes deterministic subprocess fixtures for the complete
refinement cycle. It requires neither body models nor pretrained policies.
Physics training and evaluation additionally use the
[pinned tracking backend](dependencies/README.md).

### Choose a workflow

| Goal | Start here |
| :--- | :--- |
| **Run a released policy** | [Download and verify a paired policy/reference](docs/DEMONSTRATIONS.md), then evaluate with the pinned runtime. |
| **Continue or record a policy** | Use [checkpoint continuation](docs/CONTINUATION.md) and [camera/contact-aware recording](docs/RECORDING_AND_CONTACT.md). |
| **Learn a skill from a new video** | Follow [video reconstruction, retargeting and tracking](docs/QUICKSTART.md). |
| **Refine an existing policy** | Create a manifest and [run or resume AutoRefine](docs/QUICKSTART.md#autorefine). |
| **Reproduce the paper protocols** | Prepare the inputs using the [reproduction guide](docs/REPRODUCIBILITY.md). |
| **Use the optional Isaac backend** | Follow the isolated [Isaac Lab setup and motion conversion](docs/ISAAC.md); see its current runtime validation scope. |
| **Compare trackers on additional motions** | Audit [motion inputs](docs/MOTION_STUDY.md), then prepare the [matched-budget study](docs/MOTION_STUDY_EXECUTION.md). |

GPU entry points use the compute allocation provided by your environment.
Device IDs are process-visible indices; launchers preserve inherited GPU
visibility. The core package does not install a CUDA runtime.

## Policies and Assets

Pretrained policies and their companion motion references are hosted on
Hugging Face, separate from the source code. **Start with one task:**

<p>
  <a href="https://huggingface.co/Shuaijun/MimicX-Policies/blob/main/recommended/README.md#tennis"><strong>Tennis</strong></a> &nbsp;&middot;&nbsp;
  <a href="https://huggingface.co/Shuaijun/MimicX-Policies/blob/main/recommended/README.md#football"><strong>Football</strong></a> &nbsp;&middot;&nbsp;
  <a href="https://huggingface.co/Shuaijun/MimicX-Policies/blob/main/recommended/README.md#dance"><strong>Dance</strong></a> &nbsp;&middot;&nbsp;
  <a href="https://huggingface.co/Shuaijun/MimicX-Policies/blob/main/recommended/README.md#kungfu"><strong>Kung Fu</strong></a>
</p>

| Release | Contents |
| :--- | :--- |
| [**MimicX-Policies**](https://huggingface.co/Shuaijun/MimicX-Policies) | Recommended final policies, multi-seed baselines, warmstarts, candidate checkpoints and training settings. |
| [**MimicX-Assets**](https://huggingface.co/datasets/Shuaijun/MimicX-Assets) | Matching robot references, public configurations, execution records and reproduction metadata. |

The [artifact usage guide](https://huggingface.co/Shuaijun/MimicX-Policies/blob/main/USAGE.md)
covers selective downloads and reconstruction of the exact core input bundle.
Pair each policy with its recorded reference and configuration: G1 29-DoF
and historical 23-DoF policies use different model layouts.

The [2026-10-14 demonstration release](docs/DEMONSTRATIONS.md) provides nine
standalone PPO policies and a separately identified Parkour hybrid controller.
Choose a task by name to download its policy and matching assets.
The original paper policies remain available separately for reproduction.

## Documentation

| Guide | Scope |
| :--- | :--- |
| [Installation](dependencies/README.md) | External dependencies, pinned backend setup and model requirements |
| [Video to Policy](docs/QUICKSTART.md) | Reconstruction, retargeting, warmstart training and AutoRefine |
| [Released Demonstrations](docs/DEMONSTRATIONS.md) | Task-based downloads and policy evaluation |
| [Continuation](docs/CONTINUATION.md) | Checkpoint recovery and terrain-aware policy continuation |
| [Recording and Contact](docs/RECORDING_AND_CONTACT.md) | Source-camera recording, baseline playback and contact-causal replay |
| [Hybrid Control](docs/HYBRID_CONTROL.md) | Parkour actor-prefix and predictive-controller replay |
| [Method](docs/METHOD.md) | Diagnosis, proposals, continuation and acceptance rules |
| [Paper Reproduction](docs/REPRODUCIBILITY.md) | Registered inputs, controlled comparisons and evaluation protocols |
| [HLoop](docs/HLOOP.md) | Heterogeneous execution, scheduling modes and parity checks |
| [Isaac Lab](docs/ISAAC.md) | Optional native simulator backend, named motion conversion and local-file policy entry points |
| [Motion Inputs](docs/MOTION_STUDY.md) | Prospective motion selection, source timing and kinematic validation |
| [Motion Study](docs/MOTION_STUDY_EXECUTION.md) | Shared warmstarts, controlled continuation budgets and repeated evaluation |
| [SONIC Evaluation](docs/SONIC_STUDY.md) | Released-controller playback, startup checks and common motion metrics |
| [Visualization](docs/VISUALIZATION.md) | Rollout videos, screenshots and exact-state replay |
| [Release Notes](docs/RELEASE.md) | Available components and tested workflows |

<details>
<summary><strong>Repository structure</strong></summary>

| Directory | Purpose |
| :--- | :--- |
| `mimicx/refinement/` | Diagnosis-conditioned search, verification and persistent state |
| `mimicx/runtime/` | Dependency-aware execution and parity checks |
| `mimicx/adapters/` | Motion-format bridge |
| `mimicx/backends/` | Optional simulator integrations, loaded only when selected |
| `mimicx/experiments/` | Matched motion-study execution and baseline evaluation |
| `mimicx/evaluation/` | Metric reduction and experiment evidence utilities |
| `mimicx/visualization/` | Replay, source-object registration, camera and geometry utilities |
| `dependencies/backend_overlay/` | Tracking backend modifications |
| `dependencies/assets/` | Attributed third-party visualization assets |
| `scripts/` | Setup, conversion, training-loop and export entry points |
| `mimicx/configs/paper/` | Paper experiment settings and input specifications |
| `docs/` | Project website and reproduction documentation |
| `tests/` | Portable CPU regression tests |

</details>

<details>
<summary><strong>Website and numerical data</strong></summary>

The [project website](https://nebulis-lab.github.io/MimicX/) presents the
method, paired policy videos and numerical evidence. Measured results live
only under `docs/assets/results/`, separate from implementation and runtime
inputs. The static page can also be opened from [docs/index.html](docs/index.html).
See [website maintenance](docs/WEBSITE.md) for rebuilding and publishing it.

</details>

## Citation

[Read the paper on arXiv](https://arxiv.org/abs/2610.09055).

```bibtex
@misc{liu2026mimicx,
  title={MimicX: Policy-in-the-Loop Supervision Refinement for Video-Driven Humanoid Motion Tracking},
  author={Shuaijun Liu and Chenglong Zhang and Xuhao Liu and Feiyang You and Yifan Liao and Shuyang Hao and Chaozhe Zhang and Chengyu Wu and Zhen Sun and Ningxin Su},
  year={2026},
  eprint={2610.09055},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2610.09055}
}
```

## Acknowledgments and License

MimicX builds on **GVHMR, GMR, SMPL-X, MuJoCo, MjLab and RSL-RL**.
Upstream projects, pinned dependencies and their notices are listed in
[dependencies](dependencies/README.md).

Code is released under [**Apache-2.0**](LICENSE). Included Unitree hand
visualization assets retain BSD-3-Clause, and the tracking backend retains
its upstream notices. Body models, source videos, datasets and third-party
weights follow their providers' terms; obtain the required authorized inputs
separately. Demonstration imagery and scene assets retain their respective
rights. Policy checkpoints are hosted on Hugging Face, not stored in Git.
