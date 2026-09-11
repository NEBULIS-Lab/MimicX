# MimicX

**Policy-in-the-Loop Supervision Refinement for Video-Driven Humanoid Motion Tracking**

MimicX turns reconstructed human motion into humanoid tracking policies and
uses execution feedback to refine their training supervision. It combines
failure-conditioned curricula and task-aware objectives with repeated-rollout
verification: a candidate replaces the current policy only when execution
improves and the tracking guard passes.

```text
Human video -> GVHMR / SMPL-X -> GMR -> MuJoCo motion reference
                                                |
                                      PPO tracking policy
                                                |
                         strict rollouts -> failure diagnosis
                                                |
                  objective / curriculum proposals -> continuation
                                                |
                           repeated verification -> accept / protect
```

## What Is Included

- The current single-human AutoRefine coordinator, bounded proposals, repeated
  execution gate, atomic state, checkpoint discovery, and restart support.
- Task-aware tracking objectives and curricula in an overlay for Unitree's
  MjLab backend, with G1 29-DoF and historical 23-DoF configurations.
- Video reconstruction and retargeting entry points, reference conversion,
  policy rollout telemetry, videos, screenshots, and exact-state replay export.
- MimicX-HLoop's CPU/GPU/I/O dependency scheduler, sequential and
  bulk-synchronous executors, and artifact/selection parity checks.
- Paper-specific controlled-comparison configurations and CPU tests.

The released automated search holds the registered reference motion fixed
within each loop and refines objectives and curricula. Input reference
preparation is a separate stage. See [the method](docs/METHOD.md) for the exact
implemented contract and the [project website](docs/index.html) for results.

## Start Here

```bash
git clone https://github.com/NEBULIS-Lab/MimicX.git
cd MimicX
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,visualization]'
python -m pytest -q
```

These tests are CPU-only, including deterministic subprocess fixtures for the
complete refinement cycle. They do not require body models or robot policies.

For actual training, install the pinned external backend and prepare your own
licensed motion inputs:

1. [Install dependencies](dependencies/README.md).
2. [Run video to reference to policy](docs/QUICKSTART.md).
3. [Run and resume AutoRefine](docs/QUICKSTART.md#autorefine).
4. [Export videos, screenshots, and replay states](docs/VISUALIZATION.md).
5. [Use HLoop and compare scheduling modes](docs/HLOOP.md).
6. [Reproduce the paper protocols](docs/REPRODUCIBILITY.md).

GPU entry points must run inside the compute allocation provided by your
environment. Device numbers refer to the process-visible GPU set; launchers
preserve the inherited visibility mask. Use `--devices 0` for a single visible
GPU. The lightweight package does not automatically install a CUDA runtime.

## Project Website

The self-contained [project page](docs/index.html) presents the method,
paired policy videos and complete numerical evidence. Results live only under
`docs/assets/results/`, separate from the implementation and runtime inputs.
See [website maintenance](docs/WEBSITE.md) for rebuilding and publishing it.

## Repository Map

| Directory | Purpose |
|---|---|
| `mimicx/refinement/` | Diagnosis-conditioned search, verification, persistence |
| `mimicx/runtime/` | Dependency-aware execution and parity checks |
| `mimicx/adapters/` | Motion-format bridge |
| `mimicx/evaluation/` | Metric reduction and experiment evidence utilities |
| `mimicx/visualization/` | Replay, source-object registration, camera/geometry utilities |
| `backend_overlay/` | Tracking backend modifications |
| `scripts/` | Setup, data conversion, training-loop and export entry points |
| `configs/paper/` | Frozen method settings and checksummed input specification |
| `docs/` | Project website and reproduction documentation |
| `tests/` | Portable CPU regression tests |

## Dependencies and License

MimicX's code is released under Apache-2.0. Included Unitree hand visualization
assets retain their BSD-3-Clause license. The tracking backend retains its
upstream notices. GVHMR, GMR, SMPL-X, MuJoCo, MjLab, and RSL-RL are acknowledged
in [dependencies](dependencies/README.md); their licenses remain separate.
Obtain body models, datasets, and pretrained weights from their original
providers. They are not bundled in this source release.

Policy checkpoints and source videos are not included in this Git snapshot.
Selected demonstration media are included only in the website. Exact paper
inputs are tracked by a hash manifest; their availability and the alternative
new-task workflow are documented in the [reproduction inventory](docs/REPRODUCIBILITY.md).
[Release verification](docs/RELEASE.md) records what was tested.
