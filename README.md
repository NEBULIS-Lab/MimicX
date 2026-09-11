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
- Controlled-comparison tooling, numerical result snapshots, and CPU tests.

The released automated search holds the registered reference motion fixed
within each loop and refines objectives and curricula. Input reference
preparation is a separate stage. See [the method](docs/METHOD.md) for the exact
implemented contract and [results](docs/RESULTS.md) for the evaluation protocol.

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

GPU entry points must run inside the compute allocation provided by your
environment. Device numbers refer to the process-visible GPU set; launchers
preserve the inherited visibility mask. Use `--devices 0` for a single visible
GPU. The lightweight package does not automatically install a CUDA runtime.

## Result Snapshot

The four-task controlled comparison contains 48 task/method/continuation-seed
trials, each with three evaluation rollouts. Selected findings:

| Metric | Fixed Reference | MimicX |
|---|---:|---:|
| Tennis strict success | 11.1% | 100.0% |
| Tennis aligned worst-body error, temporal mean | 0.241 m | 0.157 m |
| Football worst first-failure horizon, seed mean | 53.7 steps | 427.3 steps |

The twelve full-loop trials accepted six refinements and protected six current
policies. HLoop's median fixed-policy workload time was 124.017 s versus
241.221 s sequential and 122.258 s bulk-synchronous. Full task results,
component comparisons, reductions, and workload definitions are in
[docs/RESULTS.md](docs/RESULTS.md) and [benchmarks](benchmarks/README.md).

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
| `benchmarks/` | Numerical evidence without private filesystem metadata |
| `tests/` | Portable CPU regression tests |

## Dependencies and License

MimicX's code is released under Apache-2.0. Included Unitree hand visualization
assets retain their BSD-3-Clause license. The tracking backend retains its
upstream notices. GVHMR, GMR, SMPL-X, MuJoCo, MjLab, and RSL-RL are acknowledged
in [dependencies](dependencies/README.md); their licenses remain separate.
Obtain body models, datasets, and pretrained weights from their original
providers. They are not bundled in this source release.

Published policy checkpoints and source videos are not included in this Git
snapshot. The quickstart documents training a task-specific warmstart and
refining it. [Release verification](docs/RELEASE.md) records what was tested.
