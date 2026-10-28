# Source Release 0.4.0

## Current Release

The 2026-10-28 update adds an optional native Isaac Lab backend and a prospective
motion-study workflow. MuJoCo/MjLab remains the default; existing paper inputs,
published results, checkpoint releases and website media are unchanged.

- Isolated Isaac Sim 6 / Isaac Lab beta2 integration, named motion conversion,
  local-file training, checkpoint continuation and physical state/contact export.
- AutoRefine routing through actual Isaac rollout, diagnosis, candidate training
  and repeated execution verification.
- Motion input checks covering source timing, joint/body order, FK and a
  reference-relative evaluation protocol for upright and ground-contact motions.
- Shared native warmstarts, equal continuation-update budgets, checkpoint
  selection and held-out evaluation for the new motion study.
- Released SONIC evaluation with explicit startup-support release, event-clock
  alignment and a common canonical body metric.

Real Isaac verification includes G1 motion FK, training/save/reload/resume,
full-horizon state recording, all-body contact coverage and one physical
refinement cycle. A CPU protection replay uses those recorded physical outcomes.
The integration tests establish that the execution path works; the new
multi-task training study is separate and does not change paper results.
The optional environment is documented as experimental and is not asserted
to reproduce the older upstream simulator's collider or solver behavior.

The remote MuJoCo runtime also passed actual G1 PPO training, exact-action
checkpoint reload and resumed optimization. CPU tests, optional CPU-physics
checks, and package builds cover the new source interfaces.

See [Isaac Lab](ISAAC.md), [Motion Inputs](MOTION_STUDY.md),
[Motion Study](MOTION_STUDY_EXECUTION.md) and [SONIC Evaluation](SONIC_STUDY.md).
Simulator binaries, source motions, external weights and experiment outputs
are not included in this source release.

## Previous Release: 0.3.0

The 2026-10-14 source update integrates the later demonstration runtime while
preserving the original paper configurations and checkpoint archive:

- Task-based policy and asset downloads, validation, and evaluation.
- Portable PPO recovery, terrain continuation, atomic checkpoint recovery and
  independently recorded additional-update budgets.
- A separate `demonstrations` backend profile; the default `paper` overlay is
  unchanged.
- Named-joint and startup-force recording audits, source-camera replay, protected
  translucent references, and contact-causal ball replay.
- An explicitly separate Parkour actor-prefix/MPC controller and CPU state replay.
- Optional `artifacts` and `physics` dependencies, CPU physics CI and new guides.

See [Demonstrations](DEMONSTRATIONS.md), [Continuation](CONTINUATION.md),
[Recording and Contact](RECORDING_AND_CONTACT.md), and
[Hybrid Control](HYBRID_CONTROL.md) for runnable commands and verification scope.
Weights, motion recordings and numerical result archives remain outside Git.
No website media or paper content changes are part of this source update.

Verification on 2026-10-14 includes a clean Python3.11 install, a real selective
HF download, all nine paired artifact checks, and full-horizon physical evaluation
of all nine PPO checkpoints through the portable workers with zero early
terminations. The 12 original gate decisions reproduce exactly from raw archived
telemetry. Camera projection and encoded-video integration checks also pass.
The source distribution and wheel build successfully; the installed wheel reads
its packaged release configuration without a research checkout.
The complete CPU-plus-physics suite passes 293 tests; four tests requiring
external native assets or an explicitly enabled graphics backend are skipped.
The two graphics tests also pass separately in the configured renderer runtime.

Final review adds regression guards for motion/scene/profile content changes
during campaign reuse, the native termination-before-reference-advance order,
and body-origin linear velocities in recorded telemetry.

Physical runs preserve checkpoint/reference bytes and configuration, not
bit-identical floating-point state trajectories. Compared with the frozen HF
runner on the same seeds, the largest mean body-position difference is 0.000669 m
and the largest mean anchor-position difference is 0.005204 m; both executions
complete all nine horizons. Original published telemetry is retained unchanged.
Actual PPO training/resume and a fresh full six-reset Parkour MPC search are not
rerun by this packaging verification; their CPU contracts and recorded evidence
are tested as described in the linked guides.

## Previous Release: 0.2.0

This release packages the current single-human implementation, the tracking
backend overlay and the numerical evidence used by the manuscript. The native
research working directory is not required to run the released entry points.

## Included

- AutoRefine coordinator, diagnostics, proposals, continuation, repeated gate,
  state recovery and checkpoint discovery.
- PPO tracking overlay, video/retarget conversion interfaces, rollout and
  reference-only exports, and exact-state visualization utilities.
- HLoop and its execution baselines; controlled-comparison generation.
- Website numerical data with source records and all outcome rows.
- Frozen core task/method settings, an input inventory, and a fixed-policy
  HLoop benchmark plan generator.
- A portable CPU regression suite and CI configuration.

## Release Adaptations

Algorithmic proposal and verification rules are preserved. Packaging changes
replace workstation paths with arguments, preserve inherited GPU visibility,
serialize same-device candidate jobs, add portable setup/configuration tools,
make CLI help independent of GPU dependencies, and separate optional plotting
dependencies from the core. Historical internal method IDs remain in data;
public-facing names are documented in `assets/results/PROTOCOL.md`.

The backend requires the pinned upstream checkout plus this overlay. The
Python wheel installs the `mimicx` library; use a Git checkout for scripts,
backend overlays, tests, paper configurations and the website.

## Verification Scope

Verified on 2026-09-12:

| Check | Result |
|---|---|
| Python 3.10 CPU regression suite | 121 passed |
| Python 3.11 environment, editable install and CPU suite | 121 passed |
| Source distribution and wheel build | Passed |
| Pinned Unitree checkout and overlay installer | Passed |
| Source/artifact/credential audit | Passed |
| Core experiment inputs | 22 assets verified |
| Actual warmstart CPU shape checks | Four policies, 160 inputs / 29 actions |
| Frozen core trial preparation | 48 runnable job files generated |
| Fixed-policy HLoop preparation | 17 jobs materialized, no execution |
| Desktop/mobile browser verification | Four video pairs, table/downloads, no page errors or overflow |

The first dependency download timed out; the core installation succeeded on
retry. Visualization extras and the complete GPU dependency stack were not
fresh-installed as part of this pass.

The release test suite covers CPU unit tests, deterministic subprocess
execution of the complete candidate cycle, acceptance/protection, interrupted
training recovery, scheduler parity, motion conversion utilities, replay
serialization, object registration, and new setup/configuration entry points.
The dependency installer is additionally checked against a clean checkout of
the pinned Unitree backend.

No new GPU policy training, inference or paper experiment is claimed by this
packaging pass. Training weights and licensed input/body data are not
distributed in this Git snapshot. Selected existing research media are copied
to the website without altering their content. See the
[reproduction inventory](REPRODUCIBILITY.md) for exact input availability.
