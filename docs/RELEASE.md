# Source Release 0.2.0

This release packages the current single-human implementation, the tracking
backend overlay and the numerical evidence used by the manuscript. The native
research working directory is not required to run the released entry points.

## Included

- AutoRefine coordinator, diagnostics, proposals, continuation, repeated gate,
  state recovery and checkpoint discovery.
- PPO tracking overlay, video/retarget conversion interfaces, rollout and
  reference-only exports, and exact-state visualization utilities.
- HLoop and its execution baselines; controlled-comparison generation.
- Nine numerical CSVs with source/published checksums and all outcome rows.
- A portable CPU regression suite and CI configuration.

## Release Adaptations

Algorithmic proposal and verification rules are preserved. Packaging changes
replace workstation paths with arguments, preserve inherited GPU visibility,
serialize same-device candidate jobs, add portable setup/configuration tools,
make CLI help independent of GPU dependencies, and separate optional plotting
dependencies from the core. Historical internal method IDs remain in data;
public-facing names are documented in `RESULTS.md`.

The backend requires the pinned upstream checkout plus this overlay. The
Python wheel installs the `mimicx` library; use a Git checkout for scripts,
backend overlays, tests and benchmark CSVs.

## Verification Scope

Verified on 2026-09-12:

| Check | Result |
|---|---|
| Python 3.10 CPU regression suite | 114 passed |
| Fresh Python 3.11 environment, editable install and CPU suite | 114 passed |
| Source distribution and wheel build | Passed |
| Pinned Unitree checkout and overlay installer | Passed |
| Source/artifact/credential audit | Passed |

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
packaging pass. Training weights, licensed input/body data and high-resolution
paper media are not distributed in this Git snapshot. Their absence does not
turn the CPU fixture into a physical-policy evaluation.
