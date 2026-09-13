# Tracking Overlay

The Python files under `unitree_rl_mjlab/` contain MimicX's modifications to
the pinned Apache-2.0 Unitree backend. Install them through
`scripts/setup_dependencies.py unitree_rl_mjlab` from the repository root.
The checkout retains upstream assets, package initialization and licenses.

Included modifications: tracking telemetry, task-aware rewards, failure-window
curricula, reference-relative actions, dynamic JSON patch loading, G1/23-DoF
task configurations, checkpoint/export integration, and training/conversion
entry points. Historical task variants are retained for reproducibility; use
the dynamic curriculum and strict task named in the Quickstart for AutoRefine.

The source release also makes runtime placement portable: training and rollout
devices are logical indices within the inherited visible-device set. Updating
this overlay does not modify any installed upstream checkout until the user
explicitly installs it.
