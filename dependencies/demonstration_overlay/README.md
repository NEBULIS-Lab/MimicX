# Demonstration Backend Profile

These tracking modules are the frozen source carried by the paired
2026-10-14 demonstration artifact release. They build on the Unitree RL MjLab
tracking backend under Apache-2.0; the repository NOTICE also applies here.
The SHA-256 inventory of the original runtime is pinned in
`mimicx/configs/releases/demonstrations.json`.

Select this profile explicitly with `scripts/setup_dependencies.py
unitree_rl_mjlab --profile demonstrations`. It applies the original source
overlay first, then these task/runtime modules to a fresh pinned checkout.
The default `paper` profile and its source files remain unchanged.
