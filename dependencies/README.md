# Dependencies

## Environments

Use Python 3.11 for the tracking environment. Keep GVHMR in its own environment
according to the pinned upstream installation instructions: its vision stack
has different dependencies from the tracking runtime.

```bash
python scripts/setup_dependencies.py unitree_rl_mjlab
python -m pip install -r dependencies/tracking.txt
python -m pip install -e third_party/unitree_rl_mjlab
python -m pip install -e '.[dev,visualization]'
```

`setup_dependencies.py` checks out the exact commit in `sources.json`, then
copies the tracked MimicX overlay into that new checkout. It refuses to
overwrite an existing directory. `--source /path/to/existing/git/clone` can
reuse an existing download. The overlay is required; plain upstream task
registrations do not contain MimicX's objectives and dynamic curricula.

The pinned Python packages are observed tracking-runtime versions. An exact
historical experiment also requires its motion, warmstart, task configuration,
seed and budget, not just package versions. A fresh GPU training run is not
part of this source-release verification.

## Video and Retargeting

```bash
python scripts/setup_dependencies.py GVHMR
python scripts/setup_dependencies.py GMR
```

Install GVHMR's dependencies and pretrained checkpoints using its own README
and `docs/INSTALL.md`. Install GMR in a retargeting environment using its
README (`pip install -e third_party/GMR`). Do not silently upgrade the frozen
tracking environment when adding optional vision tools.

Register for SMPL-X and download the neutral body model under its license.
The retargeter accepts `--body-models /path/to/body_models`; this directory
should contain the `smplx/` subdirectory expected by `smplx.create`.
Only load `.pt` and pickle files from trusted sources.

| Dependency | Role | Source / licensing |
|---|---|---|
| GVHMR | Video to global human motion | https://github.com/zju3dv/GVHMR; upstream research/non-commercial terms |
| SMPL-X | Parametric human body | https://smpl-x.is.tue.mpg.de/; separate body-model license |
| GMR | Human-to-robot retargeting | https://github.com/YanjieZe/GMR; MIT code, asset-specific terms |
| Unitree RL MjLab | Robot tracking task infrastructure | https://github.com/unitreerobotics/unitree_rl_mjlab; Apache-2.0 |
| MjLab | Vectorized MuJoCo learning | https://github.com/mujocolab/mjlab |
| MuJoCo / MuJoCo Warp | Dynamics and accelerated simulation | https://github.com/google-deepmind/mujoco; https://github.com/google-deepmind/mujoco_warp |
| RSL-RL | PPO implementation | https://github.com/leggedrobotics/rsl_rl |
| BeyondMimic | Whole-body tracking foundation | https://github.com/HybridRobotics/whole_body_tracking |

The repository does not vendor GVHMR/GMR model weights, SMPL-X models, video
datasets, or simulator binaries. See `sources.json` for pinned source commits
and `tracking.txt` for the runtime packages. The optional hand meshes under
`dependencies/assets/unitree_dex3_grip` carry their own license and attribution.
