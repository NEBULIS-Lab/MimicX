# Unitree G1 Dex3 Visualization Assets

These files are used only to render the G1 right hand in a closed power-grasp
configuration for paper figures and videos.

- Source MJCF: Unitree Robotics, `unitree_rl_gym/resources/robots/g1_description/g1_29dof_with_hand_rev_1_0.xml`
- Mesh source: the matching Unitree G1 Dex3 meshes mirrored by LATENT
- License: Unitree BSD 3-Clause License, reproduced in `LICENSE`
- MJCF SHA-256: `77c98e7d34e428c6cdbd1e0d665e989cbc43bf17abb8a7c048494b4bc51ead88`

The rendered joint targets are the public G1 power-grasp targets used by the
Lucky Robots G1 manipulation challenge. They affect visualization only and do
not alter policy observations, actions, rewards, or checkpoints.

Only right-hand meshes are included. The XML is used as geometry/kinematics
input to `load_closed_grip_visuals`, not as a complete standalone simulator
model. Obtain the full upstream robot asset pack to load the entire MJCF.
