# MimicX-modified tracking overlay; see NOTICE for attribution.
from mjlab.tasks.registry import register_mjlab_task
from src.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import (
  unitree_g1_23dof_flat_tracking_env_cfg,
  unitree_g1_23dof_mimicx_adaptive_feet_lock_curriculum_env_cfg,
  unitree_g1_23dof_mimicx_curriculum_env_cfg,
  unitree_g1_23dof_mimicx_start_window_feet_z_curriculum_env_cfg,
  unitree_g1_23dof_mimicx_start_window_lower_body_curriculum_env_cfg,
  unitree_g1_23dof_mimicx_start_root_curriculum_env_cfg,
  unitree_g1_23dof_mimicx_start_root_feet_lock_curriculum_env_cfg,
  unitree_g1_23dof_mimicx_start_root_orient_curriculum_env_cfg,
)
from .rl_cfg import unitree_g1_23dof_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-G1-23Dof-Tracking",
  env_cfg=unitree_g1_23dof_flat_tracking_env_cfg(),
  play_env_cfg=unitree_g1_23dof_flat_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_23dof_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-23Dof-Tracking-No-State-Estimation",
  env_cfg=unitree_g1_23dof_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=unitree_g1_23dof_flat_tracking_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=unitree_g1_23dof_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-23Dof-Tracking-MimicX-Curriculum",
  env_cfg=unitree_g1_23dof_mimicx_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_23dof_mimicx_curriculum_env_cfg(play=True),
  rl_cfg=unitree_g1_23dof_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-23Dof-Tracking-MimicX-Start-Root-Curriculum",
  env_cfg=unitree_g1_23dof_mimicx_start_root_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_23dof_mimicx_start_root_curriculum_env_cfg(play=True),
  rl_cfg=unitree_g1_23dof_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-23Dof-Tracking-MimicX-Start-Root-Orient-Curriculum",
  env_cfg=unitree_g1_23dof_mimicx_start_root_orient_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_23dof_mimicx_start_root_orient_curriculum_env_cfg(play=True),
  rl_cfg=unitree_g1_23dof_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-23Dof-Tracking-MimicX-Start-Root-Feet-Lock-Curriculum",
  env_cfg=unitree_g1_23dof_mimicx_start_root_feet_lock_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_23dof_mimicx_start_root_feet_lock_curriculum_env_cfg(play=True),
  rl_cfg=unitree_g1_23dof_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-23Dof-Tracking-MimicX-Adaptive-Feet-Lock-Curriculum",
  env_cfg=unitree_g1_23dof_mimicx_adaptive_feet_lock_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_23dof_mimicx_adaptive_feet_lock_curriculum_env_cfg(play=True),
  rl_cfg=unitree_g1_23dof_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-23Dof-Tracking-MimicX-Start-Window-Lower-Body-Curriculum",
  env_cfg=unitree_g1_23dof_mimicx_start_window_lower_body_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_23dof_mimicx_start_window_lower_body_curriculum_env_cfg(play=True),
  rl_cfg=unitree_g1_23dof_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-23Dof-Tracking-MimicX-Start-Window-Feet-Z-Curriculum",
  env_cfg=unitree_g1_23dof_mimicx_start_window_feet_z_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_23dof_mimicx_start_window_feet_z_curriculum_env_cfg(play=True),
  rl_cfg=unitree_g1_23dof_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
