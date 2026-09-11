# MimicX-modified tracking overlay; see NOTICE for attribution.
from mjlab.tasks.registry import register_mjlab_task
from src.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import (
  unitree_g1_mimicx_autorefine_arm_contact_balanced_curriculum_env_cfg,
  unitree_g1_mimicx_autorefine_core_contact_balanced_curriculum_env_cfg,
  unitree_g1_mimicx_autorefine_dynamic_curriculum_env_cfg,
  unitree_g1_mimicx_autorefine_hierarchical_gail_curriculum_env_cfg,
  unitree_g1_mimicx_amass_residual_full_start_curriculum_env_cfg,
  unitree_g1_mimicx_amass_residual_late_window_curriculum_env_cfg,
  unitree_g1_mimicx_amass_residual_multi_upright_curriculum_env_cfg,
  unitree_g1_mimicx_amass_residual_start_window_curriculum_env_cfg,
  unitree_g1_mimicx_autorefine_hierarchical_balanced_curriculum_env_cfg,
  unitree_g1_mimicx_arm_ee_focus_curriculum_env_cfg,
  unitree_g1_flat_tracking_env_cfg,
  unitree_g1_mimicx_contact_phase_strong_curriculum_env_cfg,
  unitree_g1_mimicx_contact_phase_light_curriculum_env_cfg,
  unitree_g1_mimicx_dynamics_smooth_curriculum_env_cfg,
  unitree_g1_mimicx_hybrid_contact_dynamics_curriculum_env_cfg,
  unitree_g1_mimicx_kongfu1_failure_window_curriculum_env_cfg,
  unitree_g1_mimicx_kongfu1_full_start_consolidation_curriculum_env_cfg,
  unitree_g1_mimicx_kongfu1_late_anchor_window_curriculum_env_cfg,
  unitree_g1_mimicx_kongfu1_right_leg_z_window_curriculum_env_cfg,
  unitree_g1_mimicx_lower_body_window_guard_curriculum_env_cfg,
  unitree_g1_mimicx_curriculum_env_cfg,
  unitree_g1_mimicx_root_curriculum_env_cfg,
  unitree_g1_mimicx_start_curriculum_env_cfg,
  unitree_g1_mimicx_start_root_curriculum_env_cfg,
  unitree_g1_mimicx_start_window_anchor_guard_curriculum_env_cfg,
  unitree_g1_mimicx_start_window_overlap_curriculum_env_cfg,
)
from .rl_cfg import (
  unitree_g1_tracking_ppo_runner_cfg,
  unitree_g1_tracking_residual_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Unitree-G1-Tracking",
  env_cfg=unitree_g1_flat_tracking_env_cfg(),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Tracking-No-State-Estimation",
  env_cfg=unitree_g1_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Tracking-MimicX-Curriculum",
  env_cfg=unitree_g1_mimicx_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_mimicx_curriculum_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Tracking-MimicX-Start-Curriculum",
  env_cfg=unitree_g1_mimicx_start_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_mimicx_start_curriculum_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Tracking-MimicX-Root-Curriculum",
  env_cfg=unitree_g1_mimicx_root_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_mimicx_root_curriculum_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Tracking-MimicX-Start-Root-Curriculum",
  env_cfg=unitree_g1_mimicx_start_root_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_mimicx_start_root_curriculum_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Tracking-MimicX-AMASS-Residual-Start-Window-Curriculum",
  env_cfg=unitree_g1_mimicx_amass_residual_start_window_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_mimicx_amass_residual_start_window_curriculum_env_cfg(
    play=True
  ),
  rl_cfg=unitree_g1_tracking_residual_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Tracking-MimicX-AMASS-Residual-Full-Start-Curriculum",
  env_cfg=unitree_g1_mimicx_amass_residual_full_start_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_mimicx_amass_residual_full_start_curriculum_env_cfg(
    play=True
  ),
  rl_cfg=unitree_g1_tracking_residual_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Tracking-MimicX-AMASS-Residual-Late-Window-Curriculum",
  env_cfg=unitree_g1_mimicx_amass_residual_late_window_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_mimicx_amass_residual_late_window_curriculum_env_cfg(
    play=True
  ),
  rl_cfg=unitree_g1_tracking_residual_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Tracking-MimicX-AMASS-Residual-Multi-Upright-Curriculum",
  env_cfg=unitree_g1_mimicx_amass_residual_multi_upright_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_mimicx_amass_residual_multi_upright_curriculum_env_cfg(
    play=True
  ),
  rl_cfg=unitree_g1_tracking_residual_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Tracking-MimicX-Start-Window-Overlap-Curriculum",
  env_cfg=unitree_g1_mimicx_start_window_overlap_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_mimicx_start_window_overlap_curriculum_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Tracking-MimicX-Start-Window-Anchor-Guard-Curriculum",
  env_cfg=unitree_g1_mimicx_start_window_anchor_guard_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_mimicx_start_window_anchor_guard_curriculum_env_cfg(
    play=True
  ),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Tracking-MimicX-Lower-Body-Window-Guard-Curriculum",
  env_cfg=unitree_g1_mimicx_lower_body_window_guard_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_mimicx_lower_body_window_guard_curriculum_env_cfg(
    play=True
  ),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Tracking-MimicX-Contact-Phase-Light-Curriculum",
  env_cfg=unitree_g1_mimicx_contact_phase_light_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_mimicx_contact_phase_light_curriculum_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Tracking-MimicX-Dynamics-Smooth-Curriculum",
  env_cfg=unitree_g1_mimicx_dynamics_smooth_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_mimicx_dynamics_smooth_curriculum_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Tracking-MimicX-Contact-Phase-Strong-Curriculum",
  env_cfg=unitree_g1_mimicx_contact_phase_strong_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_mimicx_contact_phase_strong_curriculum_env_cfg(
    play=True
  ),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Tracking-MimicX-Hybrid-Contact-Dynamics-Curriculum",
  env_cfg=unitree_g1_mimicx_hybrid_contact_dynamics_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_mimicx_hybrid_contact_dynamics_curriculum_env_cfg(
    play=True
  ),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Tracking-MimicX-Arm-EE-Focus-Curriculum",
  env_cfg=unitree_g1_mimicx_arm_ee_focus_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_mimicx_arm_ee_focus_curriculum_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Tracking-MimicX-AutoRefine-Arm-Contact-Balanced-Curriculum",
  env_cfg=unitree_g1_mimicx_autorefine_arm_contact_balanced_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_mimicx_autorefine_arm_contact_balanced_curriculum_env_cfg(
    play=True
  ),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Tracking-MimicX-AutoRefine-Core-Contact-Balanced-Curriculum",
  env_cfg=unitree_g1_mimicx_autorefine_core_contact_balanced_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_mimicx_autorefine_core_contact_balanced_curriculum_env_cfg(
    play=True
  ),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Tracking-MimicX-AutoRefine-Hierarchical-Balanced-Curriculum",
  env_cfg=unitree_g1_mimicx_autorefine_hierarchical_balanced_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_mimicx_autorefine_hierarchical_balanced_curriculum_env_cfg(
    play=True
  ),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Tracking-MimicX-AutoRefine-Hierarchical-GAIL-Curriculum",
  env_cfg=unitree_g1_mimicx_autorefine_hierarchical_gail_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_mimicx_autorefine_hierarchical_gail_curriculum_env_cfg(
    play=True
  ),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Tracking-MimicX-AutoRefine-Dynamic-Curriculum",
  env_cfg=unitree_g1_mimicx_autorefine_dynamic_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_mimicx_autorefine_dynamic_curriculum_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Tracking-MimicX-Kongfu1-Failure-Window-Curriculum",
  env_cfg=unitree_g1_mimicx_kongfu1_failure_window_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_mimicx_kongfu1_failure_window_curriculum_env_cfg(
    play=True
  ),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Tracking-MimicX-Kongfu1-Right-Leg-Z-Window-Curriculum",
  env_cfg=unitree_g1_mimicx_kongfu1_right_leg_z_window_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_mimicx_kongfu1_right_leg_z_window_curriculum_env_cfg(
    play=True
  ),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Tracking-MimicX-Kongfu1-Late-Anchor-Window-Curriculum",
  env_cfg=unitree_g1_mimicx_kongfu1_late_anchor_window_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_mimicx_kongfu1_late_anchor_window_curriculum_env_cfg(
    play=True
  ),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Tracking-MimicX-Kongfu1-Full-Start-Consolidation-Curriculum",
  env_cfg=unitree_g1_mimicx_kongfu1_full_start_consolidation_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_mimicx_kongfu1_full_start_consolidation_curriculum_env_cfg(
    play=True
  ),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
