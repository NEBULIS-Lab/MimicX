# MimicX-modified tracking overlay; see NOTICE for attribution.
"""Unitree G1 flat tracking environment configurations."""

import copy
import json
import os
from pathlib import Path

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from src.tasks.tracking import mdp as mimicx_mdp
from src.tasks.tracking.mdp import MotionCommandCfg, MotionResidualJointPositionActionCfg
from src.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def unitree_g1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  cfg.viewer.body_name = "torso_link"

  # Modify observations if we don't have state estimation.
  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg


def unitree_g1_mimicx_curriculum_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create a lower-randomization G1 tracking task for MimicX reference bring-up."""
  cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=True, play=play)

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.pose_range = {}
  motion_cmd.velocity_range = {}
  motion_cmd.joint_position_range = (0.0, 0.0)

  cfg.observations["actor"].enable_corruption = False

  cfg.events.pop("push_robot", None)
  cfg.events.pop("base_com", None)
  cfg.events.pop("encoder_bias", None)
  cfg.events.pop("foot_friction", None)

  cfg.terminations["anchor_pos"].params["threshold"] = 0.35
  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  )

  cfg.rewards["motion_global_root_pos"].weight = 1.0
  cfg.rewards["motion_body_pos"].params["std"] = 0.4
  cfg.rewards["motion_body_ori"].params["std"] = 0.5
  cfg.rewards["action_rate_l2"].weight = -5.0e-2
  cfg.sim.nconmax = 96
  cfg.sim.njmax = 400

  return cfg


def unitree_g1_mimicx_start_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a start-sampled MimicX G1 tracking curriculum."""
  cfg = unitree_g1_mimicx_curriculum_env_cfg(play=play)
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.sampling_mode = "start"
  return cfg


def unitree_g1_mimicx_root_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a MimicX G1 curriculum with stronger root and feet tracking."""
  cfg = unitree_g1_mimicx_curriculum_env_cfg(play=play)

  cfg.rewards["motion_global_root_pos"].weight = 3.0
  cfg.rewards["motion_global_root_pos"].params["std"] = 0.2

  feet_pos_reward = copy.deepcopy(cfg.rewards["motion_body_pos"])
  feet_pos_reward.weight = 2.0
  feet_pos_reward.params = {
    **feet_pos_reward.params,
    "std": 0.15,
    "body_names": ("left_ankle_roll_link", "right_ankle_roll_link"),
  }
  cfg.rewards["motion_feet_pos"] = feet_pos_reward

  return cfg


def unitree_g1_mimicx_start_root_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a start-sampled root/feet MimicX G1 curriculum."""
  cfg = unitree_g1_mimicx_root_curriculum_env_cfg(play=play)
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.sampling_mode = "start"
  return cfg


def _use_motion_residual_joint_actions(
  cfg: ManagerBasedRlEnvCfg,
  *,
  scale: float = 0.16,
) -> None:
  cfg.actions["joint_pos"] = MotionResidualJointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=scale,
    use_default_offset=False,
    command_name="motion",
  )


def unitree_g1_mimicx_amass_residual_start_window_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Residual-action AMASS bring-up task with early failure-window replay."""
  cfg = unitree_g1_mimicx_autorefine_hierarchical_balanced_curriculum_env_cfg(
    play=play
  )
  _use_motion_residual_joint_actions(cfg, scale=0.16)

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.joint_position_range = (0.0, 0.0)
  if not play:
    motion_cmd.sampling_mode = "start_window"
    motion_cmd.start_windows = (
      (0, 70, 0.20),
      (35, 115, 0.35),
      (80, 170, 0.25),
      (140, 260, 0.10),
    )
    motion_cmd.start_window_start = 35
    motion_cmd.start_window_end = 115
    motion_cmd.start_window_ratio = 0.90
  else:
    motion_cmd.sampling_mode = "start"

  cfg.terminations["anchor_pos"].params["threshold"] = 0.50
  cfg.terminations["anchor_ori"].params["threshold"] = 1.20
  cfg.terminations["ee_body_pos"].params["threshold"] = 0.35

  cfg.rewards["action_rate_l2"].weight = -4.0e-2
  cfg.rewards["motion_global_root_pos"].weight = 4.0
  cfg.rewards["motion_global_root_pos"].params["std"] = 0.18
  cfg.rewards["motion_global_root_ori"].weight = 1.0
  cfg.rewards["motion_global_root_ori"].params["std"] = 0.32
  cfg.rewards["motion_body_pos"].params["std"] = 0.28
  cfg.rewards["motion_body_ori"].params["std"] = 0.36
  cfg.rewards["motion_body_lin_vel"].params["std"] = 0.85
  cfg.rewards["motion_body_ang_vel"].params["std"] = 2.6
  if "motion_feet_pos" in cfg.rewards:
    cfg.rewards["motion_feet_pos"].weight = 2.6
    cfg.rewards["motion_feet_pos"].params["std"] = 0.13

  return cfg


def unitree_g1_mimicx_amass_residual_full_start_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Residual-action AMASS consolidation task evaluated from frame zero."""
  cfg = unitree_g1_mimicx_amass_residual_start_window_curriculum_env_cfg(play=play)
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.sampling_mode = "start"
  motion_cmd.start_window_ratio = 0.0
  motion_cmd.start_windows = ()
  return cfg


def unitree_g1_mimicx_amass_residual_late_window_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Residual-action AMASS task focused on the late full-start failure window."""
  cfg = unitree_g1_mimicx_amass_residual_start_window_curriculum_env_cfg(play=play)
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  if not play:
    motion_cmd.sampling_mode = "start_window"
    motion_cmd.start_windows = (
      (150, 230, 0.20),
      (190, 285, 0.30),
      (225, 309, 0.30),
    )
    motion_cmd.start_window_start = 190
    motion_cmd.start_window_end = 285
    motion_cmd.start_window_ratio = 0.80
  return cfg


def _mimicx_motion_path(relative_path: str) -> str:
  if os.environ.get("MIMICX_ROOT"):
    return str(Path(os.environ["MIMICX_ROOT"]).expanduser() / relative_path)
  cwd = Path.cwd()
  if (cwd / "data" / "amass_g1").exists():
    repo_root = cwd
  else:
    repo_root = Path(__file__).resolve().parents[7]
  return str(repo_root / relative_path)


def unitree_g1_mimicx_amass_residual_multi_upright_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Residual-action AMASS task that samples several upright/general clips."""
  cfg = unitree_g1_mimicx_amass_residual_full_start_curriculum_env_cfg(play=play)

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_files = (
    _mimicx_motion_path(
      "data/amass_g1/mjlab_npz/Male1General_c3d/General_A1_-_Stand_stageii_g1.npz"
    ),
    _mimicx_motion_path(
      "data/amass_g1/mjlab_npz/Male1General_c3d/General_A2_-_Sway_stageii_g1.npz"
    ),
    _mimicx_motion_path(
      "data/amass_g1/mjlab_npz/Male1General_c3d/General_A3_-_Swing_Arms_While_Stand_stageii_g1.npz"
    ),
    _mimicx_motion_path(
      "data/amass_g1/mjlab_npz/Male1General_c3d/General_A4_-_Look_Around_stageii_g1.npz"
    ),
    _mimicx_motion_path(
      "data/amass_g1/mjlab_npz/Male1General_c3d/General_A5_-_Pick_Up_Box_stageii_g1.npz"
    ),
    _mimicx_motion_path(
      "data/amass_g1/mjlab_npz/Male1General_c3d/General_A6_-_Lift_Box_stageii_g1.npz"
    ),
    _mimicx_motion_path(
      "data/amass_g1/mjlab_npz/Male1General_c3d/General_A7_-_Crouch_stageii_g1.npz"
    ),
    _mimicx_motion_path(
      "data/amass_g1/mjlab_npz/Female1Walking_c3d/B1_-_stand_to_walk_stageii_g1.npz"
    ),
  )
  motion_cmd.motion_file = motion_cmd.motion_files[0]
  if not play:
    motion_cmd.sampling_mode = "uniform"
  else:
    motion_cmd.sampling_mode = "start"

  cfg.rewards["motion_global_root_pos"].params["std"] = 0.22
  cfg.rewards["motion_body_pos"].params["std"] = 0.32
  cfg.rewards["motion_body_ori"].params["std"] = 0.42
  cfg.rewards["motion_body_lin_vel"].params["std"] = 0.95
  cfg.rewards["motion_body_ang_vel"].params["std"] = 2.8
  cfg.rewards["action_rate_l2"].weight = -3.5e-2

  return cfg


def unitree_g1_mimicx_start_window_overlap_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a tennis overlap curriculum focused on the audited wrist/ankle window."""
  cfg = unitree_g1_mimicx_start_root_curriculum_env_cfg(play=play)

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  if not play:
    motion_cmd.sampling_mode = "start_window"
    motion_cmd.start_window_start = 360
    motion_cmd.start_window_end = 430
    motion_cmd.start_window_ratio = 0.20

  wrist_body_names = ("left_wrist_yaw_link", "right_wrist_yaw_link")
  arm_body_names = (
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )
  ankle_body_names = ("left_ankle_roll_link", "right_ankle_roll_link")
  lower_body_names = (
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
  )

  cfg.rewards["motion_feet_pos"].weight = 3.0
  cfg.rewards["motion_feet_pos"].params["std"] = 0.12

  wrist_pos_reward = copy.deepcopy(cfg.rewards["motion_body_pos"])
  wrist_pos_reward.weight = 3.0
  wrist_pos_reward.params = {
    **wrist_pos_reward.params,
    "std": 0.10,
    "body_names": wrist_body_names,
  }
  cfg.rewards["motion_wrist_pos"] = wrist_pos_reward

  arm_pos_reward = copy.deepcopy(cfg.rewards["motion_body_pos"])
  arm_pos_reward.weight = 1.5
  arm_pos_reward.params = {
    **arm_pos_reward.params,
    "std": 0.16,
    "body_names": arm_body_names,
  }
  cfg.rewards["motion_arm_pos"] = arm_pos_reward

  lower_body_pos_reward = copy.deepcopy(cfg.rewards["motion_body_pos"])
  lower_body_pos_reward.weight = 1.0
  lower_body_pos_reward.params = {
    **lower_body_pos_reward.params,
    "std": 0.16,
    "body_names": lower_body_names,
  }
  cfg.rewards["motion_lower_body_pos"] = lower_body_pos_reward

  wrist_lin_vel_reward = copy.deepcopy(cfg.rewards["motion_body_lin_vel"])
  wrist_lin_vel_reward.weight = 0.75
  wrist_lin_vel_reward.params = {
    **wrist_lin_vel_reward.params,
    "std": 0.7,
    "body_names": wrist_body_names,
  }
  cfg.rewards["motion_wrist_lin_vel"] = wrist_lin_vel_reward

  ankle_lin_vel_reward = copy.deepcopy(cfg.rewards["motion_body_lin_vel"])
  ankle_lin_vel_reward.weight = 0.75
  ankle_lin_vel_reward.params = {
    **ankle_lin_vel_reward.params,
    "std": 0.7,
    "body_names": ankle_body_names,
  }
  cfg.rewards["motion_ankle_lin_vel"] = ankle_lin_vel_reward

  return cfg


def unitree_g1_mimicx_start_window_anchor_guard_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a tennis overlap curriculum with stronger root/torso anchoring."""
  cfg = unitree_g1_mimicx_start_window_overlap_curriculum_env_cfg(play=play)

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  if not play:
    motion_cmd.start_window_ratio = 0.12

  core_body_names = ("pelvis", "torso_link")
  lower_body_names = (
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
  )

  cfg.rewards["motion_global_root_pos"].weight = 4.5
  cfg.rewards["motion_global_root_pos"].params["std"] = 0.16
  cfg.rewards["motion_global_root_ori"].weight = 1.0
  cfg.rewards["motion_global_root_ori"].params["std"] = 0.3
  cfg.rewards["motion_body_pos"].params["std"] = 0.28
  cfg.rewards["motion_body_ori"].params["std"] = 0.35

  core_pos_reward = copy.deepcopy(cfg.rewards["motion_body_pos"])
  core_pos_reward.weight = 2.0
  core_pos_reward.params = {
    **core_pos_reward.params,
    "std": 0.12,
    "body_names": core_body_names,
  }
  cfg.rewards["motion_core_pos"] = core_pos_reward

  lower_body_lin_vel_reward = copy.deepcopy(cfg.rewards["motion_body_lin_vel"])
  lower_body_lin_vel_reward.weight = 0.5
  lower_body_lin_vel_reward.params = {
    **lower_body_lin_vel_reward.params,
    "std": 0.8,
    "body_names": lower_body_names,
  }
  cfg.rewards["motion_lower_body_lin_vel"] = lower_body_lin_vel_reward

  return cfg


def unitree_g1_mimicx_lower_body_window_guard_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a tennis anchor-guard curriculum focused on ankle/knee window overlap."""
  cfg = unitree_g1_mimicx_start_window_anchor_guard_curriculum_env_cfg(play=play)

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  if not play:
    motion_cmd.start_window_start = 370
    motion_cmd.start_window_end = 420
    motion_cmd.start_window_ratio = 0.10

  ankle_body_names = ("left_ankle_roll_link", "right_ankle_roll_link")
  knee_body_names = ("left_knee_link", "right_knee_link")
  ankle_knee_body_names = (
    "left_knee_link",
    "left_ankle_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
  )
  lower_body_names = (
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
  )

  cfg.rewards["motion_lower_body_pos"].weight = 1.6
  cfg.rewards["motion_lower_body_pos"].params["std"] = 0.14
  cfg.rewards["motion_lower_body_lin_vel"].weight = 0.8
  cfg.rewards["motion_lower_body_lin_vel"].params["std"] = 0.65

  ankle_knee_pos_reward = copy.deepcopy(cfg.rewards["motion_body_pos"])
  ankle_knee_pos_reward.weight = 2.0
  ankle_knee_pos_reward.params = {
    **ankle_knee_pos_reward.params,
    "std": 0.10,
    "body_names": ankle_knee_body_names,
  }
  cfg.rewards["motion_ankle_knee_pos"] = ankle_knee_pos_reward

  ankle_z_reward = copy.deepcopy(cfg.rewards["motion_feet_pos"])
  ankle_z_reward.func = mimicx_mdp.motion_relative_body_position_z_error_exp
  ankle_z_reward.weight = 1.2
  ankle_z_reward.params = {
    "command_name": "motion",
    "std": 0.055,
    "body_names": ankle_body_names,
  }
  cfg.rewards["motion_ankle_z"] = ankle_z_reward

  knee_z_reward = copy.deepcopy(cfg.rewards["motion_feet_pos"])
  knee_z_reward.func = mimicx_mdp.motion_relative_body_position_z_error_exp
  knee_z_reward.weight = 0.8
  knee_z_reward.params = {
    "command_name": "motion",
    "std": 0.08,
    "body_names": knee_body_names,
  }
  cfg.rewards["motion_knee_z"] = knee_z_reward

  lower_body_ang_vel_reward = copy.deepcopy(cfg.rewards["motion_body_ang_vel"])
  lower_body_ang_vel_reward.weight = 0.4
  lower_body_ang_vel_reward.params = {
    **lower_body_ang_vel_reward.params,
    "std": 2.0,
    "body_names": lower_body_names,
  }
  cfg.rewards["motion_lower_body_ang_vel"] = lower_body_ang_vel_reward

  return cfg


def unitree_g1_mimicx_contact_phase_light_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a light contact-phase hypothesis test from the anchor-guard task."""
  cfg = unitree_g1_mimicx_start_window_anchor_guard_curriculum_env_cfg(play=play)

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  if not play:
    motion_cmd.start_window_start = 370
    motion_cmd.start_window_end = 420
    motion_cmd.start_window_ratio = 0.08

  ankle_body_names = ("left_ankle_roll_link", "right_ankle_roll_link")

  ankle_z_reward = copy.deepcopy(cfg.rewards["motion_feet_pos"])
  ankle_z_reward.func = mimicx_mdp.motion_relative_body_position_z_error_exp
  ankle_z_reward.weight = 0.6
  ankle_z_reward.params = {
    "command_name": "motion",
    "std": 0.07,
    "body_names": ankle_body_names,
  }
  cfg.rewards["motion_ankle_z_light"] = ankle_z_reward

  stance_slip_reward = copy.deepcopy(cfg.rewards["motion_body_lin_vel"])
  stance_slip_reward.func = mimicx_mdp.motion_reference_stance_foot_slip_error_exp
  stance_slip_reward.weight = 0.8
  stance_slip_reward.params = {
    "command_name": "motion",
    "std": 0.35,
    "stance_height": 0.085,
    "body_names": ankle_body_names,
    "vertical_weight": 0.25,
  }
  cfg.rewards["motion_reference_stance_slip"] = stance_slip_reward

  return cfg


def unitree_g1_mimicx_dynamics_smooth_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a dynamics/smoothness hypothesis test from the anchor-guard task."""
  cfg = unitree_g1_mimicx_start_window_anchor_guard_curriculum_env_cfg(play=play)

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  if not play:
    motion_cmd.start_window_start = 360
    motion_cmd.start_window_end = 430
    motion_cmd.start_window_ratio = 0.08

  core_body_names = ("pelvis", "torso_link")
  cfg.rewards["action_rate_l2"].weight = -8.0e-2
  cfg.rewards["motion_body_lin_vel"].params["std"] = 0.85
  cfg.rewards["motion_body_ang_vel"].params["std"] = 2.6

  core_ori_reward = copy.deepcopy(cfg.rewards["motion_body_ori"])
  core_ori_reward.weight = 1.0
  core_ori_reward.params = {
    **core_ori_reward.params,
    "std": 0.28,
    "body_names": core_body_names,
  }
  cfg.rewards["motion_core_ori"] = core_ori_reward

  torso_lin_vel_reward = copy.deepcopy(cfg.rewards["motion_body_lin_vel"])
  torso_lin_vel_reward.weight = 1.0
  torso_lin_vel_reward.params = {
    **torso_lin_vel_reward.params,
    "std": 0.55,
    "body_names": ("torso_link",),
  }
  cfg.rewards["motion_torso_lin_vel"] = torso_lin_vel_reward

  torso_ang_vel_reward = copy.deepcopy(cfg.rewards["motion_body_ang_vel"])
  torso_ang_vel_reward.weight = 0.8
  torso_ang_vel_reward.params = {
    **torso_ang_vel_reward.params,
    "std": 1.8,
    "body_names": ("torso_link",),
  }
  cfg.rewards["motion_torso_ang_vel"] = torso_ang_vel_reward

  return cfg


def unitree_g1_mimicx_contact_phase_strong_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a stronger contact-phase candidate for stance/slip control."""
  cfg = unitree_g1_mimicx_contact_phase_light_curriculum_env_cfg(play=play)

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  if not play:
    motion_cmd.start_window_start = 370
    motion_cmd.start_window_end = 420
    motion_cmd.start_window_ratio = 0.12

  cfg.rewards["motion_ankle_z_light"].weight = 0.9
  cfg.rewards["motion_ankle_z_light"].params["std"] = 0.06
  cfg.rewards["motion_reference_stance_slip"].weight = 1.2
  cfg.rewards["motion_reference_stance_slip"].params["std"] = 0.30
  cfg.rewards["motion_reference_stance_slip"].params["stance_height"] = 0.095

  return cfg


def unitree_g1_mimicx_hybrid_contact_dynamics_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a hybrid local contact plus global/core dynamics candidate."""
  cfg = unitree_g1_mimicx_contact_phase_light_curriculum_env_cfg(play=play)

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  if not play:
    motion_cmd.start_window_start = 360
    motion_cmd.start_window_end = 430
    motion_cmd.start_window_ratio = 0.10

  core_body_names = ("pelvis", "torso_link")
  cfg.rewards["action_rate_l2"].weight = -7.0e-2
  cfg.rewards["motion_body_lin_vel"].params["std"] = 0.8
  cfg.rewards["motion_body_ang_vel"].params["std"] = 2.4

  core_ori_reward = copy.deepcopy(cfg.rewards["motion_body_ori"])
  core_ori_reward.weight = 0.8
  core_ori_reward.params = {
    **core_ori_reward.params,
    "std": 0.30,
    "body_names": core_body_names,
  }
  cfg.rewards["motion_core_ori"] = core_ori_reward

  torso_lin_vel_reward = copy.deepcopy(cfg.rewards["motion_body_lin_vel"])
  torso_lin_vel_reward.weight = 0.7
  torso_lin_vel_reward.params = {
    **torso_lin_vel_reward.params,
    "std": 0.60,
    "body_names": ("torso_link",),
  }
  cfg.rewards["motion_torso_lin_vel"] = torso_lin_vel_reward

  return cfg


def unitree_g1_mimicx_arm_ee_focus_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a tennis task-aware arm/end-effector precision candidate."""
  cfg = unitree_g1_mimicx_start_window_anchor_guard_curriculum_env_cfg(play=play)

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  if not play:
    motion_cmd.start_window_start = 340
    motion_cmd.start_window_end = 430
    motion_cmd.start_window_ratio = 0.10

  wrist_body_names = ("left_wrist_yaw_link", "right_wrist_yaw_link")
  arm_body_names = (
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )

  cfg.rewards["motion_wrist_pos"].weight = 4.0
  cfg.rewards["motion_wrist_pos"].params["std"] = 0.085
  cfg.rewards["motion_arm_pos"].weight = 2.0
  cfg.rewards["motion_arm_pos"].params["std"] = 0.14
  cfg.rewards["motion_wrist_lin_vel"].weight = 1.0
  cfg.rewards["motion_wrist_lin_vel"].params["std"] = 0.60

  arm_lin_vel_reward = copy.deepcopy(cfg.rewards["motion_body_lin_vel"])
  arm_lin_vel_reward.weight = 0.5
  arm_lin_vel_reward.params = {
    **arm_lin_vel_reward.params,
    "std": 0.75,
    "body_names": arm_body_names,
  }
  cfg.rewards["motion_arm_lin_vel"] = arm_lin_vel_reward

  wrist_ang_vel_reward = copy.deepcopy(cfg.rewards["motion_body_ang_vel"])
  wrist_ang_vel_reward.weight = 0.4
  wrist_ang_vel_reward.params = {
    **wrist_ang_vel_reward.params,
    "std": 2.0,
    "body_names": wrist_body_names,
  }
  cfg.rewards["motion_wrist_ang_vel"] = wrist_ang_vel_reward

  return cfg


def unitree_g1_mimicx_autorefine_arm_contact_balanced_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create an AutoRefine candidate balancing tennis arm precision and contact."""
  cfg = unitree_g1_mimicx_contact_phase_strong_curriculum_env_cfg(play=play)

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  if not play:
    motion_cmd.start_window_start = 340
    motion_cmd.start_window_end = 430
    motion_cmd.start_window_ratio = 0.10

  wrist_body_names = ("left_wrist_yaw_link", "right_wrist_yaw_link")
  arm_body_names = (
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )

  cfg.rewards["motion_global_root_pos"].weight = 4.8
  cfg.rewards["motion_global_root_pos"].params["std"] = 0.155
  cfg.rewards["motion_wrist_pos"].weight = 3.4
  cfg.rewards["motion_wrist_pos"].params["std"] = 0.095
  cfg.rewards["motion_arm_pos"].weight = 1.8
  cfg.rewards["motion_arm_pos"].params["std"] = 0.15
  cfg.rewards["motion_wrist_lin_vel"].weight = 0.9
  cfg.rewards["motion_wrist_lin_vel"].params["std"] = 0.65

  arm_lin_vel_reward = copy.deepcopy(cfg.rewards["motion_body_lin_vel"])
  arm_lin_vel_reward.weight = 0.35
  arm_lin_vel_reward.params = {
    **arm_lin_vel_reward.params,
    "std": 0.8,
    "body_names": arm_body_names,
  }
  cfg.rewards["motion_arm_lin_vel"] = arm_lin_vel_reward

  wrist_ang_vel_reward = copy.deepcopy(cfg.rewards["motion_body_ang_vel"])
  wrist_ang_vel_reward.weight = 0.25
  wrist_ang_vel_reward.params = {
    **wrist_ang_vel_reward.params,
    "std": 2.2,
    "body_names": wrist_body_names,
  }
  cfg.rewards["motion_wrist_ang_vel"] = wrist_ang_vel_reward

  return cfg


def unitree_g1_mimicx_autorefine_core_contact_balanced_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create an AutoRefine candidate balancing contact and core dynamics."""
  cfg = unitree_g1_mimicx_contact_phase_strong_curriculum_env_cfg(play=play)

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  if not play:
    motion_cmd.start_window_start = 360
    motion_cmd.start_window_end = 430
    motion_cmd.start_window_ratio = 0.10

  core_body_names = ("pelvis", "torso_link")
  cfg.rewards["motion_global_root_pos"].weight = 5.0
  cfg.rewards["motion_global_root_pos"].params["std"] = 0.15
  cfg.rewards["motion_global_root_ori"].weight = 1.2
  cfg.rewards["motion_global_root_ori"].params["std"] = 0.28
  cfg.rewards["action_rate_l2"].weight = -7.0e-2
  cfg.rewards["motion_body_lin_vel"].params["std"] = 0.85
  cfg.rewards["motion_body_ang_vel"].params["std"] = 2.6

  core_ori_reward = copy.deepcopy(cfg.rewards["motion_body_ori"])
  core_ori_reward.weight = 0.7
  core_ori_reward.params = {
    **core_ori_reward.params,
    "std": 0.32,
    "body_names": core_body_names,
  }
  cfg.rewards["motion_core_ori"] = core_ori_reward

  torso_lin_vel_reward = copy.deepcopy(cfg.rewards["motion_body_lin_vel"])
  torso_lin_vel_reward.weight = 0.55
  torso_lin_vel_reward.params = {
    **torso_lin_vel_reward.params,
    "std": 0.65,
    "body_names": ("torso_link",),
  }
  cfg.rewards["motion_torso_lin_vel"] = torso_lin_vel_reward

  torso_ang_vel_reward = copy.deepcopy(cfg.rewards["motion_body_ang_vel"])
  torso_ang_vel_reward.weight = 0.35
  torso_ang_vel_reward.params = {
    **torso_ang_vel_reward.params,
    "std": 2.1,
    "body_names": ("torso_link",),
  }
  cfg.rewards["motion_torso_ang_vel"] = torso_ang_vel_reward

  return cfg


def unitree_g1_mimicx_autorefine_hierarchical_balanced_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create an AutoRefine candidate with local, global, and dynamics terms."""
  cfg = unitree_g1_mimicx_autorefine_arm_contact_balanced_curriculum_env_cfg(
    play=play
  )

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  if not play:
    motion_cmd.start_window_start = 340
    motion_cmd.start_window_end = 430
    motion_cmd.start_window_ratio = 0.08

  core_body_names = ("pelvis", "torso_link")
  cfg.rewards["motion_global_root_pos"].weight = 5.0
  cfg.rewards["motion_global_root_pos"].params["std"] = 0.15
  cfg.rewards["motion_global_root_ori"].weight = 1.2
  cfg.rewards["motion_global_root_ori"].params["std"] = 0.28
  cfg.rewards["action_rate_l2"].weight = -7.5e-2
  cfg.rewards["motion_body_lin_vel"].params["std"] = 0.85
  cfg.rewards["motion_body_ang_vel"].params["std"] = 2.6

  core_ori_reward = copy.deepcopy(cfg.rewards["motion_body_ori"])
  core_ori_reward.weight = 0.6
  core_ori_reward.params = {
    **core_ori_reward.params,
    "std": 0.33,
    "body_names": core_body_names,
  }
  cfg.rewards["motion_core_ori"] = core_ori_reward

  torso_lin_vel_reward = copy.deepcopy(cfg.rewards["motion_body_lin_vel"])
  torso_lin_vel_reward.weight = 0.45
  torso_lin_vel_reward.params = {
    **torso_lin_vel_reward.params,
    "std": 0.7,
    "body_names": ("torso_link",),
  }
  cfg.rewards["motion_torso_lin_vel"] = torso_lin_vel_reward

  return cfg


def unitree_g1_mimicx_autorefine_hierarchical_gail_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """AutoRefine hierarchical curriculum with an optional GAIL-style prior reward."""
  cfg = unitree_g1_mimicx_autorefine_hierarchical_balanced_curriculum_env_cfg(
    play=play
  )
  gail_reward = copy.deepcopy(cfg.rewards["motion_body_pos"])
  gail_reward.func = mimicx_mdp.motion_adversarial_imitation_reward
  gail_reward.weight = 0.25
  gail_reward.params = {
    "command_name": "motion",
    "discriminator_path": os.environ.get("MIMICX_GAIL_DISCRIMINATOR", ""),
    "body_names": None,
    "reward_mode": "gail",
    "reward_clip": 10.0,
  }
  cfg.rewards["motion_gail_prior"] = gail_reward
  return cfg


def _load_mimicx_autorefine_patch() -> dict:
  """Load an optional AutoRefine candidate patch from the environment."""
  patch_file = os.environ.get("MIMICX_AUTOREFINE_PATCH_FILE")
  patch_json = os.environ.get("MIMICX_AUTOREFINE_PATCH_JSON")
  if patch_file:
    path = Path(patch_file).expanduser().resolve()
    return json.loads(path.read_text(encoding="utf-8"))
  if patch_json:
    return json.loads(patch_json)
  return {}


def _apply_mimicx_reward_overrides(cfg: ManagerBasedRlEnvCfg, patch: dict) -> None:
  for reward_name, reward_patch in patch.get("reward_overrides", {}).items():
    if reward_name not in cfg.rewards:
      continue
    reward = cfg.rewards[reward_name]
    if "weight" in reward_patch:
      reward.weight = float(reward_patch["weight"])
    if "std" in reward_patch:
      reward.params["std"] = float(reward_patch["std"])


def _add_mimicx_body_reward(
  cfg: ManagerBasedRlEnvCfg,
  *,
  base_reward_name: str,
  reward_name: str,
  body_names: tuple[str, ...],
  weight: float,
  std: float,
  z_only: bool = False,
) -> None:
  reward = copy.deepcopy(cfg.rewards[base_reward_name])
  reward.weight = float(weight)
  reward.params = {
    **reward.params,
    "std": float(std),
    "body_names": body_names,
  }
  if z_only:
    reward.func = mimicx_mdp.motion_relative_body_position_z_error_exp
    reward.params = {
      "command_name": "motion",
      "std": float(std),
      "body_names": body_names,
    }
  cfg.rewards[reward_name] = reward


def _apply_mimicx_body_reward_patches(
  cfg: ManagerBasedRlEnvCfg, patch: dict
) -> None:
  for item in patch.get("body_pos_rewards", ()):
    _add_mimicx_body_reward(
      cfg,
      base_reward_name="motion_body_pos",
      reward_name=f"motion_autorefine_{item['name']}",
      body_names=tuple(item["body_names"]),
      weight=float(item["weight"]),
      std=float(item["std"]),
    )
  for item in patch.get("body_z_rewards", ()):
    _add_mimicx_body_reward(
      cfg,
      base_reward_name="motion_body_pos",
      reward_name=f"motion_autorefine_{item['name']}",
      body_names=tuple(item["body_names"]),
      weight=float(item["weight"]),
      std=float(item["std"]),
      z_only=True,
    )
  for item in patch.get("body_lin_vel_rewards", ()):
    _add_mimicx_body_reward(
      cfg,
      base_reward_name="motion_body_lin_vel",
      reward_name=f"motion_autorefine_{item['name']}",
      body_names=tuple(item["body_names"]),
      weight=float(item["weight"]),
      std=float(item["std"]),
    )
  for item in patch.get("body_ang_vel_rewards", ()):
    _add_mimicx_body_reward(
      cfg,
      base_reward_name="motion_body_ang_vel",
      reward_name=f"motion_autorefine_{item['name']}",
      body_names=tuple(item["body_names"]),
      weight=float(item["weight"]),
      std=float(item["std"]),
    )
  for item in patch.get("joint_vel_rewards", ()):
    reward = copy.deepcopy(cfg.rewards["motion_body_ang_vel"])
    reward.func = mimicx_mdp.motion_joint_velocity_error_exp
    reward.weight = float(item["weight"])
    reward.params = {
      "command_name": "motion",
      "std": float(item["std"]),
    }
    cfg.rewards[f"motion_autorefine_{item['name']}"] = reward


def _apply_mimicx_autorefine_patch(
  cfg: ManagerBasedRlEnvCfg,
  patch: dict,
  *,
  play: bool,
) -> ManagerBasedRlEnvCfg:
  if not patch:
    return cfg

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  sampling = patch.get("sampling", {})
  if sampling and not play:
    mode = sampling.get("mode", "window")
    if mode == "start":
      motion_cmd.sampling_mode = "start"
      motion_cmd.start_window_ratio = 0.0
      motion_cmd.start_windows = ()
    else:
      motion_cmd.sampling_mode = "start_window"
      if "windows" in sampling:
        motion_cmd.start_windows = tuple(
          (
            int(window["start"]),
            int(window["end"]),
            float(window.get("ratio", window.get("weight", 0.25))),
          )
          for window in sampling["windows"]
        )
        motion_cmd.start_window_ratio = float(
          sum(window[2] for window in motion_cmd.start_windows)
        )
        if motion_cmd.start_windows:
          motion_cmd.start_window_start = motion_cmd.start_windows[0][0]
          motion_cmd.start_window_end = motion_cmd.start_windows[0][1]
      else:
        motion_cmd.start_windows = ()
        motion_cmd.start_window_start = int(sampling["start"])
        motion_cmd.start_window_end = int(sampling["end"])
        motion_cmd.start_window_ratio = float(sampling.get("ratio", 0.25))

  for term_name, term_patch in patch.get("termination_overrides", {}).items():
    if term_name in cfg.terminations:
      cfg.terminations[term_name].params.update(term_patch)

  _apply_mimicx_reward_overrides(cfg, patch)
  _apply_mimicx_body_reward_patches(cfg, patch)
  return cfg


def unitree_g1_mimicx_autorefine_dynamic_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a generic AutoRefine task from a JSON candidate patch.

  If no patch is provided, this is equivalent to the hierarchical balanced
  curriculum. Patches are read from either `MIMICX_AUTOREFINE_PATCH_FILE` or
  `MIMICX_AUTOREFINE_PATCH_JSON`.
  """
  patch = _load_mimicx_autorefine_patch()
  base_profile = patch.get("base_profile", "hierarchical")
  if base_profile == "mimicx":
    cfg = unitree_g1_mimicx_curriculum_env_cfg(play=play)
  elif base_profile == "hierarchical":
    cfg = unitree_g1_mimicx_autorefine_hierarchical_balanced_curriculum_env_cfg(
      play=play
    )
  else:
    raise ValueError(f"Unknown AutoRefine base_profile: {base_profile}")
  return _apply_mimicx_autorefine_patch(cfg, patch, play=play)


def unitree_g1_mimicx_kongfu1_failure_window_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a kongfu1 failure-window curriculum for the first right-leg reset."""
  cfg = unitree_g1_mimicx_autorefine_hierarchical_balanced_curriculum_env_cfg(
    play=play
  )

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  if not play:
    motion_cmd.start_window_start = 165
    motion_cmd.start_window_end = 225
    motion_cmd.start_window_ratio = 0.35

  ankle_body_names = ("left_ankle_roll_link", "right_ankle_roll_link")
  knee_body_names = ("left_knee_link", "right_knee_link")
  ankle_knee_body_names = (
    "left_knee_link",
    "left_ankle_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
  )
  lower_body_names = (
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
  )

  cfg.rewards["motion_global_root_pos"].weight = 5.2
  cfg.rewards["motion_global_root_pos"].params["std"] = 0.145
  cfg.rewards["motion_lower_body_pos"].weight = 2.2
  cfg.rewards["motion_lower_body_pos"].params["std"] = 0.12
  cfg.rewards["motion_feet_pos"].weight = 3.4
  cfg.rewards["motion_feet_pos"].params["std"] = 0.105
  cfg.rewards["motion_ankle_lin_vel"].weight = 1.0
  cfg.rewards["motion_ankle_lin_vel"].params["std"] = 0.55
  cfg.rewards["motion_lower_body_lin_vel"].weight = 0.9
  cfg.rewards["motion_lower_body_lin_vel"].params["std"] = 0.60

  ankle_knee_pos_reward = copy.deepcopy(cfg.rewards["motion_body_pos"])
  ankle_knee_pos_reward.weight = 2.4
  ankle_knee_pos_reward.params = {
    **ankle_knee_pos_reward.params,
    "std": 0.10,
    "body_names": ankle_knee_body_names,
  }
  cfg.rewards["motion_kongfu1_ankle_knee_pos"] = ankle_knee_pos_reward

  ankle_z_reward = copy.deepcopy(cfg.rewards["motion_feet_pos"])
  ankle_z_reward.func = mimicx_mdp.motion_relative_body_position_z_error_exp
  ankle_z_reward.weight = 1.5
  ankle_z_reward.params = {
    "command_name": "motion",
    "std": 0.052,
    "body_names": ankle_body_names,
  }
  cfg.rewards["motion_kongfu1_ankle_z"] = ankle_z_reward

  knee_z_reward = copy.deepcopy(cfg.rewards["motion_body_pos"])
  knee_z_reward.func = mimicx_mdp.motion_relative_body_position_z_error_exp
  knee_z_reward.weight = 1.2
  knee_z_reward.params = {
    "command_name": "motion",
    "std": 0.065,
    "body_names": knee_body_names,
  }
  cfg.rewards["motion_kongfu1_knee_z"] = knee_z_reward

  lower_body_ang_vel_reward = copy.deepcopy(cfg.rewards["motion_body_ang_vel"])
  lower_body_ang_vel_reward.weight = 0.35
  lower_body_ang_vel_reward.params = {
    **lower_body_ang_vel_reward.params,
    "std": 2.0,
    "body_names": lower_body_names,
  }
  cfg.rewards["motion_kongfu1_lower_body_ang_vel"] = lower_body_ang_vel_reward

  return cfg


def unitree_g1_mimicx_kongfu1_right_leg_z_window_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a kongfu1 v2 curriculum focused on right-leg z tracking."""
  cfg = unitree_g1_mimicx_kongfu1_failure_window_curriculum_env_cfg(play=play)

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  if not play:
    motion_cmd.start_window_start = 195
    motion_cmd.start_window_end = 255
    motion_cmd.start_window_ratio = 0.55

  right_leg_body_names = (
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
  )
  right_knee_body_names = ("right_knee_link",)
  right_ankle_body_names = ("right_ankle_roll_link",)
  ankle_body_names = ("left_ankle_roll_link", "right_ankle_roll_link")

  cfg.rewards["motion_global_root_pos"].weight = 4.8
  cfg.rewards["motion_global_root_pos"].params["std"] = 0.15
  cfg.rewards["action_rate_l2"].weight = -6.0e-2
  cfg.rewards["motion_lower_body_pos"].weight = 2.6
  cfg.rewards["motion_lower_body_pos"].params["std"] = 0.115
  cfg.rewards["motion_ankle_lin_vel"].weight = 1.15
  cfg.rewards["motion_ankle_lin_vel"].params["std"] = 0.50
  cfg.rewards["motion_kongfu1_ankle_z"].weight = 1.8
  cfg.rewards["motion_kongfu1_ankle_z"].params["std"] = 0.048
  cfg.rewards["motion_kongfu1_knee_z"].weight = 1.8
  cfg.rewards["motion_kongfu1_knee_z"].params["std"] = 0.055

  right_leg_pos_reward = copy.deepcopy(cfg.rewards["motion_body_pos"])
  right_leg_pos_reward.weight = 2.4
  right_leg_pos_reward.params = {
    **right_leg_pos_reward.params,
    "std": 0.09,
    "body_names": right_leg_body_names,
  }
  cfg.rewards["motion_kongfu1_right_leg_pos"] = right_leg_pos_reward

  right_knee_z_reward = copy.deepcopy(cfg.rewards["motion_body_pos"])
  right_knee_z_reward.func = mimicx_mdp.motion_relative_body_position_z_error_exp
  right_knee_z_reward.weight = 2.3
  right_knee_z_reward.params = {
    "command_name": "motion",
    "std": 0.045,
    "body_names": right_knee_body_names,
  }
  cfg.rewards["motion_kongfu1_right_knee_z"] = right_knee_z_reward

  right_ankle_z_reward = copy.deepcopy(cfg.rewards["motion_body_pos"])
  right_ankle_z_reward.func = mimicx_mdp.motion_relative_body_position_z_error_exp
  right_ankle_z_reward.weight = 2.0
  right_ankle_z_reward.params = {
    "command_name": "motion",
    "std": 0.045,
    "body_names": right_ankle_body_names,
  }
  cfg.rewards["motion_kongfu1_right_ankle_z"] = right_ankle_z_reward

  ankle_clearance_reward = copy.deepcopy(cfg.rewards["motion_body_pos"])
  ankle_clearance_reward.func = mimicx_mdp.motion_relative_body_position_z_error_exp
  ankle_clearance_reward.weight = 1.2
  ankle_clearance_reward.params = {
    "command_name": "motion",
    "std": 0.055,
    "body_names": ankle_body_names,
  }
  cfg.rewards["motion_kongfu1_bilateral_ankle_z"] = ankle_clearance_reward

  return cfg


def unitree_g1_mimicx_kongfu1_late_anchor_window_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a kongfu1 v3 curriculum for the late anchor/right-side drift."""
  cfg = unitree_g1_mimicx_kongfu1_right_leg_z_window_curriculum_env_cfg(
    play=play
  )

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  if not play:
    motion_cmd.start_window_start = 560
    motion_cmd.start_window_end = 650
    motion_cmd.start_window_ratio = 0.65

  core_body_names = ("pelvis", "torso_link")
  right_side_body_names = (
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )
  right_arm_body_names = (
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )

  cfg.rewards["motion_global_root_pos"].weight = 5.8
  cfg.rewards["motion_global_root_pos"].params["std"] = 0.13
  cfg.rewards["motion_global_root_ori"].weight = 1.35
  cfg.rewards["motion_global_root_ori"].params["std"] = 0.25
  cfg.rewards["motion_body_pos"].params["std"] = 0.24
  cfg.rewards["motion_body_ori"].params["std"] = 0.32
  cfg.rewards["motion_core_pos"].weight = 2.8
  cfg.rewards["motion_core_pos"].params["std"] = 0.10
  cfg.rewards["motion_core_ori"].weight = 0.9
  cfg.rewards["motion_core_ori"].params["std"] = 0.26
  cfg.rewards["motion_torso_lin_vel"].weight = 0.75
  cfg.rewards["motion_torso_lin_vel"].params["std"] = 0.55
  cfg.rewards["action_rate_l2"].weight = -6.5e-2

  right_side_pos_reward = copy.deepcopy(cfg.rewards["motion_body_pos"])
  right_side_pos_reward.weight = 2.2
  right_side_pos_reward.params = {
    **right_side_pos_reward.params,
    "std": 0.105,
    "body_names": right_side_body_names,
  }
  cfg.rewards["motion_kongfu1_right_side_pos"] = right_side_pos_reward

  right_arm_pos_reward = copy.deepcopy(cfg.rewards["motion_body_pos"])
  right_arm_pos_reward.weight = 1.5
  right_arm_pos_reward.params = {
    **right_arm_pos_reward.params,
    "std": 0.12,
    "body_names": right_arm_body_names,
  }
  cfg.rewards["motion_kongfu1_right_arm_pos"] = right_arm_pos_reward

  right_side_lin_vel_reward = copy.deepcopy(cfg.rewards["motion_body_lin_vel"])
  right_side_lin_vel_reward.weight = 0.65
  right_side_lin_vel_reward.params = {
    **right_side_lin_vel_reward.params,
    "std": 0.65,
    "body_names": right_side_body_names,
  }
  cfg.rewards["motion_kongfu1_right_side_lin_vel"] = right_side_lin_vel_reward

  return cfg


def unitree_g1_mimicx_kongfu1_full_start_consolidation_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a kongfu1 full-start consolidation curriculum."""
  cfg = unitree_g1_mimicx_kongfu1_right_leg_z_window_curriculum_env_cfg(
    play=play
  )

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  if not play:
    motion_cmd.sampling_mode = "start"
    motion_cmd.start_window_ratio = 0.0

  cfg.rewards["motion_global_root_pos"].weight = 5.0
  cfg.rewards["motion_global_root_pos"].params["std"] = 0.145
  cfg.rewards["motion_global_root_ori"].weight = 1.25
  cfg.rewards["motion_global_root_ori"].params["std"] = 0.27
  cfg.rewards["motion_core_pos"].weight = 2.4
  cfg.rewards["motion_core_pos"].params["std"] = 0.11
  cfg.rewards["motion_core_ori"].weight = 0.75
  cfg.rewards["motion_core_ori"].params["std"] = 0.30
  cfg.rewards["action_rate_l2"].weight = -6.0e-2

  return cfg
