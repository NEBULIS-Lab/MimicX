# MimicX-modified tracking overlay; see NOTICE for attribution.
"""Unitree G1_23Dof flat tracking environment configurations."""

import copy

from src.assets.robots.unitree_g1.g1_23dof_constants import (
  G1_23DOF_ACTION_SCALE,
  get_g1_23dof_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from src.tasks.tracking.mdp import MotionCommandCfg
from src.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from src.tasks.tracking import mdp as mimicx_mdp


def unitree_g1_23dof_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1_23Dof flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_g1_23dof_robot_cfg()}

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
  joint_pos_action.scale = G1_23DOF_ACTION_SCALE

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
    "left_wrist_roll_rubber_hand",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_roll_rubber_hand",
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_roll_rubber_hand",
    "right_wrist_roll_rubber_hand",
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


def unitree_g1_23dof_mimicx_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a lower-randomization G1_23Dof tracking task for MimicX bring-up."""
  cfg = unitree_g1_23dof_flat_tracking_env_cfg(has_state_estimation=True, play=play)

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


def unitree_g1_23dof_mimicx_start_root_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a start-sampled root/feet MimicX G1_23Dof curriculum."""
  cfg = unitree_g1_23dof_mimicx_curriculum_env_cfg(play=play)

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.sampling_mode = "start"

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


def unitree_g1_23dof_mimicx_start_root_orient_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a root-orientation/feet repair curriculum for MimicX dance tracking."""
  cfg = unitree_g1_23dof_mimicx_start_root_curriculum_env_cfg(play=play)

  cfg.rewards["motion_global_root_ori"].weight = 2.0
  cfg.rewards["motion_global_root_ori"].params["std"] = 0.25
  cfg.rewards["motion_body_ori"].params["std"] = 0.35

  cfg.rewards["motion_feet_pos"].weight = 3.0
  cfg.rewards["motion_feet_pos"].params["std"] = 0.12

  return cfg


def unitree_g1_23dof_mimicx_start_root_feet_lock_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a feet-lock repair curriculum after root orientation is stable."""
  cfg = unitree_g1_23dof_mimicx_start_root_orient_curriculum_env_cfg(play=play)

  ankle_body_names = ("left_ankle_roll_link", "right_ankle_roll_link")
  cfg.rewards["motion_feet_pos"].weight = 5.0
  cfg.rewards["motion_feet_pos"].params["std"] = 0.08

  feet_lin_vel_reward = copy.deepcopy(cfg.rewards["motion_body_lin_vel"])
  feet_lin_vel_reward.weight = 1.5
  feet_lin_vel_reward.params = {
    **feet_lin_vel_reward.params,
    "std": 0.6,
    "body_names": ankle_body_names,
  }
  cfg.rewards["motion_feet_lin_vel"] = feet_lin_vel_reward

  return cfg


def unitree_g1_23dof_mimicx_adaptive_feet_lock_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create an adaptive-sampling feet-lock repair curriculum."""
  cfg = unitree_g1_23dof_mimicx_start_root_feet_lock_curriculum_env_cfg(play=play)

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  if not play:
    motion_cmd.sampling_mode = "adaptive"
    motion_cmd.adaptive_kernel_size = 3
    motion_cmd.adaptive_lambda = 0.8
    motion_cmd.adaptive_uniform_ratio = 0.05
    motion_cmd.adaptive_alpha = 0.05

  return cfg


def unitree_g1_23dof_mimicx_start_window_lower_body_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a gentle failure-window replay curriculum for the dance right-leg spike."""
  cfg = unitree_g1_23dof_mimicx_start_root_feet_lock_curriculum_env_cfg(play=play)

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  if not play:
    motion_cmd.sampling_mode = "start_window"
    motion_cmd.start_window_start = 500
    motion_cmd.start_window_end = 545
    motion_cmd.start_window_ratio = 0.25

  lower_body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
  )

  lower_body_pos_reward = copy.deepcopy(cfg.rewards["motion_body_pos"])
  lower_body_pos_reward.weight = 2.0
  lower_body_pos_reward.params = {
    **lower_body_pos_reward.params,
    "std": 0.16,
    "body_names": lower_body_names,
  }
  cfg.rewards["motion_lower_body_pos"] = lower_body_pos_reward

  lower_body_lin_vel_reward = copy.deepcopy(cfg.rewards["motion_body_lin_vel"])
  lower_body_lin_vel_reward.weight = 1.0
  lower_body_lin_vel_reward.params = {
    **lower_body_lin_vel_reward.params,
    "std": 0.55,
    "body_names": lower_body_names,
  }
  cfg.rewards["motion_lower_body_lin_vel"] = lower_body_lin_vel_reward

  return cfg


def unitree_g1_23dof_mimicx_start_window_feet_z_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a conservative failure-window curriculum with direct ankle-z reward."""
  cfg = unitree_g1_23dof_mimicx_start_window_lower_body_curriculum_env_cfg(play=play)

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  if not play:
    motion_cmd.start_window_ratio = 0.15

  ankle_body_names = ("left_ankle_roll_link", "right_ankle_roll_link")
  feet_z_reward = copy.deepcopy(cfg.rewards["motion_feet_pos"])
  feet_z_reward.func = mimicx_mdp.motion_relative_body_position_z_error_exp
  feet_z_reward.weight = 3.0
  feet_z_reward.params = {
    "command_name": "motion",
    "std": 0.055,
    "body_names": ankle_body_names,
  }
  cfg.rewards["motion_feet_z"] = feet_z_reward

  return cfg
