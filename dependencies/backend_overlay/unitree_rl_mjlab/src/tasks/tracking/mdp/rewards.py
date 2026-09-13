# MimicX-modified tracking overlay; see NOTICE for attribution.
from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_error_magnitude

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DISCRIMINATOR_CACHE: dict[tuple[str, str], torch.jit.ScriptModule | torch.nn.Module] = {}


def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def _load_scripted_discriminator(path: str, device: torch.device | str):
  if path == "":
    return None
  key = (path, str(device))
  model = _DISCRIMINATOR_CACHE.get(key)
  if model is None:
    model = torch.jit.load(path, map_location=device)
    model.eval()
    _DISCRIMINATOR_CACHE[key] = model
  return model


def _motion_state_feature(
  command: MotionCommand,
  body_names: tuple[str, ...] | None,
) -> torch.Tensor:
  """State-only motion feature shared by expert extraction and GAIL-style reward."""
  body_indexes = _get_body_indexes(command, body_names)
  robot_body_pos = command.robot_body_pos_w[:, body_indexes]
  robot_body_lin_vel = command.robot_body_lin_vel_w[:, body_indexes]
  robot_body_ang_vel = command.robot_body_ang_vel_w[:, body_indexes]
  anchor_pos = command.robot_anchor_pos_w[:, None, :]
  body_pos_relative = robot_body_pos - anchor_pos
  return torch.cat(
    [
      command.robot_joint_pos,
      command.robot_joint_vel,
      body_pos_relative.reshape(command.robot_joint_pos.shape[0], -1),
      robot_body_lin_vel.reshape(command.robot_joint_pos.shape[0], -1),
      robot_body_ang_vel.reshape(command.robot_joint_pos.shape[0], -1),
    ],
    dim=-1,
  )


def motion_adversarial_imitation_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  discriminator_path: str = "",
  body_names: tuple[str, ...] | None = None,
  reward_mode: str = "gail",
  reward_clip: float = 10.0,
) -> torch.Tensor:
  """Frozen GAIL/AMP-style discriminator reward for motion tracking.

  The discriminator is expected to be a TorchScript module that maps the
  state-only motion feature to an expert logit. This reward does not train the
  discriminator online; it lets the existing PPO pipeline consume a learned
  imitation prior after offline discriminator training.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  if discriminator_path == "":
    return torch.zeros(command.robot_joint_pos.shape[0], device=command.device)
  model = _load_scripted_discriminator(discriminator_path, command.device)
  if model is None:
    return torch.zeros(command.robot_joint_pos.shape[0], device=command.device)
  features = _motion_state_feature(command, body_names)
  with torch.no_grad():
    logits = model(features).reshape(-1)
    if reward_mode == "amp":
      reward = torch.clamp(1.0 - 0.25 * torch.square(logits - 1.0), min=0.0)
    else:
      # GAIL generator reward: -log(1 - D) = softplus(logit), D=sigmoid(logit).
      reward = torch.nn.functional.softplus(logits)
    return torch.clamp(reward, min=0.0, max=reward_clip)


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_pos_relative_w[:, body_indexes]
      - command.robot_body_pos_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_position_z_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.square(
    command.body_pos_relative_w[:, body_indexes, 2]
    - command.robot_body_pos_w[:, body_indexes, 2]
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = (
    quat_error_magnitude(
      command.body_quat_relative_w[:, body_indexes],
      command.robot_body_quat_w[:, body_indexes],
    )
    ** 2
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_lin_vel_w[:, body_indexes]
      - command.robot_body_lin_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_ang_vel_w[:, body_indexes]
      - command.robot_body_ang_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_joint_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.square(command.joint_vel - command.robot_joint_vel)
  return torch.exp(-error.mean(-1) / std**2)


def motion_reference_stance_foot_slip_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  stance_height: float,
  body_names: tuple[str, ...],
  vertical_weight: float = 0.25,
) -> torch.Tensor:
  """Reward low robot foot velocity when the reference ankle is in a low stance phase."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  ref_z = command.body_pos_relative_w[:, body_indexes, 2]
  stance_mask = (ref_z < stance_height).float()
  robot_vel = command.robot_body_lin_vel_w[:, body_indexes]
  horizontal_error = torch.sum(torch.square(robot_vel[..., :2]), dim=-1)
  vertical_error = vertical_weight * torch.square(robot_vel[..., 2])
  error = (horizontal_error + vertical_error) * stance_mask
  denom = torch.clamp(stance_mask.sum(dim=-1), min=1.0)
  return torch.exp(-(error.sum(dim=-1) / denom) / std**2)


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)
