# MimicX-modified tracking overlay; see NOTICE for attribution.
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class MotionResidualJointPositionActionCfg(JointPositionActionCfg):
  """Joint position residual action around the current motion reference pose."""

  command_name: str = "motion"

  def build(self, env: ManagerBasedRlEnv) -> MotionResidualJointPositionAction:
    return MotionResidualJointPositionAction(self, env)


class MotionResidualJointPositionAction(JointPositionAction):
  """PD joint target action with a per-step reference-joint offset."""

  cfg: MotionResidualJointPositionActionCfg

  def __init__(
    self,
    cfg: MotionResidualJointPositionActionCfg,
    env: ManagerBasedRlEnv,
  ):
    super().__init__(cfg=cfg, env=env)
    self._env = env

  def process_actions(self, actions):
    self._raw_actions[:] = actions
    command = cast(
      MotionCommand, self._env.command_manager.get_term(self.cfg.command_name)
    )
    reference_joint_pos = command.joint_pos[:, self._target_ids]
    self._processed_actions = self._raw_actions * self._scale + reference_joint_pos
