"""Isaac task extensions. Import only after AppLauncher has initialized Kit."""

import torch
from mimicx.backends.isaaclab.compat import (
    prepare_upstream_import, TorchArticulationView, torch_view, canonical_to_native_quaternion,
)

prepare_upstream_import()

from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, yaw_quat, quat_apply_inverse
from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
from whole_body_tracking.tasks.tracking import mdp


class RefinementMotionCommand(MotionCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if hasattr(self.robot.data.joint_pos, "torch"):
            # Lab3 data AND math functions use XYZW. Files remain canonical
            # WXYZ; convert exactly once at the loader/native-math boundary.
            self.motion._body_quat_w = canonical_to_native_quaternion(self.motion._body_quat_w, xyzw=True)
        self.robot = TorchArticulationView(self.robot)

    def refresh_relative(self):
        """Initialize alignment after reset FK without advancing motion time."""
        anchor = self.anchor_pos_w[:, None, :]
        robot_anchor = self.robot_anchor_pos_w[:, None, :].clone()
        robot_anchor[..., 2] = anchor[..., 2]
        heading = yaw_quat(quat_mul(self.robot_anchor_quat_w, quat_inv(self.anchor_quat_w)))[:, None, :]
        heading = heading.expand(-1, len(self.cfg.body_names), -1)
        self.body_quat_relative_w = quat_mul(heading, self.body_quat_w)
        self.body_pos_relative_w = robot_anchor + quat_apply(heading, self.body_pos_w - anchor)

    def _adaptive_sampling(self, env_ids):
        sampling = getattr(self.cfg, "mimicx_sampling", None)
        if sampling is None:
            return super()._adaptive_sampling(env_ids)
        self.time_steps[env_ids] = 0
        if sampling["mode"] == "window":
            start, end = sampling["start"], sampling["end"]
            draws = torch.randint(start, end, (len(env_ids),), device=self.device)
            mask = torch.rand(len(env_ids), device=self.device) < sampling["ratio"]
            self.time_steps[env_ids] = torch.where(mask, draws, 0)

    def _update_command(self):
        if getattr(self.cfg, "mimicx_strict", False):
            # Upstream wraps at the last frame by teleporting the robot. Never do
            # that inside a phase-zero evaluation trajectory.
            self.time_steps.clamp_(max=self.motion.time_step_total - 2)
        super()._update_command()


def body_z_reward(env, command_name, std, body_names):
    command = env.command_manager.get_term(command_name)
    ids = [command.cfg.body_names.index(name) for name in body_names]
    delta = command.body_pos_relative_w[:, ids, 2] - command.robot_body_pos_w[:, ids, 2]
    return torch.exp(-delta.square().mean(-1) / std**2)


def bad_anchor_ori(env, asset_cfg, command_name, threshold):
    command = env.command_manager.get_term(command_name)
    gravity = torch_view(env.scene[asset_cfg.name].data.GRAVITY_VEC_W)
    reference = quat_apply_inverse(command.anchor_quat_w, gravity)
    actual = quat_apply_inverse(command.robot_anchor_quat_w, gravity)
    return (reference[:, 2] - actual[:, 2]).abs() > threshold


def randomize_joint_default_pos(env, env_ids, asset_cfg, pos_distribution_params=None,
                               operation="abs", distribution="uniform"):
    from isaaclab.envs.mdp.events import _randomize_prop_by_op
    asset = env.scene[asset_cfg.name]
    default = torch_view(asset.data.default_joint_pos)
    asset.data.default_joint_pos_nominal = default[0].clone()
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    joint_ids = asset_cfg.joint_ids
    if pos_distribution_params is not None:
        randomized = _randomize_prop_by_op(default.clone(), pos_distribution_params,
                                           env_ids, joint_ids, operation, distribution)
        selection = env_ids if joint_ids == slice(None) else env_ids[:, None]
        values = randomized[selection, joint_ids]
        default[selection, joint_ids] = values
        torch_view(env.action_manager.get_term("joint_pos")._offset)[selection, joint_ids] = values


@configclass
class MimicXEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        migrated = not hasattr(self.sim, "physx")
        if migrated:
            from isaaclab_physx.physics import PhysxCfg
            if self.sim.physics is None:
                self.sim.physics = PhysxCfg()
            self.sim.physx = self.sim.physics
        try:
            super().__post_init__()
        finally:
            if migrated:
                del self.sim.physx
        self.commands.motion.class_type = RefinementMotionCommand
        self.events.add_joint_default_pos.func = randomize_joint_default_pos
        self.terminations.anchor_ori.func = bad_anchor_ori
        if migrated:
            # Lab3's native term implements the same additive COM randomization
            # through the new backend API; the upstream term uses root_physx_view.
            from isaaclab.envs.mdp.events import randomize_rigid_body_com
            self.events.base_com.func = randomize_rigid_body_com
            from mimicx.backends.isaaclab.sensors import FullBodyContactSensor, spawn_with_full_contact_reports
            self.scene.robot.spawn.func = spawn_with_full_contact_reports
            self.scene.contact_forces.class_type = FullBodyContactSensor


def configure(env_cfg, *, motion_file, strict=False, patch=None, frames, fps):
    from mimicx.backends.isaaclab.configuration import apply_patch

    command = env_cfg.commands.motion
    command.motion_file = str(motion_file)
    command.debug_vis = False
    command.mimicx_strict = strict
    if strict:
        command.mimicx_sampling = {"mode": "start"}
        command.pose_range = {}
        command.velocity_range = {}
        command.joint_position_range = (0.0, 0.0)
        env_cfg.observations.policy.enable_corruption = False
        env_cfg.events.push_robot = None
        # Do not confuse normal episode expiry with physical failure.
        env_cfg.episode_length_s = (frames + 2) / fps
    if patch is not None:
        if strict:
            raise ValueError("Training patches cannot alter strict evaluation")
        apply_patch(env_cfg, patch, frames=frames, reward_factory=RewardTermCfg,
                    functions={"body_pos_rewards": mdp.motion_relative_body_position_error_exp,
                               "body_z_rewards": body_z_reward,
                               "body_lin_vel_rewards": mdp.motion_global_body_linear_velocity_error_exp,
                               "body_ang_vel_rewards": mdp.motion_global_body_angular_velocity_error_exp})
    return env_cfg


def register_tasks():
    import gymnasium as gym
    for task in ("MimicX-Tracking-Flat-G1-v0", "MimicX-Tracking-Flat-G1-Strict-v0",
                 "BeyondMimic-Tracking-Flat-G1-v0", "BeyondMimic-Tracking-Flat-G1-Strict-v0"):
        if task not in gym.registry:
            gym.register(id=task, entry_point="isaaclab.envs:ManagerBasedRLEnv",
                         disable_env_checker=True,
                         kwargs={"env_cfg_entry_point": MimicXEnvCfg})
