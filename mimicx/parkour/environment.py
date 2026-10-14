"""Scene-aware tracking registration, without modifying backend task defaults."""
from dataclasses import dataclass
import json
from pathlib import Path

from .scene_geometry import validate_scene

TASK = 'MimicX-G1-Parkour-Scene'


def install_initial_reference_refresh():
    """Scope the initial reset cache refresh to this standalone process."""
    import mjlab.envs
    from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul, yaw_quat
    base = mjlab.envs.ManagerBasedRlEnv

    class InitialReferenceAlignedEnv(base):
        def reset(self, **kwargs):
            _, extras = super().reset(**kwargs)
            command = self.command_manager.get_term('motion')
            # reset() forwards the robot state but does not update the command's
            # relative-target cache. Refresh that cache without advancing time.
            anchor = command.anchor_pos_w[:, None, :]
            delta_pos = command.robot_anchor_pos_w[:, None, :].clone()
            delta_pos[..., 2] = anchor[..., 2]
            delta_rot = yaw_quat(quat_mul(command.robot_anchor_quat_w[:, None, :],
                                         quat_inv(command.anchor_quat_w[:, None, :])))
            count = len(command.cfg.body_names)
            delta_rot = delta_rot.expand(-1, count, -1)
            command.body_pos_relative_w = delta_pos + quat_apply(delta_rot, command.body_pos_w-anchor)
            command.body_quat_relative_w = quat_mul(delta_rot, command.body_quat_w)
            self.obs_buf = self.observation_manager.compute(update_history=False)
            return self.obs_buf, extras

    mjlab.envs.ManagerBasedRlEnv = InitialReferenceAlignedEnv
    return base, InitialReferenceAlignedEnv


def phase_body_tracking(env, command_name, body_names, start_frame, end_frame, std):
    import torch
    command=env.command_manager.get_term(command_name)
    ids=[command.cfg.body_names.index(name) for name in body_names]
    error=(command.body_pos_w[:,ids]-command.robot_body_pos_w[:,ids]).square().sum(-1).mean(-1)
    active=(command.time_steps>=start_frame)&(command.time_steps<end_frame)
    return torch.exp(-error/(std*std))*active


@dataclass(frozen=True)
class SceneBuilder:
    scene_path: str

    def __call__(self, spec):
        import mujoco
        import numpy as np
        scene=json.loads(Path(self.scene_path).read_text())
        validate_scene(scene)
        spec.visual.headlight.ambient=[.55,.55,.55]
        spec.visual.headlight.diffuse=[.65,.65,.65]
        spec.worldbody.add_geom(name='parkour_ground',type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0,0,.05],friction=[.8,.005,.0001],rgba=[.7,.73,.74,1])
        for b in scene['boxes']:
            spec.worldbody.add_geom(name='parkour_'+b['name'],type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=b['center'],size=b['half_size'],
                quat=[np.cos(b['yaw']/2),0,0,np.sin(b['yaw']/2)],
                friction=[b['friction'],.005,.0001],rgba=b['rgba'])


def register(scene_path, method='fixed_reference'):
    import copy
    from mjlab.tasks.registry import register_mjlab_task, list_tasks
    from src.tasks.tracking.config.g1.env_cfgs import unitree_g1_mimicx_curriculum_env_cfg
    from src.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
    from src.tasks.tracking.rl import MotionTrackingOnPolicyRunner

    if method not in ('fixed_reference','contact_aware'):
        raise ValueError(f'Unknown training configuration: {method}')
    if TASK in list_tasks():
        raise RuntimeError('Register one immutable parkour configuration per process')
    def make(play):
        cfg=unitree_g1_mimicx_curriculum_env_cfg(play=play)
        # Each Warp world gets one identical scene at the origin. Grid offsets
        # would move robots away from a single shared-coordinate wall.
        cfg.scene.terrain=None
        cfg.scene.spec_fn=SceneBuilder(str(Path(scene_path).resolve()))
        cfg.scene.extent=7.
        cfg.sim.nconmax=192
        cfg.sim.njmax=800
        cfg.viewer.distance=5.5
        cfg.viewer.elevation=-25.
        cfg.viewer.azimuth=210.
        cfg.viewer.origin_type=cfg.viewer.OriginType.WORLD
        scene=json.loads(Path(scene_path).read_text())
        cfg.viewer.lookat=(*scene['boxes'][0]['center'][:2],.55)
        cfg.viewer.max_extra_envs=0
        cfg.commands['motion'].viz.ground_align=False
        cfg.commands['motion'].viz.ghost_color=(.2,.48,.85,.4) if method=='fixed_reference' else (.25,.75,.4,.4)
        cfg.commands['motion'].sampling_mode='start' if play else 'adaptive'
        if play:
            cfg.commands['motion'].joint_position_range=(-.01,.01)
        if method=='contact_aware':
            from mjlab.managers.reward_manager import RewardTermCfg
            for name,bodies,weight in (
                ('parkour_palms',('left_wrist_yaw_link','right_wrist_yaw_link'),1.5),
                ('parkour_feet',('left_ankle_roll_link','right_ankle_roll_link'),1.5)):
                reward=copy.deepcopy(cfg.rewards['motion_body_pos'])
                reward.weight=weight
                reward.params.update(body_names=bodies,std=.18)
                cfg.rewards[name]=reward
            cfg.rewards['motion_global_root_pos'].weight=2.
            cfg.rewards['parkour_wall_phase']=RewardTermCfg(func=phase_body_tracking,weight=2.,
                params=dict(command_name='motion',body_names=('left_wrist_yaw_link',),
                            start_frame=70,end_frame=92,std=.12))
            cfg.rewards['parkour_landing_phase']=RewardTermCfg(func=phase_body_tracking,weight=2.,
                params=dict(command_name='motion',body_names=('left_ankle_roll_link','right_ankle_roll_link'),
                            start_frame=105,end_frame=117,std=.12))
        return cfg
    register_mjlab_task(task_id=TASK,env_cfg=make(False),play_env_cfg=make(True),
        rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),runner_cls=MotionTrackingOnPolicyRunner)
    return TASK
