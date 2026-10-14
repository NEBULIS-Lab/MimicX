"""Fixed-scene expert tasks for video-conditioned terrain motion tracking."""
import copy
import json
from pathlib import Path

from .environment import SceneBuilder, phase_body_tracking

TASK='MimicX-G1-Terrain-Reference'


def register(scene_path, profile_path):
    from mjlab.tasks.registry import register_mjlab_task
    from mjlab.managers.reward_manager import RewardTermCfg
    from src.tasks.tracking.config.g1.env_cfgs import unitree_g1_mimicx_curriculum_env_cfg
    from src.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
    from src.tasks.tracking.rl import MotionTrackingOnPolicyRunner
    scene=json.loads(Path(scene_path).read_text())
    profile=json.loads(Path(profile_path).read_text())

    def make(play):
        cfg=unitree_g1_mimicx_curriculum_env_cfg(play=play)
        cfg.scene.terrain=None;cfg.scene.spec_fn=SceneBuilder(str(Path(scene_path).resolve()))
        cfg.scene.extent=12.;cfg.sim.nconmax=192;cfg.sim.njmax=800
        cfg.episode_length_s=max(20.,profile['duration']+2.)
        cfg.viewer.origin_type=cfg.viewer.OriginType.WORLD
        cfg.viewer.lookat=tuple(scene['viewer_lookat'])
        cfg.viewer.distance=scene['viewer_distance'];cfg.viewer.elevation=-22.;cfg.viewer.azimuth=130.
        cfg.viewer.max_extra_envs=0
        cmd=cfg.commands['motion'];cmd.viz.ground_align=False
        cmd.sampling_mode='start' if play else profile.get('sampling','uniform')
        cmd.viz.ghost_color=(.22,.45,.8,.4) if not profile.get('refined') else (.73,.29,.23,.4)
        if play:cmd.joint_position_range=(-.01,.01)
        if not play and cmd.sampling_mode=='start_window':
            cmd.start_windows=tuple(tuple(w) for w in profile['windows'])
        if profile.get('refined'):
            cfg.rewards['motion_global_root_pos'].weight=2.
            feet=copy.deepcopy(cfg.rewards['motion_body_pos']);feet.weight=2.
            feet.params.update(body_names=('left_ankle_roll_link','right_ankle_roll_link'),std=.18)
            cfg.rewards['terrain_feet']=feet
            for name,frames in profile.get('events_control_frames',{}).items():
                if name not in ('landing','feet_on_wall','left_palm_on_wall','step_down'):continue
                bodies=('left_wrist_yaw_link',) if name=='left_palm_on_wall' else ('left_ankle_roll_link','right_ankle_roll_link')
                cfg.rewards['terrain_'+name]=RewardTermCfg(func=phase_body_tracking,weight=1.5,
                    params=dict(command_name='motion',body_names=bodies,start_frame=frames[0],end_frame=frames[1],std=.16))
        return cfg
    register_mjlab_task(task_id=TASK,env_cfg=make(False),play_env_cfg=make(True),
        rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),runner_cls=MotionTrackingOnPolicyRunner)
    return TASK
