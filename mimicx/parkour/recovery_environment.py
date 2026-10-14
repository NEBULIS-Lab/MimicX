"""Process-local continuation objectives; scene, reference and evaluator stay fixed."""
import copy
import json
from pathlib import Path

TASK = 'MimicX-G1-Collision-Recovery'


def register(config, windows, noise=.01):
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.tasks.registry import register_mjlab_task
    from src.tasks.tracking.config.g1.env_cfgs import unitree_g1_mimicx_curriculum_env_cfg
    from src.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
    from mimicx.parkour.environment import SceneBuilder, phase_body_tracking
    from mimicx.policy_recovery.environment import world_body_tracking

    scene=json.loads(Path(config['scene']).read_text())
    objective=config.get('objective_route',config['route'])
    fidelity=objective=='world_fidelity'
    event_route=objective=='event_transition'

    def make(play):
        cfg=unitree_g1_mimicx_curriculum_env_cfg(play=play)
        cfg.scene.terrain=None
        cfg.scene.spec_fn=SceneBuilder(config['scene'])
        cfg.scene.extent=12.;cfg.sim.nconmax=192;cfg.sim.njmax=800
        cfg.episode_length_s=max(20.,config['horizon']/50.+2.)
        cfg.viewer.origin_type=cfg.viewer.OriginType.WORLD
        cfg.viewer.lookat=tuple(scene['viewer_lookat'])
        cfg.viewer.distance=scene['viewer_distance']
        cfg.viewer.elevation=-22.;cfg.viewer.azimuth=130.;cfg.viewer.max_extra_envs=0
        cmd=cfg.commands['motion'];cmd.motion_file=config['motion']
        cmd.viz.ground_align=False;cmd.viz.ghost_color=(.831,.357,.298,.23)
        cmd.sampling_mode='start' if play else 'start_window'
        cmd.joint_position_range=(-noise,noise) if play else (-.01,.01)
        if play:
            return cfg
        cmd.start_windows=tuple(tuple(w) for w in windows)
        cfg.rewards['motion_global_root_pos'].weight=4. if fidelity else 2.
        cfg.rewards['motion_global_root_pos'].params['std']=.20 if fidelity else .30
        cfg.rewards['motion_global_root_ori'].weight=1.5 if fidelity else 2.
        cfg.rewards['motion_body_pos'].weight=2.
        cfg.rewards['motion_body_pos'].params['std']=.22 if fidelity else .32
        cfg.rewards['motion_body_ori'].weight=1.5 if fidelity else 2.
        cfg.rewards['motion_body_ori'].params['std']=.4 if fidelity else .5
        cfg.rewards['motion_body_lin_vel'].weight=1. if fidelity else 1.5
        cfg.rewards['motion_body_ang_vel'].weight=1. if fidelity else 1.5
        cfg.rewards['action_rate_l2'].weight=-.05 if fidelity else -.03
        cfg.rewards['recovery_world_body']=RewardTermCfg(func=world_body_tracking,
            weight=1.5 if fidelity else 1.,params=dict(command_name='motion',std=.25 if fidelity else .35))
        groups=[('feet',('left_ankle_roll_link','right_ankle_roll_link'),2.5),
                ('hands',('left_wrist_yaw_link','right_wrist_yaw_link'),2. if config['case']=='bench' else 1.)]
        for name,bodies,weight in groups:
            term=copy.deepcopy(cfg.rewards['motion_body_pos']);term.weight=weight
            term.params.update(body_names=bodies,std=.17 if fidelity else .28)
            cfg.rewards['recovery_'+name]=term
        for name,frames in config['events'].items():
            if name not in ('landing','feet_on_wall','left_palm_on_wall','step_down','roll'):
                continue
            bodies=('left_wrist_yaw_link',) if name=='left_palm_on_wall' else ('left_ankle_roll_link','right_ankle_roll_link')
            if name=='roll':
                bodies=('pelvis','torso_link','left_knee_link','right_knee_link',*bodies)
            cfg.rewards['recovery_event_'+name]=RewardTermCfg(func=phase_body_tracking,
                weight=3. if event_route else 1.5,
                params=dict(command_name='motion',body_names=bodies,
                    start_frame=max(0,frames[0]-10),end_frame=min(config['horizon'],frames[1]+10),
                    std=.30 if event_route else .25))
        if config.get('phase_dynamics',False):
            from .phase_dynamics import add_phase_dynamics
            add_phase_dynamics(cfg,config)
        if config.get('tail_objectives',False):
            from .tail_objectives import add_tail_objectives
            add_tail_objectives(cfg,config)
        return cfg
    register_mjlab_task(task_id=TASK,env_cfg=make(False),play_env_cfg=make(True),
                       rl_cfg=unitree_g1_tracking_ppo_runner_cfg())
    return TASK
