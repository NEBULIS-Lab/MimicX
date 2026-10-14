"""Phase-local rotational tracking and landing-to-roll reference practice."""


def phase_orientation_tracking(env, command_name, body_names, start_frame, end_frame, std):
    from mjlab.utils.lab_api.math import quat_error_magnitude
    command=env.command_manager.get_term(command_name)
    ids=[command.cfg.body_names.index(name) for name in body_names]
    error=quat_error_magnitude(command.body_quat_w[:,ids],command.robot_body_quat_w[:,ids]).square().mean(-1)
    active=(command.time_steps>=start_frame)&(command.time_steps<end_frame)
    # A bounded rational kernel retains a useful score far from the reference.
    return active/(1.+error/(std*std))


def phase_angular_velocity_tracking(env, command_name, body_names, start_frame, end_frame, std):
    command=env.command_manager.get_term(command_name)
    ids=[command.cfg.body_names.index(name) for name in body_names]
    error=(command.body_ang_vel_w[:,ids]-command.robot_body_ang_vel_w[:,ids]).square().sum(-1).mean(-1)
    active=(command.time_steps>=start_frame)&(command.time_steps<end_frame)
    return active/(1.+error/(std*std))


def bridge_windows(config, block):
    landing=config['events']['landing'][0]
    roll=config['events']['roll'][0]
    horizon=config['horizon']
    early=block<config['blocks']//2
    values=[(landing-15,roll+5,.30 if early else .20),
            (roll-5,roll+25,.25 if early else .15),(0,horizon,.10)]
    return [[max(0,a),min(horizon,b),p] for a,b,p in values]


def add_phase_dynamics(cfg, config):
    from mjlab.managers.reward_manager import RewardTermCfg
    common=dict(command_name='motion',body_names=('pelvis','torso_link'),
                start_frame=config['events']['landing'][0],end_frame=config['horizon'])
    cfg.rewards['roll_orientation']=RewardTermCfg(func=phase_orientation_tracking,
        weight=2.,params=dict(**common,std=.9))
    cfg.rewards['roll_angular_velocity']=RewardTermCfg(func=phase_angular_velocity_tracking,
        weight=1.,params=dict(**common,std=6.))
