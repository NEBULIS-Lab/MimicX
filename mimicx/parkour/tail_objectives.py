"""Training-only tail objectives and completion-protected execution selection."""
import math
import numpy as np


def phase_joint_tracking(env, command_name, joint_ids, start_frame, end_frame, std,
                         expected_joint_names=None):
    command = env.command_manager.get_term(command_name)
    if expected_joint_names is not None and tuple(command.robot.joint_names) != tuple(expected_joint_names):
        raise ValueError('Joint order differs from the recorded reference')
    # G1 hinges have bounded ranges: modulo-2pi errors would hide invalid poses.
    error = (command.joint_pos[:, joint_ids] - command.robot_joint_pos[:, joint_ids]).square()
    active = (command.time_steps >= start_frame) & (command.time_steps < end_frame)
    return active * (1. / (1. + error / (std * std))).mean(-1)


def phase_local_body_tracking(env, command_name, body_names, start_frame, end_frame, std):
    command = env.command_manager.get_term(command_name)
    ids = [command.cfg.body_names.index(name) for name in body_names]
    error = (command.body_pos_relative_w[:, ids] - command.robot_body_pos_w[:, ids]).square().sum(-1)
    active = (command.time_steps >= start_frame) & (command.time_steps < end_frame)
    return active * (1. / (1. + error / (std * std))).mean(-1)


def add_tail_objectives(cfg, config):
    from mjlab.managers.reward_manager import RewardTermCfg
    from .phase_dynamics import phase_orientation_tracking
    start = max(0, config['tail_start_frame'] - 15)
    common = dict(command_name='motion', start_frame=start, end_frame=config['horizon'])
    names = tuple(config['tail_joint_names'])
    hip_names = ('left_hip_pitch_joint', 'right_hip_pitch_joint', 'left_knee_joint',
                 'right_knee_joint', 'waist_yaw_joint', 'waist_pitch_joint', 'waist_roll_joint')
    cfg.rewards['tail_joint_pose'] = RewardTermCfg(func=phase_joint_tracking, weight=2.,
        params=dict(**common, joint_ids=list(range(len(names))), std=.7, expected_joint_names=names))
    cfg.rewards['tail_hip_knee_waist'] = RewardTermCfg(func=phase_joint_tracking, weight=4.,
        params=dict(**common, joint_ids=[names.index(n) for n in hip_names], std=1., expected_joint_names=names))
    bodies = ('left_knee_link', 'right_knee_link', 'left_ankle_roll_link',
              'right_ankle_roll_link', 'left_wrist_yaw_link', 'right_wrist_yaw_link')
    cfg.rewards['tail_articulation'] = RewardTermCfg(func=phase_local_body_tracking, weight=3.,
        params=dict(**common, body_names=bodies, std=.4))
    cfg.rewards['tail_world_orientation'] = RewardTermCfg(func=phase_orientation_tracking, weight=3.,
        params=dict(**common, body_names=('pelvis','torso_link'), std=1.))


def tail_windows(config, block):
    roll, end = config['tail_start_frame'], config['horizon']
    early = block < config['blocks'] // 2
    values = [(config['events']['landing'][0]-5, roll+5, .20 if early else .10),
              (roll-15, end, .35 if early else .15), (0,end,.10)]
    return [[max(0,a),min(end,b),p] for a,b,p in values]


def tail_metrics(rows, config):
    from .metrics import summarize
    summarize(rows, config['horizon'])
    start = config['tail_start_frame']
    if not 0 < start < len(rows):
        raise ValueError('Tail interval must leave a nonempty protected prefix')
    keys = ('body_pos_error_max', 'error_anchor_rot', 'error_joint_pos')
    values = np.array([[float(r[k]) for k in keys] for r in rows], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite phase diagnostic')
    tail = values[start:].copy()
    tail[:,2] /= math.sqrt(config['tail_joint_count'])
    return dict(start_frame=start, samples=len(tail),
        body_max_mean_m=float(tail[:,0].mean()), body_max_p95_m=float(np.quantile(tail[:,0],.95)),
        anchor_rot_mean_rad=float(tail[:,1].mean()), joint_rms_mean_rad=float(tail[:,2].mean()),
        prefix_body_p95_m=float(np.quantile(values[:start,0],.95)))


def tail_decision(old, new, config, require_improvement=True):
    from .metrics import decision
    base = decision(old,new,config['horizon'],require_improvement=False)
    a = [tail_metrics(r,config) for r in old]
    b = [tail_metrics(r,config) for r in new]
    keys = ('body_max_mean_m','anchor_rot_mean_rad','joint_rms_mean_rad')
    # Normalize channels by the paired incumbent, not by arbitrarily mixed units.
    ratios = np.array([[(y[k]+1e-12)/(x[k]+1e-12) for k in keys] for x,y in zip(a,b)])
    gain = 1. - float(ratios.mean())
    complete = all(r['complete'] for r in base['old']+base['new'])
    tail_guard = all(y[k] <= 1.10*x[k]+(.005 if k.endswith('_m') else .02)
                     for x,y in zip(a,b) for k in keys)
    prefix_guard = all(y['prefix_body_p95_m'] <= 1.05*x['prefix_body_p95_m']+.005 for x,y in zip(a,b))
    quality = gain >= (.03 if require_improvement else -.005)
    base.update(accepted=bool(base['accepted'] and complete and tail_guard and prefix_guard and quality),
        gate_version='complete-tail-fidelity-v1', tail_old=a, tail_new=b,
        tail_relative_gain=gain, tail_tracking_guards=bool(tail_guard),
        protected_prefix_guard=bool(prefix_guard), complete_required=bool(complete),
        tail_quality_improved=bool(gain>=.03), tail_quality_condition=bool(quality))
    return base
