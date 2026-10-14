"""Process-local task registration; shared backend defaults remain untouched."""
import copy

TASK = 'MimicX-G1-Policy-Recovery'


def world_body_tracking(env, command_name, std):
    import torch
    command = env.command_manager.get_term(command_name)
    error = (command.body_pos_w - command.robot_body_pos_w).square().sum(-1).mean(-1)
    return torch.exp(-error / (std * std))


def register(config, window, noise=0.):
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.tasks.registry import register_mjlab_task
    from src.tasks.tracking.config.g1.env_cfgs import unitree_g1_mimicx_curriculum_env_cfg
    from src.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
    from src.tasks.tracking.mdp.rewards import motion_reference_stance_foot_slip_error_exp

    fidelity = config['route'] == 'world_fidelity'
    def make(play):
        cfg = unitree_g1_mimicx_curriculum_env_cfg(play=play)
        cmd = cfg.commands['motion']
        cmd.motion_file = config['motion']
        cfg.episode_length_s = config['horizon'] / 50. + 2.
        cmd.sampling_mode = 'start' if play else 'start_window'
        cmd.start_windows = ((int(window[0]), int(window[1]), .30),
                             (0, config['horizon'] - 1, .20))
        cmd.joint_position_range = (-noise, noise) if play else (-.015, .015)
        cfg.viewer.distance = 3.8
        cfg.viewer.elevation = -15.
        cfg.viewer.max_extra_envs = 0
        # No translation/yaw corrections are applied to the visible reference.
        cmd.viz.ground_align = False
        if play:
            return cfg
        cfg.rewards['motion_global_root_pos'].weight = 4. if fidelity else 2.
        cfg.rewards['motion_global_root_pos'].params['std'] = .20 if fidelity else .30
        cfg.rewards['motion_global_root_ori'].weight = 2. if fidelity else 1.
        cfg.rewards['motion_body_pos'].weight = 2.
        cfg.rewards['motion_body_pos'].params['std'] = .20 if fidelity else .30
        cfg.rewards['motion_body_ori'].params['std'] = .4
        cfg.rewards['world_body_tracking'] = RewardTermCfg(
            func=world_body_tracking, weight=2. if fidelity else 1.,
            params=dict(command_name='motion', std=.25 if fidelity else .35))
        feet = ('left_ankle_roll_link', 'right_ankle_roll_link')
        hands = ('left_wrist_yaw_link', 'right_wrist_yaw_link')
        for name, bodies, weight in (('recovery_feet', feet, 2.), ('recovery_hands', hands, 1.5)):
            reward = copy.deepcopy(cfg.rewards['motion_body_pos'])
            reward.weight = weight
            reward.params.update(body_names=bodies, std=.15 if fidelity else .20)
            cfg.rewards[name] = reward
        cfg.rewards['recovery_stance'] = RewardTermCfg(
            func=motion_reference_stance_foot_slip_error_exp, weight=.5,
            params=dict(command_name='motion', std=.3, stance_height=.12,
                        body_names=feet, vertical_weight=.2))
        return cfg
    register_mjlab_task(task_id=TASK, env_cfg=make(False), play_env_cfg=make(True),
                       rl_cfg=unitree_g1_tracking_ppo_runner_cfg())
    return TASK
