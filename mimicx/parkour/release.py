"""The frozen scene-v1 evaluation registration, without an external script loader."""
import json


def register_evaluation(row, paths):
    from mjlab.tasks.registry import register_mjlab_task
    from src.tasks.tracking.config.g1.env_cfgs import unitree_g1_mimicx_curriculum_env_cfg
    from src.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
    from mimicx.parkour.environment import SceneBuilder
    scene = json.loads(paths['scene'].read_text())
    cfg = unitree_g1_mimicx_curriculum_env_cfg(play=True)
    cfg.scene.terrain = None
    cfg.scene.spec_fn = SceneBuilder(str(paths['scene']))
    cfg.scene.extent = 12.
    cfg.sim.nconmax = 192
    cfg.sim.njmax = 800
    cfg.episode_length_s = max(20., row['horizon'] / 50. + 2.)
    cfg.viewer.origin_type = cfg.viewer.OriginType.WORLD
    cfg.viewer.lookat = tuple(scene['viewer_lookat'])
    cfg.viewer.distance = scene['viewer_distance']
    cfg.viewer.elevation = -22.
    cfg.viewer.azimuth = 130.
    cfg.viewer.max_extra_envs = 0
    cmd = cfg.commands['motion']
    cmd.motion_file = str(paths['motion'])
    cmd.viz.ground_align = False
    cmd.viz.ghost_color = (.831, .357, .298, .16)
    cmd.sampling_mode = 'start'
    cmd.joint_position_range = (-.01, .01)
    task = 'MimicX-G1-Released-Collision-Policy'
    register_mjlab_task(task_id=task, env_cfg=cfg, play_env_cfg=cfg,
                       rl_cfg=unitree_g1_tracking_ppo_runner_cfg())
    return task
