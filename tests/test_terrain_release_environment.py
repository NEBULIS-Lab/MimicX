"""Registration boundaries tested without importing a GPU runtime."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize(('name', 'sha'), [
    ('mimicx/policy_recovery/environment.py', 'd21f736f2ea8f3f52271f3c33aa788432dc3766b12cf228e576d1aac2872f6bc'),
    ('mimicx/parkour/environment.py', 'a8bbdcffae66395bf425539a82e75a00fb909496c07a31210a9e1165cba6e5ca'),
    ('mimicx/parkour/scene_geometry.py', 'ceb2260f5f449da4326a450fc70d06a888cc7d1edf78dd6839c41ad7c64ebf61'),
    ('mimicx/parkour/terrain_environment.py', 'f5f70593aa6b2c732ede10d58d884da6fe41e3a14058f42288d0f2d0aa63abf1'),
])
def test_published_runtime_source_identity(name, sha):
    path = Path(__file__).resolve().parents[1] / name
    assert path.exists(), f'Missing released runtime module: {name}'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha


@pytest.fixture
def registry(monkeypatch):
    captures = []
    template = SimpleNamespace(
        scene=SimpleNamespace(terrain='plane'), sim=SimpleNamespace(),
        viewer=SimpleNamespace(OriginType=SimpleNamespace(WORLD='world')),
        commands={'motion': SimpleNamespace(viz=SimpleNamespace(), joint_position_range=(-.1, .1))},
        rewards={name: SimpleNamespace(weight=1., params={}) for name in (
            'motion_global_root_pos', 'motion_global_root_ori', 'motion_body_pos',
            'motion_body_ori', 'motion_body_lin_vel', 'motion_body_ang_vel', 'action_rate_l2')})
    modules = {
        'mjlab.managers.reward_manager': {'RewardTermCfg': SimpleNamespace},
        'mjlab.tasks.registry': {'register_mjlab_task': lambda **kw: captures.append(kw), 'list_tasks': lambda: []},
        'src.tasks.tracking.config.g1.env_cfgs': {'unitree_g1_mimicx_curriculum_env_cfg': lambda play: copy.deepcopy(template)},
        'src.tasks.tracking.config.g1.rl_cfg': {'unitree_g1_tracking_ppo_runner_cfg': lambda: None},
        'src.tasks.tracking.mdp.rewards': {'motion_reference_stance_foot_slip_error_exp': lambda: None},
        'src.tasks.tracking.rl': {'MotionTrackingOnPolicyRunner': object},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    return captures, template


def scene_file(tmp_path):
    path = tmp_path / 'scene.json'
    path.write_text(json.dumps(dict(viewer_lookat=[1, 2, 3], viewer_distance=4,
        boxes=[dict(name='support', center=[1, 2, .5], half_size=[.5, .5, .5],
                    yaw=0, friction=.8, rgba=[.5, .5, .5, 1])])) )
    return path


def test_recovery_registration_preserves_noise_window_rewards_and_backend_defaults(registry):
    from mimicx.policy_recovery.environment import register
    captures, template = registry
    register(dict(route='world_fidelity', motion='reference.npz', horizon=240), (60, 100), .005)
    train, play = captures[0]['env_cfg'], captures[0]['play_env_cfg']
    assert train.commands['motion'].start_windows == ((60, 100, .30), (0, 239, .20))
    assert play.commands['motion'].sampling_mode == 'start'
    assert play.commands['motion'].joint_position_range == (-.005, .005)
    assert not play.commands['motion'].viz.ground_align
    assert train.rewards['motion_global_root_pos'].weight == 4.
    assert 'recovery_feet' not in play.rewards
    assert template.rewards['motion_global_root_pos'].weight == 1.


def test_terrain_profile_keeps_collision_and_reference_height(registry, tmp_path):
    from mimicx.parkour.terrain_environment import register
    scene = scene_file(tmp_path)
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps(dict(duration=30, sampling='start_window', windows=[[10, 20, .3]],
        refined=True, events_control_frames={'landing': [40, 50]})))
    register(scene, profile)
    train, play = registry[0][0]['env_cfg'], registry[0][0]['play_env_cfg']
    assert train.scene.terrain is None
    assert train.sim.nconmax == 192 and train.sim.njmax == 800
    assert train.scene.spec_fn.scene_path == str(scene)
    assert train.episode_length_s == 32.
    assert train.commands['motion'].start_windows == ((10, 20, .3),)
    assert play.commands['motion'].joint_position_range == (-.01, .01)
    assert not play.commands['motion'].viz.ground_align


def test_collision_recovery_imports_optional_objectives_as_package_modules(registry, tmp_path):
    from mimicx.parkour.recovery_environment import register
    names = ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'left_knee_joint',
             'right_knee_joint', 'waist_yaw_joint', 'waist_pitch_joint', 'waist_roll_joint']
    config = dict(scene=str(scene_file(tmp_path)), motion='reference.npz', route='event_transition',
        case='parkour', horizon=200, events={'landing': [80, 90], 'roll': [90, 110]},
        phase_dynamics=True, tail_objectives=True, tail_start_frame=100, tail_joint_names=names)
    register(config, [[50, 100, .4]])
    train, play = registry[0][0]['env_cfg'], registry[0][0]['play_env_cfg']
    assert train.commands['motion'].start_windows == ((50, 100, .4),)
    assert train.rewards['recovery_event_roll'].weight == 3.
    assert train.rewards['roll_orientation'].weight == 2.
    assert train.rewards['tail_hip_knee_waist'].params['joint_ids'] == list(range(7))
    assert 'tail_joint_pose' not in play.rewards
    assert play.commands['motion'].joint_position_range == (-.01, .01)


def test_scene_builder_keeps_collision_geometry_and_world_coordinates(tmp_path, monkeypatch):
    from mimicx.parkour.environment import SceneBuilder
    module = ModuleType('mujoco')
    module.mjtGeom = SimpleNamespace(mjGEOM_PLANE='plane', mjGEOM_BOX='box')
    monkeypatch.setitem(sys.modules, 'mujoco', module)
    geometries = []
    spec = SimpleNamespace(visual=SimpleNamespace(headlight=SimpleNamespace()),
                           worldbody=SimpleNamespace(add_geom=lambda **kw: geometries.append(kw)))
    SceneBuilder(str(scene_file(tmp_path)))(spec)
    assert geometries[0]['type'] == 'plane'
    assert geometries[1]['pos'] == [1, 2, .5]
    assert geometries[1]['size'] == [.5, .5, .5]
    assert geometries[1]['friction'] == [.8, .005, .0001]


def test_descending_support_and_registered_contact_geometry():
    from mimicx.parkour.scene_geometry import descending_staircase, box_signed_distance, validate_scene
    from mimicx.parkour.registered_contacts import wall_patch_target, contact_envelope
    boxes = descending_staircase(.84)
    validate_scene({'boxes': boxes})
    assert len(boxes) == 2
    assert [2 * box['half_size'][2] for box in boxes] == [.17, .085]
    np.testing.assert_allclose(box_signed_distance([[0, 0, 0], [2, 0, 0]], [0, 0, 0], [1, 1, 1], 0), [-1, 1])
    box = dict(center=[0, 0, .5], half_size=[1, 1, .5], yaw=0.)
    np.testing.assert_allclose(wall_patch_target([2, 2, 5], box), [.95, .95, 1.03])
    assert contact_envelope(8, 10, 20, ramp=4) == .5


@pytest.mark.parametrize('runtime_kind', ['scene_v1', 'terrain_profile'])
def test_worker_evaluates_published_runtime_kind_without_training_config(registry, tmp_path, runtime_kind):
    from mimicx.policy_recovery.worker import register_task
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps({'duration': 4.8}))
    config = dict(runtime_kind=runtime_kind, scene=str(scene_file(tmp_path)),
                  profile=str(profile), motion='motion.npz', horizon=240)
    args = SimpleNamespace(mode='evaluate', window=[0, 100], noise=.01)
    task = register_task(config, args, kind='terrain', practice={'windows': []})
    assert task == ('MimicX-G1-Released-Collision-Policy' if runtime_kind == 'scene_v1'
                    else 'MimicX-G1-Terrain-Reference')
    play = registry[0][0]['play_env_cfg']
    assert play.scene.terrain is None
    assert play.commands['motion'].joint_position_range == (-.01, .01)
    assert not play.commands['motion'].viz.ground_align
    if runtime_kind == 'scene_v1':
        assert play.commands['motion'].motion_file == 'motion.npz'
        assert play.commands['motion'].viz.ghost_color == (.831, .357, .298, .16)
    args.mode = 'train'
    with pytest.raises(ValueError, match='evaluation-only'):
        register_task(config, args, kind='terrain', practice={'windows': []})


def test_initial_reference_refresh_recomputes_observations_without_advancing_time(monkeypatch):
    from mimicx.parkour.environment import install_initial_reference_refresh

    class Tensor(np.ndarray):
        def clone(self):
            return self.copy()

        def expand(self, *shape):
            return np.broadcast_to(self, tuple(old if new == -1 else new for old, new in zip(self.shape, shape))).view(Tensor)

    def tensor(values):
        return np.array(values, dtype=float).view(Tensor)

    command = SimpleNamespace(anchor_pos_w=tensor([[1, 2, 3]]),
        robot_anchor_pos_w=tensor([[10, 20, 30]]), anchor_quat_w=tensor([[1, 0, 0, 0]]),
        robot_anchor_quat_w=tensor([[1, 0, 0, 0]]),
        body_pos_w=tensor([[[1, 2, 3], [2, 2, 4]]]),
        body_quat_w=tensor([[[1, 0, 0, 0], [1, 0, 0, 0]]]),
        cfg=SimpleNamespace(body_names=['pelvis', 'foot']), time_steps=0)

    class Base:
        def reset(self, **kwargs):
            return 'stale', {'reset': True}

    modules = {name: ModuleType(name) for name in (
        'mjlab', 'mjlab.envs', 'mjlab.utils', 'mjlab.utils.lab_api', 'mjlab.utils.lab_api.math')}
    modules['mjlab'].envs = modules['mjlab.envs']
    modules['mjlab.envs'].ManagerBasedRlEnv = Base
    modules['mjlab.utils.lab_api.math'].__dict__.update(
        quat_apply=lambda rotation, positions: positions,
        quat_inv=lambda rotation: rotation, quat_mul=lambda a, b: np.broadcast_arrays(a, b)[1].view(Tensor),
        yaw_quat=lambda rotation: rotation)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    original, refreshed = install_initial_reference_refresh()
    env = refreshed()
    env.command_manager = SimpleNamespace(get_term=lambda name: command)
    histories = []

    def compute(update_history):
        histories.append(update_history)
        return command.body_pos_relative_w.copy()

    env.observation_manager = SimpleNamespace(compute=compute)
    observations, extras = env.reset()
    assert original is Base
    np.testing.assert_array_equal(observations, [[[10, 20, 3], [11, 20, 4]]])
    assert command.time_steps == 0
    assert histories == [False]
    assert extras == {'reset': True}
