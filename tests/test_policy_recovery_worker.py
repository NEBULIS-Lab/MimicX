"""Portable worker contracts without loading the optional simulator stack."""
from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest


def test_public_imports_and_cli_help_do_not_load_simulation_dependencies():
    code = '''
import importlib.abc, sys
class BlockSimulation(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'mjlab', 'mujoco', 'warp', 'rsl_rl'}:
            raise AssertionError('Unexpected heavy import: ' + fullname)
sys.meta_path.insert(0, BlockSimulation())
import mimicx.policy_recovery.environment
import mimicx.policy_recovery.metrics
import mimicx.policy_recovery.resume
import mimicx.policy_recovery.worker
import mimicx.parkour.environment
import mimicx.parkour.terrain_environment
import mimicx.parkour.recovery_environment
import mimicx.parkour.registered_contacts
import mimicx.parkour.phase_dynamics
import mimicx.parkour.tail_objectives
import scripts.policy_recovery.campaign
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for script in ('scripts/policy_recovery/worker.py', 'scripts/parkour/worker.py',
                   'scripts/policy_recovery/campaign.py'):
        result = subprocess.run([sys.executable, script, '--help'], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert '--config' in result.stdout


@pytest.mark.parametrize(('saved', 'completed', 'remaining', 'next_iteration'), [
    (None, 0, 750, 7496), (7750, 255, 495, 7751), (8245, 750, 0, 8246),
])
def test_resume_counts_saved_updates_once(saved, completed, remaining, next_iteration):
    from mimicx.policy_recovery.resume import remaining_updates
    assert remaining_updates(7496, 750, saved) == dict(
        completed_updates=completed, remaining_updates=remaining, next_iteration=next_iteration)


@pytest.mark.parametrize('saved', [7495, 8246])
def test_resume_rejects_checkpoint_outside_block(saved):
    from mimicx.policy_recovery.resume import remaining_updates
    with pytest.raises(ValueError):
        remaining_updates(7496, 750, saved)


def test_atomic_save_preserves_previous_checkpoint_on_interruption(tmp_path):
    from mimicx.policy_recovery.resume import atomic_save
    path = tmp_path / 'model_10.pt'
    path.write_bytes(b'previous')

    def interrupted(temporary, infos=None):
        Path(temporary).write_bytes(b'partial')
        raise OSError('interrupted')

    with pytest.raises(OSError):
        atomic_save(interrupted, path)
    assert path.read_bytes() == b'previous'
    assert not list(tmp_path.glob('*.pending'))

    def complete(temporary, infos=None):
        Path(temporary).write_text(infos['value'])

    atomic_save(complete, path, infos={'value': 'complete'})
    assert path.read_text() == 'complete'


def test_resume_skips_corruption_and_ignores_unrelated_filenames(tmp_path, monkeypatch):
    from mimicx.policy_recovery import resume
    monkeypatch.setattr(resume, 'load_checkpoint', lambda path: json.loads(Path(path).read_text()))
    state = dict(actor_state_dict={}, critic_state_dict={}, optimizer_state_dict={}, iter=7750)
    (tmp_path / 'model_7750.pt').write_text(json.dumps(state))
    (tmp_path / 'model_7800.pt').write_text('interrupted')
    (tmp_path / 'model_best.pt').write_text('unrelated')
    (tmp_path / '.model_7850.pt.pending').write_text('incomplete')
    result = resume.checkpoint_plan(tmp_path, 7496, 750)
    assert result['checkpoint'] == str(tmp_path / 'model_7750.pt')
    assert result['remaining_updates'] == 495
    assert len(result['invalid_checkpoints']) == 1


def test_invalid_checkpoints_do_not_restart_budget(tmp_path, monkeypatch):
    from mimicx.policy_recovery import resume
    monkeypatch.setattr(resume, 'load_checkpoint', lambda path: {'iter': 7750})
    (tmp_path / 'model_7750.pt').touch()
    with pytest.raises(ValueError, match='No valid resume checkpoint'):
        resume.checkpoint_plan(tmp_path, 7496, 750)


def test_final_checkpoint_must_represent_full_block(tmp_path, monkeypatch):
    from mimicx.policy_recovery import resume
    state = dict(actor_state_dict={}, critic_state_dict={}, optimizer_state_dict={}, iter=7750)
    monkeypatch.setattr(resume, 'load_checkpoint', lambda path: state)
    (tmp_path / 'final.pt').touch()
    with pytest.raises(ValueError, match='No valid resume checkpoint'):
        resume.checkpoint_plan(tmp_path, 7496, 750)


def test_config_paths_resolve_relative_to_config_not_process_cwd(tmp_path, monkeypatch):
    from mimicx.policy_recovery.worker import load_config, runtime_paths
    (tmp_path / 'backend' / 'src').mkdir(parents=True)
    (tmp_path / 'native' / 'scripts').mkdir(parents=True)
    (tmp_path / 'native' / 'scripts' / 'rollout_mjlab_motion_smoke.py').touch()
    config = tmp_path / 'run.json'
    config.write_text(json.dumps(dict(backend_root='backend', native_root='native',
                                     motion='motion.npz', scene='scene.json')))
    monkeypatch.chdir(tmp_path.parent)
    loaded = load_config(config)
    assert loaded['motion'] == str(tmp_path / 'motion.npz')
    assert loaded['scene'] == str(tmp_path / 'scene.json')
    paths = runtime_paths(loaded)
    assert paths['backend'] == tmp_path / 'backend'
    assert paths['rollout_script'] == tmp_path / 'native/scripts/rollout_mjlab_motion_smoke.py'


@pytest.mark.parametrize(('kind', 'seed_key', 'decay', 'interval'), [
    ('recovery', 'train_seed', .85, 250), ('terrain', 'training_seed', .9, 50),
])
def test_training_preserves_source_schedule_and_seed(kind, seed_key, decay, interval):
    from mimicx.policy_recovery.worker import configure_training
    cfg = SimpleNamespace(scene=SimpleNamespace())
    agent = SimpleNamespace(algorithm=SimpleNamespace())
    config = {seed_key: 101, 'learning_rate': .0001, 'num_envs': 64, 'route': 'world_fidelity'}
    lr = configure_training(cfg, agent, config, block=2, kind=kind)
    assert lr == pytest.approx(.0001 * decay ** 2)
    assert cfg.seed == agent.seed == 103
    assert cfg.scene.num_envs == 64
    assert agent.algorithm.schedule == 'fixed'
    assert agent.algorithm.entropy_coef == .001
    assert agent.save_interval == interval


@pytest.mark.parametrize(('block', 'checkpoint', 'load_optimizer'), [
    (0, None, False), (1, None, True), (0, 'partial.pt', True),
])
def test_resume_restores_counter_and_overwrites_all_optimizer_learning_rates(block, checkpoint, load_optimizer):
    from mimicx.policy_recovery.worker import restore_runner
    calls = []
    runner = SimpleNamespace(alg=SimpleNamespace(learning_rate=.9,
        optimizer=SimpleNamespace(param_groups=[{'lr': .9}, {'lr': .8}])),
        load=lambda path, **kwargs: calls.append((path, kwargs)), save=lambda *a, **kw: None)
    plan = {'checkpoint': checkpoint, 'next_iteration': 7751}
    restore_runner(runner, plan, Path('initial.pt'), block=block, learning_rate=.00003, device='cpu')
    assert calls[0][0] == (checkpoint or 'initial.pt')
    assert calls[0][1]['load_cfg'] == dict(actor=True, critic=True, optimizer=load_optimizer, iteration=True)
    assert runner.current_learning_iteration == 7751
    assert runner.alg.learning_rate == .00003
    assert [group['lr'] for group in runner.alg.optimizer.param_groups] == [.00003, .00003]


def test_immutable_resume_contract_rejects_changed_budget_and_missing_receipt(tmp_path):
    from mimicx.policy_recovery.worker import ensure_training_contract
    actual = {'inherited_iteration': 100, 'additional_updates': 50, 'learning_rate': .0001}
    assert ensure_training_contract(tmp_path, actual) == actual
    assert ensure_training_contract(tmp_path, actual) == actual
    with pytest.raises(ValueError, match='Resume contract changed'):
        ensure_training_contract(tmp_path, {**actual, 'additional_updates': 100})
    (tmp_path / 'effective_training.json').unlink()
    (tmp_path / 'model_125.pt').touch()
    with pytest.raises(ValueError, match='metadata'):
        ensure_training_contract(tmp_path, actual)


def test_evaluation_preserves_horizon_noise_and_original_reference_height():
    from mimicx.policy_recovery.worker import evaluation_argv
    args = SimpleNamespace(checkpoint=Path('checkpoint.pt'), output=Path('eval'),
                           seed=1001, device='cpu', video=True, no_ghost=True)
    argv = evaluation_argv(Path('rollout.py'), 'task', {'motion': 'motion.npz', 'horizon': 240},
                           args, kind='terrain')
    assert argv[argv.index('--steps') + 1] == '240'
    assert argv[argv.index('--seed') + 1] == '1001'
    assert '--no-reference-visual-ground-align' in argv
    assert '--hide-reference-visual' in argv
    assert '--replay-output' in argv
    assert '--disable-terminations' not in argv
    assert '--disable-joint-init-noise' not in argv


def test_core_evaluation_always_exports_replay_without_video():
    from mimicx.policy_recovery.worker import evaluation_argv
    args = SimpleNamespace(checkpoint=Path('checkpoint.pt'), output=Path('eval'),
                           seed=1001, device='cpu', video=False, no_ghost=False)
    argv = evaluation_argv(Path('rollout.py'), 'task', {'motion': 'motion.npz', 'horizon': 240},
                           args, kind='recovery')
    assert argv[argv.index('--replay-output') + 1] == 'eval/replay'
    assert '--video-output' not in argv


@pytest.mark.parametrize('kind', ['recovery', 'terrain'])
def test_evaluation_uses_latest_release_ghost_alpha(kind):
    from mimicx.policy_recovery.worker import evaluation_argv
    args = SimpleNamespace(checkpoint=Path('checkpoint.pt'), output=Path('eval'),
                           seed=1001, device='cpu', video=True, no_ghost=False)
    argv = evaluation_argv(Path('rollout.py'), 'task', {'motion': 'motion.npz', 'horizon': 240},
                           args, kind=kind)
    assert argv[argv.index('--ghost-color') + 1] == '0.831,0.357,0.298,0.16'


def test_evaluation_refuses_existing_output_before_loading_simulator(tmp_path):
    from mimicx.policy_recovery.worker import main
    config = tmp_path / 'config.json'
    config.write_text('{}')
    output = tmp_path / 'evaluation'
    output.mkdir()
    sentinel = output / 'steps.csv'
    sentinel.write_text('original measurements')
    with pytest.raises(FileExistsError):
        main(['--config', str(config), '--mode', 'evaluate', '--checkpoint', 'unused.pt',
              '--output', str(output)])
    assert sentinel.read_text() == 'original measurements'


def test_evaluation_rejects_inherited_device_override_before_loading_simulator(tmp_path, monkeypatch):
    from mimicx.policy_recovery.worker import main
    config = tmp_path / 'config.json'
    config.write_text('{}')
    monkeypatch.setenv('MIMICX_DEVICE_ID', '1')
    with pytest.raises(ValueError, match='MIMICX_DEVICE_ID'):
        main(['--config', str(config), '--mode', 'evaluate', '--checkpoint', 'unused.pt',
              '--output', str(tmp_path / 'evaluation')])
    assert not (tmp_path / 'evaluation').exists()


def test_interrupted_training_resumes_remaining_updates_then_becomes_noop(tmp_path, monkeypatch):
    from mimicx.policy_recovery import worker, resume

    @dataclass
    class Scene:
        num_envs: int = 0

    @dataclass
    class EnvConfig:
        scene: Scene = field(default_factory=Scene)
        seed: int = 0

    @dataclass
    class Algorithm:
        learning_rate: float = 1.
        entropy_coef: float = 0.
        schedule: str = 'adaptive'

    @dataclass
    class Agent:
        algorithm: Algorithm = field(default_factory=Algorithm)
        num_steps_per_env: int = 24
        clip_actions: float = 1.
        seed: int = 0
        logger: str = ''
        upload_model: bool = True
        save_interval: int = 1

    executed, environments, loaded_optimizer = [], [], []
    interrupted = [False]

    class Env:
        def __init__(self, cfg, device):
            assert device == 'cpu'
            environments.append(self)
            self.closed = False
            self.unwrapped = SimpleNamespace(action_space=SimpleNamespace(shape=(2, 29)))

        def get_observations(self):
            return SimpleNamespace(shape=(2, 100))

        def close(self):
            self.closed = True

    class Runner:
        def __init__(self, env, cfg, output, device):
            assert device == 'cpu'
            self.output = Path(output)
            self.current_learning_iteration = 0
            self.alg = SimpleNamespace(learning_rate=.9,
                optimizer=SimpleNamespace(param_groups=[{'lr': .9}, {'lr': .8}]))

        def load(self, path, map_location, load_cfg):
            assert map_location == 'cpu'
            loaded_optimizer.append(load_cfg['optimizer'])
            self.current_learning_iteration = json.loads(Path(path).read_text())['iter']

        def learn(self, count, init_at_random_ep_len):
            assert not init_at_random_ep_len
            assert [g['lr'] for g in self.alg.optimizer.param_groups] == [.0001, .0001]
            start = self.current_learning_iteration
            for iteration in range(start, start + count):
                executed.append(iteration)
                self.current_learning_iteration = iteration
                self.save(self.output / f'model_{iteration}.pt')
                if iteration == 101 and not interrupted[0]:
                    interrupted[0] = True
                    raise RuntimeError('simulated interruption')

        def save(self, path, infos=None):
            Path(path).write_text(json.dumps(dict(iter=self.current_learning_iteration,
                actor_state_dict={}, critic_state_dict={}, optimizer_state_dict={}, infos=infos)))

    modules = {
        'mjlab.envs': {'ManagerBasedRlEnv': Env},
        'mjlab.rl': {'MjlabOnPolicyRunner': Runner, 'RslRlVecEnvWrapper': lambda env, clip_actions: env},
        'mjlab.tasks.registry': {'load_env_cfg': lambda task: EnvConfig(), 'load_rl_cfg': lambda task: Agent()},
        'mjlab.utils.os': {'dump_yaml': lambda path, value: None},
        'mjlab.utils.torch': {'configure_torch_backends': lambda: None},
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    read = lambda path: json.loads(Path(path).read_text())
    monkeypatch.setattr(worker, 'load_checkpoint', read)
    monkeypatch.setattr(resume, 'load_checkpoint', read)
    checkpoint = tmp_path / 'initial.pt'
    checkpoint.write_text(json.dumps({'iter': 100}))
    motion = tmp_path / 'motion.npz'
    motion.write_bytes(b'reference remains fixed')
    config = dict(train_seed=101, learning_rate=.0001, num_envs=2, route='execution_recovery',
                  updates_per_block=4, motion=str(motion))
    args = SimpleNamespace(checkpoint=checkpoint, output=tmp_path / 'training', block=0, device='cpu')
    with pytest.raises(RuntimeError, match='simulated interruption'):
        worker.train(config, args, 'task', kind='recovery', practice={'window': [0, 100]})
    result = worker.train(config, args, 'task', kind='recovery', practice={'window': [0, 100]})
    assert executed == [100, 101, 102, 103]
    assert loaded_optimizer == [False, True]
    assert result['executed_updates'] == 2
    assert result['executed_transitions'] == 96
    assert result['resume']['completed_updates'] == 2
    assert read(args.output / 'final.pt')['iter'] == 103
    assert read(args.output / 'effective_training.json')['additional_transitions'] == 192
    result = worker.train(config, args, 'task', kind='recovery', practice={'window': [0, 100]})
    assert result['status'] == 'already_trained'
    assert result['executed_updates'] == 0
    assert len(environments) == 2
    assert all(env.closed for env in environments)
    assert not list(args.output.glob('*.pending'))
    # A completed periodic checkpoint is sufficient even if finalization was interrupted.
    (args.output / 'final.pt').unlink()
    result = worker.train(config, args, 'task', kind='recovery', practice={'window': [0, 100]})
    assert result['executed_updates'] == 0
    assert read(args.output / 'final.pt')['iter'] == 103
    assert len(environments) == 2


def test_campaign_keeps_evaluation_outputs_new_and_reuses_verified_stages(tmp_path, monkeypatch):
    import csv
    from scripts.policy_recovery import campaign
    from tests.test_policy_recovery_metrics import rows

    backend = tmp_path / 'backend'
    (backend / 'src').mkdir(parents=True)
    checkpoint, motion = tmp_path / 'initial.pt', tmp_path / 'motion.npz'
    checkpoint.write_bytes(b'initial')
    motion.write_bytes(b'motion')
    config_path = tmp_path / 'config.json'
    config = dict(backend_root=str(backend), checkpoint=str(checkpoint), motion=str(motion),
        output=str(tmp_path / 'campaign'), task='test', route='execution_recovery', horizon=40,
        blocks=1, updates_per_block=4, selection_seeds=[1, 2, 3], holdout_seeds=[11, 12, 13])
    config_path.write_text(json.dumps(config))
    calls = []

    def run(command, **kwargs):
        mode = command[command.index('--mode') + 1]
        output = Path(command[command.index('--output') + 1])
        calls.append(mode)
        if mode == 'train':
            output.mkdir(parents=True, exist_ok=True)
            (output / 'final.pt').write_bytes(b'trained')
            (output / 'effective_training.json').write_text('{}')
            return
        assert not output.exists(), 'An evaluation must never overwrite a previous output'
        output.mkdir(parents=True)
        with (output / 'steps.csv').open('w') as stream:
            data = rows(error=.03 if 'block00' in str(output) else .04)
            writer = csv.DictWriter(stream, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        (output / 'metrics.json').write_text('{}')
        if '--video' in command:
            (output / 'policy.mp4').write_bytes(b'video')

    monkeypatch.setattr(campaign.subprocess, 'run', run)
    monkeypatch.setattr(sys, 'argv', ['campaign.py', '--config', str(config_path)])
    campaign.main()
    count = len(calls)
    report = json.loads((tmp_path / 'campaign/final_report.json').read_text())
    assert report['completed_updates'] == 4
    assert len(report['selection']) == 3 and len(report['holdout']) == 3
    assert report['checkpoint'].endswith('block00/train/final.pt')
    campaign.main()
    assert len(calls) == count
    motion.write_bytes(b'new reference at the same path')
    with pytest.raises(RuntimeError, match='mismatched stage'):
        campaign.main()
    assert len(calls) == count
