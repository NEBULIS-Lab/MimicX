"""Portable continuation stages using the recorded PPO implementation and defaults."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import sys
import time

from .resume import atomic_save, checkpoint_plan, digest, load_checkpoint, save_json


def load_config(path):
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text())
    for key in ('backend_root', 'native_root', 'snapshot', 'rollout_script', 'source_hashes',
                'motion', 'scene', 'profile', 'checkpoint', 'output', 'campaign'):
        if config.get(key) is not None:
            value = Path(config[key]).expanduser()
            config[key] = str((path.parent / value).resolve())
    return config


def runtime_paths(config, backend_root=None, native_root=None):
    snapshot = Path(config['snapshot']) if config.get('snapshot') else None
    backend = backend_root or config.get('backend_root') or (snapshot / 'backend' if snapshot else None)
    native = native_root or config.get('native_root') or (snapshot / 'native' if snapshot else Path(__file__).resolve().parents[2])
    if backend is None:
        raise ValueError('Set backend_root in the configuration or pass --backend-root')
    backend, native = Path(backend).expanduser().resolve(), Path(native).expanduser().resolve()
    if not (backend / 'src').is_dir():
        raise FileNotFoundError(f'Backend src directory not found: {backend / "src"}')
    script = Path(config.get('rollout_script', native / 'scripts/rollout_mjlab_motion_smoke.py'))
    return dict(backend=backend, native=native, rollout_script=script)


def configure_training(cfg, agent, config, block, kind):
    if block < 0 or kind not in ('recovery', 'terrain'):
        raise ValueError('Invalid training block or task family')
    terrain = kind == 'terrain'
    cfg.seed = agent.seed = config['training_seed' if terrain else 'train_seed'] + block
    cfg.scene.num_envs = config['num_envs']
    agent.logger = 'tensorboard'
    agent.upload_model = False
    agent.save_interval = 50 if terrain else 250
    agent.algorithm.schedule = 'fixed'
    agent.algorithm.entropy_coef = (.001 if config['route'] == 'world_fidelity' else .002) if terrain else (
        .002 if config['route'] == 'execution_recovery' else .001)
    lr = config['learning_rate'] * ((.9 if terrain else .85) ** block)
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError('Learning rate must be finite and positive')
    agent.algorithm.learning_rate = lr
    return lr


def restore_runner(runner, plan, checkpoint, *, block, learning_rate, device):
    runner.load(plan['checkpoint'] or str(checkpoint), map_location=device, load_cfg={
        'actor': True, 'critic': True, 'optimizer': bool(plan['checkpoint']) or block > 0,
        'iteration': True})
    # Saved iter is the last completed update. Fresh blocks retain the source's
    # inherited-index convention; restarts of the same block start at saved + 1.
    runner.current_learning_iteration = plan['next_iteration']
    original_save = runner.save
    runner.save = lambda path, infos=None: atomic_save(original_save, path, infos)
    runner.alg.learning_rate = learning_rate
    for group in runner.alg.optimizer.param_groups:
        group['lr'] = learning_rate


def ensure_training_contract(output, actual):
    output = Path(output)
    previous = output / 'effective_training.json'
    if previous.exists():
        original = json.loads(previous.read_text())
        if original != actual:
            changed = sorted(key for key in original.keys() | actual.keys() if original.get(key) != actual.get(key))
            raise ValueError('Resume contract changed: ' + ', '.join(changed))
        return original
    if any(output.glob('*.pt')):
        raise ValueError('Partial checkpoints lack original training metadata')
    save_json(previous, actual)
    return actual


def train(config, args, task, *, kind, practice):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends

    cfg, agent = load_env_cfg(task), load_rl_cfg(task)
    lr = configure_training(cfg, agent, config, args.block, kind)
    inherited = int(load_checkpoint(args.checkpoint)['iter'])
    budget = config['updates_per_block']
    actual = dict(learning_rate=lr, checkpoint=str(args.checkpoint),
        checkpoint_sha256=digest(args.checkpoint), inherited_iteration=inherited,
        additional_updates=budget, num_envs=config['num_envs'], steps_per_env=agent.num_steps_per_env,
        additional_transitions=budget * config['num_envs'] * agent.num_steps_per_env,
        block=args.block, kind=kind, practice=practice, config=config,
        inputs_sha256={key: digest(config[key]) for key in ('motion', 'scene', 'profile') if key in config})
    ensure_training_contract(args.output, actual)
    plan = checkpoint_plan(args.output, inherited, budget)
    attempt = dict(started=time.time(), resume=plan, device=args.device,
        resume_sha256=digest(plan['checkpoint']) if plan['checkpoint'] else None,
        save_interval=agent.save_interval, bitwise_resume=False,
        environment_state_resume='Counter restored; physical episodes restart')
    receipt = args.output / 'attempts' / f'{time.time_ns()}.json'
    save_json(receipt, attempt)
    if not plan['remaining_updates']:
        if Path(plan['checkpoint']) != args.output / 'final.pt':
            atomic_save(lambda path, infos=None: shutil.copy2(plan['checkpoint'], path), args.output / 'final.pt')
        attempt.update(finished=time.time(), status='already_trained', executed_updates=0, executed_transitions=0)
        save_json(receipt, attempt)
        return attempt
    configure_torch_backends()
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=args.device), clip_actions=agent.clip_actions)
    try:
        runner = MjlabOnPolicyRunner(env, asdict(agent), str(args.output), device=args.device)
        restore_runner(runner, plan, args.checkpoint, block=args.block, learning_rate=lr, device=args.device)
        dump_yaml(args.output / 'params/env.yaml', asdict(cfg))
        dump_yaml(args.output / 'params/agent.yaml', asdict(agent))
        attempt['optimizer_lrs'] = [group['lr'] for group in runner.alg.optimizer.param_groups]
        attempt['observation_shape'] = str(env.get_observations().shape)
        attempt['action_shape'] = list(env.unwrapped.action_space.shape)
        save_json(receipt, attempt)
        runner.learn(plan['remaining_updates'], init_at_random_ep_len=False)
        if runner.current_learning_iteration != inherited + budget - 1:
            raise RuntimeError('Runner counter does not match the requested training budget')
        runner.save(str(args.output / 'final.pt'), infos={
            'terrain_recovery' if kind == 'terrain' else 'recovery': actual})
        attempt.update(finished=time.time(), status='completed', executed_updates=plan['remaining_updates'],
            executed_transitions=plan['remaining_updates'] * config['num_envs'] * agent.num_steps_per_env)
        save_json(receipt, attempt)
        return attempt
    except BaseException as exc:
        attempt.update(finished=time.time(), status='interrupted', error=repr(exc))
        save_json(receipt, attempt)
        raise
    finally:
        env.close()


def evaluation_argv(script, task, config, args, *, kind):
    argv = [str(script), '--task', task, '--motion-file', config['motion'],
        '--checkpoint-file', str(args.checkpoint), '--num-envs', '1', '--steps', str(config['horizon']),
        '--seed', str(args.seed), '--device', args.device, '--no-reference-visual-ground-align',
        '--metrics-output', str(args.output / 'metrics.json'), '--step-metrics-output', str(args.output / 'steps.csv'),
        '--replay-output', str(args.output / 'replay')]
    if args.video:
        argv += ['--video-output', str(args.output / 'policy.mp4'), '--video-width', '1920',
            '--video-height', '1080', '--video-fps', '50', '--screenshot-dir', str(args.output / 'frames'),
            '--screenshot-every', str(max(1, config['horizon'] // 8)), '--ghost-color',
            '0.831,0.357,0.298,0.16']
        if args.no_ghost:
            argv += ['--hide-reference-visual']
    return argv


def register_task(config, args, *, kind, practice):
    runtime_kind = config.get('runtime_kind')
    if runtime_kind in ('scene_v1', 'terrain_profile'):
        if args.mode != 'evaluate':
            raise ValueError('Published scene/profile registrations are evaluation-only in this worker')
        if args.noise != .01:
            raise ValueError('Published scene/profile evaluation requires noise=.01')
        if runtime_kind == 'scene_v1':
            from mimicx.parkour.release import register_evaluation
            return register_evaluation(config, {key: Path(config[key]) for key in ('scene', 'motion')})
        from mimicx.parkour.terrain_environment import register
        return register(config['scene'], config['profile'])
    if runtime_kind not in (None, 'core_recovery', 'terrain_recovery'):
        raise ValueError(f'Unknown runtime_kind: {runtime_kind}')
    if kind == 'terrain':
        if runtime_kind == 'core_recovery':
            raise ValueError('core_recovery requires the policy_recovery worker')
        from mimicx.parkour.recovery_environment import register
        return register(config, practice['windows'], noise=args.noise)
    if runtime_kind == 'terrain_recovery':
        raise ValueError('terrain_recovery requires the parkour worker')
    from .environment import register
    return register(config, args.window, args.noise)


def main(argv=None, *, kind='recovery'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--mode', choices=['train', 'evaluate'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend-root', type=Path)
    parser.add_argument('--native-root', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--window', type=int, nargs=2, default=[0, 100])
    parser.add_argument('--practice', type=Path)
    parser.add_argument('--block', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1001)
    parser.add_argument('--noise', type=float, help='Defaults to joint_init_noise in config, otherwise the source worker default')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--no-ghost', action='store_true')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    args.checkpoint, args.output = args.checkpoint.expanduser().resolve(), args.output.expanduser().resolve()
    if args.mode == 'evaluate':
        if args.output.exists():
            raise FileExistsError(args.output)
        if 'MIMICX_DEVICE_ID' in os.environ:
            raise ValueError('Unset MIMICX_DEVICE_ID and select the process-visible device with --device')
    if args.noise is None:
        args.noise = config.get('joint_init_noise', .01 if kind == 'terrain' else .005)
    if args.mode == 'evaluate' and os.environ.get('MIMICX_AUTOREFINE_PATCH_FILE'):
        raise ValueError('Unset MIMICX_AUTOREFINE_PATCH_FILE for fixed-reference evaluation')
    paths = runtime_paths(config, args.backend_root, args.native_root)
    config.update(backend_root=str(paths['backend']), native_root=str(paths['native']),
                  rollout_script=str(paths['rollout_script']))
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.mode == 'evaluate' and not paths['rollout_script'].is_file():
        raise FileNotFoundError(paths['rollout_script'])
    practice = json.loads(args.practice.read_text()) if args.practice else {'windows': []}
    # Keep the public mimicx package in use; native_root selects only the rollout
    # script. Backend roots expose src.tasks without synthetic module packages.
    sys.path[:0] = [str(paths['backend']), str(paths['backend'] / 'src')]
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    import mjlab.tasks
    import src.tasks
    from mimicx.parkour.environment import install_initial_reference_refresh
    task = register_task(config, args, kind=kind, practice=practice)
    if kind != 'terrain':
        practice = {'window': args.window}
    install_initial_reference_refresh()
    args.output.mkdir(parents=True, exist_ok=args.mode == 'train')
    try:
        if args.mode == 'train':
            train(config, args, task, kind=kind, practice=practice)
        else:
            previous_argv = sys.argv
            try:
                sys.argv = evaluation_argv(paths['rollout_script'], task, config, args, kind=kind)
                runpy.run_path(str(paths['rollout_script']), run_name='__main__')
            finally:
                sys.argv = previous_argv
    finally:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != visible:
            raise RuntimeError('GPU visibility unexpectedly changed')


if __name__ == '__main__':
    main()
