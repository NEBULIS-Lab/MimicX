"""Budget-matched motion study: immutable preparation and native PPO pilot execution."""
import argparse
import ast
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

from mimicx.policy_recovery.resume import atomic_save, checkpoint_plan, digest, load_checkpoint, save_json
from mimicx.policy_recovery.worker import ensure_training_contract

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / 'scripts/experiments/motion_study.py'
METHODS = ('beyond_mimic', 'static_search', 'mimicx')
BLOCKED = 'Backend training not implemented'
V4_STRATEGY = 'joint_neutral_v4'


@contextmanager
def duplicate_guard(folder):
    """Only protects a single experiment; never represents device ownership."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def read(path):
    return json.loads(Path(path).read_text())


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Immutable record changed: {path}')
    else:
        save_json(path, value)


def resolve(value, base):
    return str((Path(base) / Path(value).expanduser()).resolve())


def source_identity(runtime):
    backend = Path(runtime['backend_root'])
    files = list((backend / 'src').rglob('*.py')) + [backend / 'scripts/train.py', Path(__file__), ENTRY]
    files += list((ROOT / 'mimicx/policy_recovery').glob('*.py'))
    files += list((ROOT / 'mimicx/experiments').glob('motion_study_*.py'))
    files += [ROOT / 'mimicx/refinement/proposals.py', ROOT / 'scripts/autorefine/mine_failures.py',
              ROOT / 'scripts/rollout_mjlab_motion_smoke.py', ROOT / 'mimicx/parkour/environment.py',
              ROOT / 'mimicx/visualization/replay_package.py']
    if runtime.get('rollout_script'):
        files.append(Path(runtime['rollout_script']))
    if runtime.get('legacy_backend_asset_proof'):
        files.append(Path(runtime['legacy_backend_asset_proof']))
    if runtime.get('common_model_xml'):
        from .sonic_study import asset_files
        files += list(asset_files(runtime['common_model_xml']))
        files += [ROOT / 'mimicx/experiments/sonic_study.py', ROOT / 'mimicx/experiments/sonic_helpers.py']
    if not (backend / 'src').is_dir():
        raise FileNotFoundError(backend / 'src')
    return {str(p.resolve()): digest(p) for p in sorted(set(files))}


def validate_task(task, base):
    task = dict(task)
    for key in ('task_id', 'split', 'source', 'source_path', 'motion_path', 'frames', 'fps', 'duration', 'sha256'):
        if key not in task:
            raise ValueError(f'Manifest task missing {key}')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', task['task_id']):
        raise ValueError('Unsafe task_id')
    if task['split'] not in ('development', 'confirmation'):
        raise ValueError('split must be development or confirmation')
    quality = task.get('quality_status', task.get('quality', {}))
    if isinstance(quality, dict):
        quality = quality.get('status')
    if quality not in ('passed', 'pass', 'valid', 'approved', 'blocked'):
        raise ValueError(f'Input QA not passed: {task["task_id"]}')
    task['training_eligible'] = quality != 'blocked'
    if quality == 'blocked':
        for key in ('motion_path', 'source_path'):
            if task[key]:
                task[key] = resolve(task[key], base)
        if task['motion_path'] and task['sha256'] and digest(task['motion_path']) != task['sha256']:
            raise ValueError('Blocked motion sha256 mismatch')
        return task
    if isinstance(task['frames'], bool) or not isinstance(task['frames'], int) or task['frames'] < 2:
        raise ValueError('frames must be an integer >= 2')
    for key in ('fps', 'duration'):
        if not math.isfinite(float(task[key])) or float(task[key]) <= 0:
            raise ValueError(f'Invalid {key}')
    # Both sample-count and endpoint-span conventions are accepted within one frame.
    if abs(float(task['duration']) - task['frames'] / float(task['fps'])) > 1.01 / float(task['fps']):
        raise ValueError('Duration disagrees with frames/fps')
    if float(task['fps']) != 50.:
        raise ValueError('Current scorer/worker contract requires 50 Hz references')
    task['motion_path'] = resolve(task['motion_path'], base)
    if task['source_path']:
        task['source_path'] = resolve(task['source_path'], base)
    if digest(task['motion_path']) != task['sha256']:
        raise ValueError('Motion sha256 mismatch')
    return task


def make_jobs(tasks, runtime):
    jobs = []
    for task in tasks:
        seeds = runtime.get('seeds', {}).get(task['split'], [101, 202] if task['split'] == 'development' else [101, 202, 303])
        if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError('Seeds must be unique nonnegative integers')
        for backend in runtime.get('backends', ['mjlab']):
            if backend not in ('mjlab', 'isaaclab'):
                raise ValueError('Unknown backend')
            for seed in seeds:
                prefix = f'{task["task_id"]}/{backend}/seed{seed}'
                shared = prefix + '/warmstart'
                eligible = task['training_eligible']
                jobs.append(dict(id=shared, task_id=task['task_id'], backend=backend, seed=seed,
                                 method='warmstart', dependencies=[], updates=3000, executable=backend == 'mjlab' and eligible,
                                 blocked_reason=None if backend == 'mjlab' and eligible else 'Input QA blocked or backend unsupported'))
                for method in METHODS:
                    static = method == 'static_search'
                    stages = [dict(name=f'branch{i}' if static else f'stage{i}', updates=1500,
                                   parent='warmstart' if static or i == 0 else 'protected_incumbent',
                                   selection_updates=[500, 1000, 1500]) for i in range(2)]
                    if method == 'beyond_mimic':
                        stages = [dict(name='continuous', updates=3000, parent='warmstart',
                                       selection_updates=list(range(500, 3001, 500)))]
                    if static:
                        from .motion_study_loop import static_patches
                        for stage, patch in zip(stages, static_patches(task)):
                            stage['patch'] = patch
                    jobs.append(dict(id=prefix + '/' + method, task_id=task['task_id'], backend=backend,
                                     seed=seed, method=method, dependencies=[shared], updates=3000,
                                     stages=stages, selection_slots=6, executable=backend == 'mjlab' and eligible,
                                     blocked_reason=None if backend == 'mjlab' and eligible else (
                                         BLOCKED if eligible else 'Input QA blocked')))
    return jobs


def prepare(manifest, runtime_path, output):
    manifest, runtime_path, output = map(lambda p: Path(p).expanduser().resolve(), (manifest, runtime_path, output))
    runtime = read(runtime_path)
    if runtime.get('refinement_strategy', 'legacy_v3') not in ('legacy_v3', V4_STRATEGY):
        raise ValueError('Unknown refinement strategy')
    for key in ('backend_root', 'native_root', 'rollout_script', 'common_model_xml', 'legacy_backend_asset_proof'):
        if key in runtime:
            runtime[key] = resolve(runtime[key], runtime_path.parent)
    if 'python' in runtime:
        # Dereferencing a venv interpreter symlink selects the base environment.
        runtime['python'] = os.path.abspath(os.path.join(str(runtime_path.parent), os.path.expanduser(runtime['python'])))
    runtime.setdefault('python', sys.executable)
    runtime.setdefault('device', 'cuda:0')
    runtime.setdefault('num_envs', 4096)
    runtime.setdefault('steps_per_env', 24)
    from .motion_study_adoption import proof_config
    proof_config(runtime)
    if type(runtime['num_envs']) is not int or runtime['num_envs'] <= 0:
        raise ValueError('num_envs must be positive')
    if runtime['steps_per_env'] != 24:
        raise ValueError('Native baseline requires 24 rollout steps per environment')
    if runtime['device'] != 'cpu' and not re.fullmatch(r'cuda(?::[0-9]+)?', runtime['device']):
        raise ValueError('Use cpu or a standard CUDA device')
    from .motion_study_loop import validation_protocol
    runtime['validation_protocol'] = validation_protocol(runtime.get('validation_protocol'))
    data = read(manifest)
    from .motion_study_scoring import common_protocol
    protocol = data.get('protocol', common_protocol())
    if any(protocol.get(key) != value for key, value in common_protocol().items()):
        raise ValueError('Common physical protocol must match the prospectively frozen all-method contract')
    protocol_sha256 = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if data.get('protocol_sha256', protocol_sha256) != protocol_sha256:
        raise ValueError('Dataset common protocol hash mismatch')
    if runtime.get('common_protocol', protocol) != protocol:
        raise ValueError('Runtime common protocol differs from the dataset frozen protocol object')
    runtime['common_protocol'] = protocol
    reference_contract = None
    canonical = data.get('canonical_g1')
    if canonical:
        reference_contract = {key: canonical[key] for key in ('joint_names', 'body_names', 'xml_sha256')}
        runtime.setdefault('common_model_xml', str(Path(runtime['backend_root']) / 'src/assets/robots/unitree_g1/xmls/g1.xml'))
        if digest(runtime['common_model_xml']) != canonical['xml_sha256']:
            raise ValueError('Common canonical model differs from dataset input QA asset')
    tasks = [validate_task(t, manifest.parent) for t in data['tasks']]
    for task in tasks:
        if task['training_eligible'] and task.get('requires_ground_contact_protocol'):
            if task.get('protocol_sha256') != protocol_sha256 or task.get('protocol_id') != protocol.get('protocol_id'):
                raise ValueError('Ground-contact task lacks the frozen dataset common-protocol identity')
    if not tasks or len({t['task_id'] for t in tasks}) != len(tasks):
        raise ValueError('Tasks must be nonempty and unique')
    if runtime.get('refinement_strategy') == V4_STRATEGY:
        if reference_contract is None:
            raise ValueError('v4 requires canonical FK capture')
        if any(t['split'] != 'development' for t in tasks):
            raise ValueError('v4 is development-only; do not enable confirmation tasks')
    campaign = dict(schema_version=2, scope='mjlab three-arm closed-loop study; Isaac execution unsupported',
                    tasks=tasks, runtime=runtime, manifest_sha256=digest(manifest), reference_contract=reference_contract,
                    common_protocol_sha256=protocol_sha256,
                    sources=source_identity(runtime), budget=dict(shared_warmstart=3000, per_arm=3000,
                    selection_slots=6, selection_interval=500), jobs=make_jobs(tasks, runtime))
    with duplicate_guard(output):
        immutable_json(output / 'campaign.json', campaign)
        for job in campaign['jobs']:
            immutable_json(output / 'jobs' / job['id'] / 'job.json', job)
    return campaign


def verify_campaign(campaign):
    for path, expected in campaign['sources'].items():
        if digest(path) != expected:
            raise RuntimeError(f'Frozen source changed: {path}')
    for task in campaign['tasks']:
        if task['motion_path'] and task['sha256'] and digest(task['motion_path']) != task['sha256']:
            raise RuntimeError(f'Frozen motion changed: {task["task_id"]}')


def check_execution_environment(device):
    for key in ('MIMICX_AUTOREFINE_PATCH_FILE', 'MIMICX_AUTOREFINE_PATCH_JSON', 'MIMICX_DEVICE_ID', 'MIMICX_TRAIN_DEVICE_IDS'):
        if os.environ.get(key):
            raise ValueError(f'Unset ambient override {key}')


def command_for(campaign_path, job):
    runtime = read(campaign_path)['runtime']
    return [runtime['python'], str(ENTRY), '_worker', '--campaign', str(campaign_path), '--job', job['id']]


def verified_receipt(folder):
    receipt = read(folder / 'done.json')
    for name, expected in receipt['artifacts'].items():
        if digest(folder / name) != expected:
            raise RuntimeError(f'Corrupt completed artifact: {folder / name}')
    return receipt


def run_job(campaign_path, job_id):
    campaign_path = Path(campaign_path).resolve()
    campaign = read(campaign_path)
    verify_campaign(campaign)
    job = next((j for j in campaign['jobs'] if j['id'] == job_id), None)
    if job is None:
        raise ValueError('Unknown job')
    if not job['executable']:
        raise NotImplementedError(job.get('blocked_reason') or 'Backend training not implemented')
    # Serialize all arms of this task/seed around the shared dependency.
    family = campaign_path.parent / 'jobs' / Path(job_id).parent
    with duplicate_guard(family):
        for dependency in job['dependencies']:
            _execute(campaign_path, campaign, next(j for j in campaign['jobs'] if j['id'] == dependency))
        return _execute(campaign_path, campaign, job)


def adopt_warmstart(campaign_path, job_id, donor):
    """Import verified pilot training into a newer frozen coordinator version."""
    import shutil
    campaign_path, donor = Path(campaign_path).resolve(), Path(donor).resolve()
    campaign = read(campaign_path)
    verify_campaign(campaign)
    job = next(j for j in campaign['jobs'] if j['id'] == job_id)
    if job['method'] != 'warmstart' or not job['executable']:
        raise ValueError('Only eligible warmstarts can be adopted')
    receipt = verified_receipt(donor)
    original = read(donor / 'effective_training.json')
    source_campaign, source_job = original['campaign'], original['job']
    for key in ('task_id', 'seed', 'backend', 'method', 'updates'):
        if source_job[key] != job[key]:
            raise ValueError(f'Donor warmstart {key} mismatch')
    task = next(t for t in campaign['tasks'] if t['task_id'] == job['task_id'])
    source_task = next(t for t in source_campaign['tasks'] if t['task_id'] == job['task_id'])
    for key in ('sha256', 'frames', 'fps'):
        if source_task[key] != task[key]:
            raise ValueError(f'Donor reference {key} mismatch')
    if not source_task['training_eligible']:
        raise ValueError('Donor input QA was not passed')
    for key in ('num_envs', 'steps_per_env'):
        if source_campaign['runtime'][key] != campaign['runtime'][key]:
            raise ValueError('Donor interaction budget mismatch')
    from .motion_study_adoption import check_backend_compatibility
    backend_audit = check_backend_compatibility(source_campaign, campaign)
    if original['start_iteration'] != 0 or original['parent_sha256'] is not None or original['additional_updates'] != 3000:
        raise ValueError('Donor was not scratch warmstart training')
    refresh_audit = None
    if original.get('initial_reference_refresh') is not True:
        # Early aligned pilot snapshots predate the explicit contract field.
        records = [(Path(p), sha) for p, sha in source_campaign['sources'].items()
                   if Path(p).parts[-3:] == ('mimicx', 'experiments', 'motion_study.py')]
        if len(records) != 1 or digest(records[0][0]) != records[0][1]:
            raise ValueError('Legacy donor lacks a verifiable frozen training source')
        tree = ast.parse(records[0][0].read_text())
        native = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train_native')
        calls = {node.func.id for node in ast.walk(native) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        if not {'install_initial_reference_refresh', 'aligned_env'}.issubset(calls):
            raise ValueError('Donor predates the required initial-reference cache refresh')
        refresh_audit = dict(source=str(records[0][0]), sha256=records[0][1], initial_reference_refresh=True)
    _, agent = native_configs(campaign['runtime'], task, job['seed'])
    if original['agent'] != json.loads(json.dumps(asdict(agent), default=str)):
        raise ValueError('Donor native PPO configuration changed')
    if receipt['training']['completed_updates'] != 3000 or load_checkpoint(donor / 'final.pt')['iter'] != 2999:
        raise ValueError('Donor warmstart did not complete exactly 3000 scratch updates')
    folder = campaign_path.parent / 'jobs' / job_id
    with duplicate_guard(folder.parent):
        if (folder / 'done.json').exists() or any(folder.glob('*.pt')):
            raise FileExistsError('Warmstart destination already has training artifacts')
        signature = dict(campaign_sha256=digest(campaign_path), job=job)
        immutable_json(folder / 'contract.json', signature)
        for name in ('final.pt', 'effective_training.json', 'training.json'):
            atomic_save(lambda path, infos=None, source=donor / name: shutil.copy2(source, path), folder / name)
        save_json(folder / 'adoption.json', dict(donor=str(donor), donor_receipt=receipt,
            donor_effective_sha256=digest(donor / 'effective_training.json'),
            legacy_refresh_source_audit=refresh_audit,
            backend_identity_audit=backend_audit,
            coordinator_only_migration=True, training_cost_reused_not_zero=True))
        names = ('final.pt', 'effective_training.json', 'training.json', 'adoption.json')
        save_json(folder / 'done.json', dict(signature=signature, status='warmstart_adopted',
            artifacts={name: digest(folder / name) for name in names}, training=read(folder / 'training.json')))
    return read(folder / 'done.json')


def _execute(campaign_path, campaign, job):
    folder = campaign_path.parent / 'jobs' / job['id']
    signature = dict(campaign_sha256=digest(campaign_path), job=job)
    immutable_json(folder / 'contract.json', signature)
    if (folder / 'done.json').exists():
        receipt = verified_receipt(folder)
        if receipt['signature'] != signature:
            raise RuntimeError('Completed job contract mismatch')
        return receipt
    runtime = campaign['runtime']
    check_execution_environment(runtime['device'])
    cmd = command_for(campaign_path, job)
    started, before = time.time(), resource.getrusage(resource.RUSAGE_CHILDREN)
    attempt_path = folder / 'attempts' / f'{time.time_ns()}.json'
    attempt = dict(command=cmd, cwd=runtime['backend_root'], started=started, status='running',
                   pid=os.getpid(), tmux=os.environ.get('TMUX'), gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   log=str(folder / 'run.log'), output=str(folder), job_id=job['id'])
    save_json(attempt_path, attempt)
    try:
        with (folder / 'run.log').open('a') as log:
            result = subprocess.run(cmd, cwd=runtime['backend_root'], stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'PYTHONUNBUFFERED': '1', 'WANDB_MODE': 'disabled',
                                         'GIT_PYTHON_REFRESH': os.environ.get('GIT_PYTHON_REFRESH', 'quiet')}, check=False)
        attempt['returncode'] = result.returncode
        if result.returncode:
            raise RuntimeError(f'Worker failed ({result.returncode}); see {folder / "run.log"}')
        summary = read(folder / 'training.json')
        if summary['completed_updates'] != job['updates']:
            raise RuntimeError('Incomplete training receipt')
        names = ['final.pt', 'effective_training.json', 'training.json']
        names += summary['selection_checkpoints']
        names += summary.get('artifacts', [])
        receipt = dict(signature=signature, status=summary['status'],
                       artifacts={name: digest(folder / name) for name in names}, training=summary)
        save_json(folder / 'done.json', receipt)
        attempt['status'] = 'completed'
        return receipt
    except BaseException as exc:
        attempt.update(status='failed', error=repr(exc))
        raise
    finally:
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        wall = time.time() - started
        attempt.update(finished=time.time(), wall_seconds=wall,
                       child_cpu_seconds=after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
                       assigned_gpu_wall_seconds=wall if runtime['device'] != 'cpu' else 0,
                       measured_gpu_compute_seconds=None, measured_gpu_energy_joules=None)
        save_json(attempt_path, attempt)


def native_configs(runtime, task, seed):
    """Use the real backend TrainConfig parser/default factory, not recovery defaults."""
    backend = Path(runtime['backend_root'])
    sys.path[:0] = [str(backend), str(backend / 'src')]
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    import src.tasks.tracking.config.g1  # noqa: F401
    spec = importlib.util.spec_from_file_location('motion_study_native_train', backend / 'scripts/train.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    config = module.TrainConfig.from_task('Unitree-G1-Tracking')
    cfg, agent = config.env, config.agent
    cfg.commands['motion'].motion_file = task['motion_path']
    if getattr(cfg.commands['motion'], 'motion_files', ()):
        raise ValueError('Native baseline unexpectedly contains embedded multiple motions')
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = runtime['num_envs']
    agent.logger, agent.upload_model = 'tensorboard', False
    # The wrapper filters saves to exact 500-update boundaries.
    agent.save_interval = 1
    if agent.num_steps_per_env != runtime['steps_per_env']:
        raise ValueError('Native transition budget mismatch')
    return cfg, agent


def train_native(campaign, job, folder, *, stage=None):
    """Scratch/resume with native objectives and PPO; shared atomic worker helpers."""
    runtime = campaign['runtime']
    task = next(t for t in campaign['tasks'] if t['task_id'] == job['task_id'])
    cfg, agent = native_configs(runtime, task, job['seed'])
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends

    parent = Path(stage['parent']) if stage else (folder.parent / 'warmstart/final.pt' if job['dependencies'] else None)
    if stage:
        from .motion_study_loop import apply_patch
        if digest(parent) != stage['parent_sha256']:
            raise ValueError('Stage parent changed')
        apply_patch(cfg, stage['patch'], task)
    start = int(load_checkpoint(parent)['iter']) + 1 if parent else 0
    budget = stage['updates'] if stage else job['updates']
    contract = dict(campaign=campaign, job=job, start_iteration=start,
                    parent_sha256=digest(parent) if parent else None, additional_updates=budget,
                    initial_reference_refresh=True,
                    additional_transitions=budget * runtime['num_envs'] * agent.num_steps_per_env,
                    env=asdict(cfg), agent=asdict(agent), stage=stage)
    # Native configs contain callables and tuples; YAML preserves the full config,
    # while the resume contract stores deterministic scalar PPO settings.
    contract.pop('env')
    contract['agent'] = json.loads(json.dumps(contract['agent'], default=str))
    ensure_training_contract(folder, contract)
    plan = checkpoint_plan(folder, start, budget)
    slots = [f'model_{start + n - 1}.pt' for n in range(500, budget + 1, 500)]
    if plan['remaining_updates']:
        from mimicx.parkour.environment import install_initial_reference_refresh
        original_env, aligned_env = install_initial_reference_refresh()
        env = None
        try:
            configure_torch_backends()
            env = RslRlVecEnvWrapper(aligned_env(cfg, device=runtime['device']), clip_actions=agent.clip_actions)
            runner = MjlabOnPolicyRunner(env, asdict(agent), str(folder), device=runtime['device'])
            checkpoint = plan['checkpoint'] or (str(parent) if parent else None)
            if checkpoint:
                runner.load(checkpoint, map_location=runtime['device'], load_cfg={
                    'actor': True, 'critic': True, 'optimizer': True, 'iteration': True})
                saved = load_checkpoint(checkpoint)
                info = saved.get('infos') or {}
                if 'motion_study_learning_rate' not in info:
                    raise ValueError('Resume checkpoint lacks adaptive learning-rate state')
                runner.alg.learning_rate = info['motion_study_learning_rate']
            runner.current_learning_iteration = plan['next_iteration']
            original_save = runner.save

            def save_slot(path, infos=None):
                iteration = runner.current_learning_iteration
                if (iteration - start + 1) % 500:
                    return
                atomic_save(original_save, folder / f'model_{iteration}.pt',
                            {'motion_study_learning_rate': runner.alg.learning_rate})

            runner.save = save_slot
            dump_yaml(folder / 'params/env.yaml', asdict(cfg))
            dump_yaml(folder / 'params/agent.yaml', asdict(agent))
            runner.learn(plan['remaining_updates'], init_at_random_ep_len=True)
            if runner.current_learning_iteration != start + budget - 1:
                raise RuntimeError('Native runner update counter mismatch')
            save_slot(None)
        finally:
            if env is not None:
                env.close()
            sys.modules['mjlab.envs'].ManagerBasedRlEnv = original_env
    final = folder / slots[-1]
    if checkpoint_plan(folder, start, budget)['remaining_updates']:
        raise RuntimeError('Training did not produce complete checkpoints')
    for name in slots:
        state = load_checkpoint(folder / name)
        if state['iter'] != int(Path(name).stem.split('_')[-1]):
            raise RuntimeError('Selection slot counter mismatch')
    import shutil
    atomic_save(lambda path, infos=None: shutil.copy2(final, path), folder / 'final.pt')
    summary = dict(completed_updates=budget, transitions=contract['additional_transitions'],
                   executed_updates_this_attempt=plan['remaining_updates'], resume=plan,
                   worker_pid=os.getpid(), worker_parent_pid=os.getppid(),
                   selection_checkpoints=slots if job['method'] != 'warmstart' else [],
                   status='training_complete_selection_pending', bitwise_resume=False,
                   restart_note='Physics/RNG episodes restart; work lost after last durable slot may be repeated',
                   attempted_transitions_exact=None)
    save_json(folder / 'training.json', summary)
    return summary


def refinement_v4(campaign, task, repeats, folder, tried):
    """Keep the native objective when every selection repeat already completes.

    Full-motion replay retains two distinct sampling candidates without treating
    a precision maximum as a failure or inventing body-specific reward evidence.
    """
    from . import motion_study_loop as loop
    from mimicx.policy_recovery.metrics import summarize
    from mimicx.refinement.proposals import ProposalContext, validate_patch
    n = loop.horizon(task)
    if not all(summarize(r, n)['complete'] for r in repeats):
        return loop.untried_patch(loop.diagnose(campaign, task, repeats, folder), tried)
    context = ProposalContext(task['task_id'], n, 0, None, (0, n), (), (), (), ())
    signature = lambda p: {k: v for k, v in p.items() if k != 'id'}
    for ratio in (.45, .65):
        patch = dict(id=f'native_full_replay_{int(ratio * 100)}',
                     sampling=dict(mode='window', start=0, end=n, ratio=ratio),
                     reward_overrides={}, body_pos_rewards=[], body_z_rewards=[],
                     body_lin_vel_rewards=[], body_ang_vel_rewards=[])
        if all(signature(patch) != signature(old) for old in tried):
            return validate_patch(patch, context=context)
    raise ValueError('No untried native-objective replay remains')


def accept_v4(old_rows, new_rows, horizon, old_evidence, new_evidence):
    """Matched-seed Pareto guard on same-clock, pre-reset local/global errors.

    No task-specific tolerance: both mean and p95 must be nonincreasing for
    every seed. The existing 2% precision-improvement rule is applied to the
    sum of the two metre-valued errors, not a mixed metre/radian proxy score.
    """
    import numpy as np
    from .motion_study_gate import accept
    result = accept(old_rows, new_rows, horizon)
    if len(old_evidence) != len(old_rows) or len(new_evidence) != len(new_rows):
        raise ValueError('Joint gate requires matched canonical evidence')
    keys = ('world_body_error_m', 'root_local_fk_body_error_m')
    for evidence, summaries in ((old_evidence, result['old']), (new_evidence, result['new'])):
        for item, summary in zip(evidence, summaries):
            frontier = item.get('pre_failure_frames')
            if type(frontier) is not int or not 0 <= frontier <= horizon + 1:
                raise ValueError('Invalid common physical frontier')
            valid = min(summary['valid_steps'] + 1, frontier)
            for key in keys:
                values = np.asarray(item.get(key), dtype=float)
                if (values.shape != (horizon + 1,) or not np.isfinite(values[1:valid]).all()
                        or (values[1:valid] < 0).any()):
                    raise ValueError('Invalid same-clock canonical tracking evidence')
    checks, scores_old, scores_new = [], [], []
    joint_guard = True
    for index, (a, b, sa, sb) in enumerate(zip(old_evidence, new_evidence, result['old'], result['new'])):
        frontier_ok = b['pre_failure_frames'] >= a['pre_failure_frames']
        n = min(sa['valid_steps'], sb['valid_steps'],
                max(0, a['pre_failure_frames'] - 1), max(0, b['pre_failure_frames'] - 1))
        measured = {}
        for key in keys:
            if n:
                va, vb = np.asarray(a[key])[1:n + 1], np.asarray(b[key])[1:n + 1]
                before = [float(va.mean()), float(np.quantile(va, .95))]
                after = [float(vb.mean()), float(np.quantile(vb, .95))]
                joint_guard &= all(y <= x + 1e-12 for x, y in zip(before, after))
                scores_old.append(before[0])
                scores_new.append(after[0])
                measured[key] = dict(old=before, new=after)
        joint_guard &= frontier_ok
        checks.append(dict(seed_index=index, common_valid_steps=n, frontier_guard=frontier_ok, measured=measured))
    frontier_improved = any(b['valid_steps'] > a['valid_steps'] for a, b in zip(result['old'], result['new']))
    frontier_improved |= any(b['pre_failure_frames'] > a['pre_failure_frames']
                             for a, b in zip(old_evidence, new_evidence))
    precision_improved = bool(scores_old) and sum(scores_new) < .98 * sum(scores_old)
    native_safe = result['accepted'] or result['reason'] == 'No execution improvement'
    accepted = native_safe and joint_guard and (frontier_improved or precision_improved)
    reason = (result['reason'] if not native_safe else 'Joint local/global guard' if not joint_guard
              else 'Accepted' if accepted else 'No joint precision improvement')
    # Legacy prefix tuples must round-trip before immutable resume comparison.
    return json.loads(json.dumps({**result, 'accepted': accepted, 'reason': reason, 'gate_version': V4_STRATEGY,
                                  'joint_guard': joint_guard, 'joint_checks': checks, 'uses_holdout': False}))


def selection_evidence_v4(folder, seeds):
    import numpy as np
    common = read(folder / 'summary.json')['common_metrics']
    if len(common) != len(seeds):
        raise ValueError('Missing canonical selection metrics')
    evidence = []
    for seed, metrics in zip(seeds, common):
        with np.load(folder / f'seed{seed}/rollout/common_steps.npz', allow_pickle=False) as data:
            evidence.append({key: data[key].copy() for key in
                             ('world_body_error_m', 'root_local_fk_body_error_m')})
        evidence[-1]['pre_failure_frames'] = metrics['pre_failure_frames']
    return evidence


def execute_arm_v4(campaign_path, campaign, job, folder):
    """Opt-in development strategy; freeze selection before independent scoring."""
    import shutil
    from . import motion_study_loop as loop
    from mimicx.policy_recovery.metrics import qualifies, summarize
    task = next(t for t in campaign['tasks'] if t['task_id'] == job['task_id'])
    if task['split'] != 'development' or not campaign.get('reference_contract'):
        raise ValueError('v4 requires development tasks and canonical capture')
    n = loop.horizon(task)
    protocol = campaign['runtime']['validation_protocol']
    seeds, noise = protocol['selection_seeds'], protocol['selection_noise']
    warm_job = next(j for j in campaign['jobs'] if j['id'] == job['dependencies'][0])
    warm = folder.parent / 'warmstart/final.pt'
    warm_folder = folder.parent / 'shared_validation/warmstart'
    warm_rows = loop.evaluate(campaign_path, campaign, warm_job, warm, warm_folder, seeds, noise)
    warm_evidence = selection_evidence_v4(warm_folder, seeds)
    incumbent, incumbent_rows, incumbent_evidence = warm, warm_rows, warm_evidence
    incumbent_folder = warm_folder
    history, slots, stages, tried = [], [], [], []
    for index, declaration in enumerate(job['stages']):
        current = folder if job['method'] == 'beyond_mimic' else folder / declaration['name']
        spec = None
        if job['method'] != 'beyond_mimic':
            parent = warm if job['method'] == 'static_search' else incumbent
            path = current / 'stage.json'
            if path.exists():
                spec = read(path)
                if spec['parent'] != str(parent) or spec['parent_sha256'] != digest(parent):
                    raise ValueError('Stage protected-parent contract changed')
            else:
                patch = declaration.get('patch')
                if job['method'] == 'mimicx':
                    patch = refinement_v4(campaign, task, incumbent_rows, current / 'diagnosis', tried)
                spec = dict(updates=declaration['updates'], parent=str(parent),
                            parent_sha256=digest(parent), patch=patch)
                immutable_json(path, spec)
            tried.append(spec['patch'])
            immutable_json(current / 'proposal_history.json', dict(tried=tried[:-1], selected=spec['patch'],
                rule='Native-objective full-motion replay when selection completion saturates; existing diagnosis otherwise'))
        training = train_native(campaign, job, current, stage=spec)
        stages.append(dict(folder=str(current.relative_to(folder)), updates=training['completed_updates'],
                           transitions=training['transitions']))
        for name in training['selection_checkpoints']:
            checkpoint = current / name
            slot = len(slots)
            output = folder / 'validation' / f'slot{slot:02d}'
            candidate_rows = loop.evaluate(campaign_path, campaign, job, checkpoint, output, seeds, noise)
            candidate_evidence = selection_evidence_v4(output, seeds)
            gate = accept_v4(incumbent_rows, candidate_rows, n, incumbent_evidence, candidate_evidence)
            warm_gate = accept_v4(warm_rows, candidate_rows, n, warm_evidence, candidate_evidence)
            if gate['accepted'] and not warm_gate['joint_guard']:
                gate.update(accepted=False, reason='Warmstart joint local/global guard')
            gate.update(slot=slot, stage=index, candidate=str(checkpoint), candidate_sha256=digest(checkpoint),
                        incumbent=str(incumbent), incumbent_sha256=digest(incumbent),
                        warmstart_joint_guard=warm_gate['joint_guard'])
            immutable_json(folder / 'gates' / f'slot{slot:02d}.json', gate)
            history.append(gate)
            slots.append(str(checkpoint.relative_to(folder)))
            if gate['accepted']:
                incumbent, incumbent_rows, incumbent_evidence = checkpoint, candidate_rows, candidate_evidence
                incumbent_folder = output
    if len(slots) != 6 or sum(s['updates'] for s in stages) != 3000:
        raise RuntimeError('Arm must consume exactly 3000 updates and six selection slots')
    selection = dict(checkpoint=str(incumbent), sha256=digest(incumbent), history=history,
                     seeds=seeds, criterion=V4_STRATEGY, slots=6, uses_holdout=False)
    immutable_json(folder / 'selection.json', selection)
    held, common_held = {}, {}
    for kind in ('nominal', 'perturbed'):
        output = folder / 'holdout' / kind
        rows = loop.evaluate(campaign_path, campaign, job, incumbent, output,
                             protocol['holdout'][task['split']][kind], protocol[kind + '_noise'])
        held[kind] = [summarize(r, n) for r in rows]
        common_held[kind] = read(output / 'summary.json')['common_metrics']
    common_selection = read(incumbent_folder / 'summary.json')['common_metrics']
    native_qualified = qualifies([summarize(r, n) for r in incumbent_rows]) and all(qualifies(v) for v in held.values())
    common_qualified = len(common_selection) == len(seeds) and all(m['complete_motion'] for m in common_selection)
    common_qualified = common_qualified and all(
        len(common_held[k]) == len(protocol['holdout'][task['split']][k])
        and all(m['complete_motion'] for m in common_held[k]) for k in held)
    transitions_per_update = campaign['runtime']['num_envs'] * campaign['runtime']['steps_per_env']
    shared_updates = warm_job['updates'] + sum(j['updates'] for j in campaign['jobs']
                                              if j['dependencies'] == [warm_job['id']])
    budget = dict(shared_warmstart_updates=warm_job['updates'], continuation_updates=job['updates'],
                  arm_updates_including_reused_warmstart=warm_job['updates'] + job['updates'],
                  shared_three_arm_updates=shared_updates, shared_three_arm_transitions=shared_updates * transitions_per_update,
                  reused_warmstart_training_cost_not_zero=True, candidate_selection_rollouts=6 * len(seeds),
                  shared_warmstart_selection_rollouts=len(seeds),
                  independent_holdout_rollouts=sum(len(protocol['holdout'][task['split']][k]) for k in held))
    immutable_json(folder / 'report.json', dict(status='completed', holdout=held,
        selected_checkpoint=str(incumbent), selected_sha256=digest(incumbent), selected_warmstart=incumbent == warm,
        selection=[summarize(r, n) for r in incumbent_rows], qualified=native_qualified and common_qualified,
        native_qualified=native_qualified, common_physical_qualified=common_qualified,
        common_selection_metrics=common_selection, common_physical_metrics=common_held,
        common_protocol=campaign['runtime']['common_protocol'], common_protocol_sha256=campaign['common_protocol_sha256'],
        completed_updates=3000, transitions=sum(s['transitions'] for s in stages), stages=stages,
        metrics_contract=V4_STRATEGY, budget_accounting=budget))
    atomic_save(lambda path, infos=None: shutil.copy2(incumbent, path), folder / 'final.pt')
    if job['method'] != 'beyond_mimic':
        immutable_json(folder / 'effective_training.json', dict(campaign=campaign, job=job, stages=stages))
    files = [p for p in folder.rglob('*') if p.is_file() and p != folder / 'training.json' and p.name not in
             ('.run.lock', 'done.json', 'run.log', 'contract.json')
             and 'attempts' not in p.parts and not any(part.startswith('incomplete_rollout_') for part in p.parts)]
    summary = dict(status='completed', completed_updates=3000, transitions=sum(s['transitions'] for s in stages),
                   selection_checkpoints=slots, artifacts=[str(p.relative_to(folder)) for p in files],
                   stages=stages, selected_checkpoint=str(incumbent), selected_sha256=digest(incumbent),
                   qualified=native_qualified and common_qualified, budget_accounting=budget, attempted_transitions_exact=None)
    save_json(folder / 'training.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action', required=True)
    p = subs.add_parser('prepare')
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--runtime', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p = subs.add_parser('prove-backend-assets')
    p.add_argument('--backend-root', required=True, type=Path)
    p.add_argument('--model-xml', required=True, type=Path)
    p.add_argument('--git-commit', required=True)
    p.add_argument('--output', required=True, type=Path)
    p = subs.add_parser('check-adoption')
    p.add_argument('--campaign', required=True, type=Path)
    p.add_argument('--donor-campaign', required=True, type=Path)
    p.add_argument('--output', type=Path)
    for action in ('jobs', 'run', 'resume', 'adopt-warmstart', 'validate-checkpoint', '_worker', '_evaluate'):
        p = subs.add_parser(action)
        p.add_argument('--campaign', required=True, type=Path)
        if action in ('_worker', '_evaluate'):
            p.add_argument('--job', required=True)
            if action == '_evaluate':
                p.add_argument('--checkpoint', type=Path, required=True)
                p.add_argument('--output', type=Path, required=True)
                p.add_argument('--seed', type=int, required=True)
                p.add_argument('--noise', type=float, required=True)
        elif action != 'jobs':
            p.add_argument('--task', required=True)
            p.add_argument('--seed', type=int, required=True)
            p.add_argument('--backend', choices=('mjlab', 'isaaclab'), default='mjlab')
            p.add_argument('--method', choices=('warmstart',) + METHODS, required=True)
            if action == 'adopt-warmstart':
                p.add_argument('--from-job', type=Path, required=True)
            if action == 'validate-checkpoint':
                p.add_argument('--checkpoint', type=Path, required=True)
                p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == 'prove-backend-assets':
        from .motion_study_adoption import build_asset_proof
        print(json.dumps(build_asset_proof(args.backend_root, args.model_xml, args.git_commit, args.output)))
    elif args.action == 'check-adoption':
        from .motion_study_adoption import check_backend_compatibility
        campaign, donor = read(args.campaign), read(args.donor_campaign)
        verify_campaign(campaign)
        verify_campaign(donor)
        audit = check_backend_compatibility(donor, campaign)
        if args.output:
            save_json(args.output, audit)
        print(json.dumps(audit))
    elif args.action == 'prepare':
        campaign = prepare(args.manifest, args.runtime, args.output)
        print(json.dumps(dict(status='prepared', jobs=len(campaign['jobs']), executed=0)))
    elif args.action == 'jobs':
        campaign = read(args.campaign)
        for job in campaign['jobs']:
            folder = args.campaign.resolve().parent / 'jobs' / job['id']
            status = verified_receipt(folder)['status'] if (folder / 'done.json').exists() else (
                'prepared' if job['executable'] else 'blocked')
            print(json.dumps({**job, 'status': status}))
    elif args.action in ('_worker', '_evaluate'):
        campaign = read(args.campaign)
        verify_campaign(campaign)
        check_execution_environment(campaign['runtime']['device'])
        job = next(j for j in campaign['jobs'] if j['id'] == args.job)
        if not job['executable']:
            raise NotImplementedError(BLOCKED)
        folder = args.campaign.resolve().parent / 'jobs' / job['id']
        if args.action == '_evaluate':
            from .motion_study_loop import evaluate_native
            evaluate_native(campaign, job, args)
            return
        with duplicate_guard(folder):
            print(json.dumps(dict(worker_pid=os.getpid(), parent_pid=os.getppid(),
                                  command=sys.argv, job=job['id'])), flush=True)
            if job['method'] == 'warmstart':
                train_native(campaign, job, folder)
            else:
                from .motion_study_loop import execute_arm
                executor = execute_arm_v4 if campaign['runtime'].get('refinement_strategy') == V4_STRATEGY else execute_arm
                executor(args.campaign.resolve(), campaign, job, folder)
    else:
        job_id = f'{args.task}/{args.backend}/seed{args.seed}/{args.method}'
        if args.action == 'adopt-warmstart':
            print(json.dumps(adopt_warmstart(args.campaign, job_id, args.from_job)))
        elif args.action == 'validate-checkpoint':
            from .motion_study_loop import validate_checkpoint
            print(json.dumps(validate_checkpoint(args.campaign, job_id, args.checkpoint, args.output)))
        else:
            print(json.dumps(run_job(args.campaign, job_id)))


if __name__ == '__main__':
    main()
