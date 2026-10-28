"""CPU contracts; fake simulator tests are not physical experiment evidence."""
from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from mimicx.experiments import motion_study as study
from mimicx.policy_recovery import resume


@pytest.fixture
def campaign(tmp_path):
    backend = tmp_path / 'backend'
    (backend / 'src').mkdir(parents=True)
    (backend / 'scripts').mkdir()
    (backend / 'scripts/train.py').write_text('# native fixture')
    (backend / 'src/task.py').write_text('# task fixture')
    motion = tmp_path / 'motion.npz'
    motion.write_bytes(b'read-only motion fixture')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'tasks': [dict(task_id='walk', split='development', source='BONES',
        source_path='original.csv', motion_path='motion.npz', frames=301, fps=50., duration=6.,
        sha256=study.digest(motion), quality_status='passed')]}))
    runtime = tmp_path / 'runtime.json'
    runtime.write_text(json.dumps(dict(backend_root='backend', python=sys.executable, device='cpu', num_envs=2)))
    output = tmp_path / 'campaign'
    study.prepare(manifest, runtime, output)
    return output / 'campaign.json'


def test_lightweight_cli_does_not_import_simulators():
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch','mjlab','mujoco','warp','rsl_rl'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from mimicx.experiments.motion_study import main
main(['--help'])
"""
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([sys.executable, str(study.ENTRY), 'run', '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert '--method' in result.stdout


def test_prepare_budgets_dependencies_and_no_implicit_execution(campaign):
    data = study.read(campaign)
    assert len(data['jobs']) == 8  # two seeds, one shared warmstart and three arms
    for job in data['jobs']:
        assert job['updates'] == 3000
        if job['method'] != 'warmstart':
            assert job['dependencies'] == [str(Path(job['id']).parent / 'warmstart')]
            assert sum(s['updates'] for s in job['stages']) == 3000
            assert sum(len(s['selection_updates']) for s in job['stages']) == 6
    static = next(j for j in data['jobs'] if j['method'] == 'static_search')
    assert all(s['parent'] == 'warmstart' for s in static['stages'])
    assert not list(campaign.parent.rglob('done.json'))
    assert not list(campaign.parent.rglob('*.pt'))


@pytest.mark.parametrize('relative', [False, True])
def test_runtime_python_preserves_virtual_environment_symlink(campaign, tmp_path, relative):
    executable = tmp_path / 'tracking-venv/bin/python'
    executable.parent.mkdir(parents=True)
    executable.symlink_to(Path(sys.executable).resolve())
    runtime_path = tmp_path / 'runtime.json'
    runtime = study.read(runtime_path)
    runtime['python'] = str(executable.relative_to(tmp_path)) if relative else str(executable)
    study.save_json(runtime_path, runtime)
    output = tmp_path / 'preserved-interpreter'
    prepared = study.prepare(tmp_path / 'manifest.json', runtime_path, output)
    assert prepared['runtime']['python'] == str(executable)
    assert prepared['runtime']['python'] != str(executable.resolve())
    assert study.command_for(output / 'campaign.json', prepared['jobs'][0])[0] == str(executable)


def test_prepare_rejects_changed_inputs_and_retains_blocked(tmp_path, campaign):
    manifest, runtime = tmp_path / 'manifest.json', tmp_path / 'runtime.json'
    study.prepare(manifest, runtime, campaign.parent)
    data = study.read(manifest)
    data['tasks'][0].update(quality_status='blocked', motion_path=None, sha256=None)
    manifest.write_text(json.dumps(data))
    blocked = study.prepare(manifest, runtime, tmp_path / 'blocked')
    assert len(blocked['jobs']) == 8
    assert not any(j['executable'] for j in blocked['jobs'])
    with pytest.raises(ValueError, match='Immutable'):
        study.prepare(manifest, runtime, campaign.parent)


def test_fail_closed_unsupported_backend_and_changed_reference(campaign):
    data = study.read(campaign)
    assert all(j['executable'] for j in data['jobs'])
    with pytest.raises(NotImplementedError):
        blocked = {**data['jobs'][0], 'id': 'walk/isaaclab/seed101/warmstart', 'executable': False}
        data['jobs'].append(blocked)
        study.save_json(campaign, data)
        study.run_job(campaign, blocked['id'])
    assert not list(campaign.parent.rglob('attempts/*.json'))
    Path(study.read(campaign)['tasks'][0]['motion_path']).write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='Frozen motion'):
        study.run_job(campaign, 'walk/mjlab/seed101/warmstart')


def test_run_duplicate_lock_and_gpu_contract(tmp_path, monkeypatch):
    with study.duplicate_guard(tmp_path):
        with pytest.raises(BlockingIOError):
            with study.duplicate_guard(tmp_path):
                pass
    for key in ('TMUX', 'CUDA_VISIBLE_DEVICES', 'MIMICX_DEVICE_ID', 'MIMICX_AUTOREFINE_PATCH_FILE', 'MIMICX_TRAIN_DEVICE_IDS'):
        monkeypatch.delenv(key, raising=False)
    study.check_execution_environment('cpu')
    study.check_execution_environment('cuda:0')
    monkeypatch.setenv('TMUX', 'caller-session')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-abcd-123')
    study.check_execution_environment('cuda:0')
    monkeypatch.setenv('MIMICX_AUTOREFINE_PATCH_FILE', 'ambient.json')
    with pytest.raises(ValueError, match='ambient'):
        study.check_execution_environment('cpu')


def fake_runtime(monkeypatch, *, interrupt=False, interrupt_at=499):
    from mimicx.experiments import motion_study_loop as loop
    original_run = subprocess.run
    @dataclass
    class Algorithm:
        schedule: str = 'adaptive'
        entropy_coef: float = .005

    @dataclass
    class Agent:
        algorithm: Algorithm = field(default_factory=Algorithm)
        num_steps_per_env: int = 24
        clip_actions: float = 1.

    @dataclass
    class Config:
        seed: int = 101

    calls, executed = [], []
    fail = [interrupt]

    class Env:
        def __init__(self, cfg, device):
            assert device == 'cpu'
            assert calls[-1] == 'initial_reference_refresh'
        def close(self):
            calls.append('close')

    class Runner:
        def __init__(self, env, cfg, folder, device):
            assert cfg['algorithm']['schedule'] == 'adaptive'
            assert cfg['algorithm']['entropy_coef'] == .005
            self.folder = Path(folder)
            self.alg = SimpleNamespace(learning_rate=.001)
        def load(self, path, **kwargs):
            assert kwargs['load_cfg']['optimizer'] is True
            calls.append(('load', str(path)))
        def save(self, path, infos=None):
            Path(path).write_text(json.dumps(dict(iter=self.current_learning_iteration, infos=infos,
                actor_state_dict={}, critic_state_dict={}, optimizer_state_dict={})))
        def learn(self, count, init_at_random_ep_len):
            assert init_at_random_ep_len
            start = self.current_learning_iteration
            for it in range(start, start + count):
                self.current_learning_iteration = it
                executed.append(it)
                self.alg.learning_rate = .0003
                self.save('unused')
                if it == interrupt_at and fail[0]:
                    fail[0] = False
                    raise RuntimeError('interruption')

    for name, attrs in {
        'mjlab.envs': {'ManagerBasedRlEnv': Env},
        'mjlab.rl': {'MjlabOnPolicyRunner': Runner, 'RslRlVecEnvWrapper': lambda env, **kw: env},
        'mjlab.utils.os': {'dump_yaml': lambda *args: None},
        'mjlab.utils.torch': {'configure_torch_backends': lambda: None},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(study, 'native_configs', lambda *args: (Config(), Agent()))
    def initial_reference_refresh():
        calls.append('initial_reference_refresh')
        return Env, Env
    monkeypatch.setattr('mimicx.parkour.environment.install_initial_reference_refresh', initial_reference_refresh)
    monkeypatch.setattr(loop, 'apply_patch', lambda cfg, patch, task: calls.append(('patch', patch['id'])))
    monkeypatch.setattr(study, 'load_checkpoint', study.read)
    monkeypatch.setattr(resume, 'load_checkpoint', study.read)

    def run(cmd, **kwargs):
        if 'mine_failures.py' in cmd[1]:
            return original_run(cmd, **kwargs)
        path = Path(cmd[cmd.index('--campaign') + 1])
        job_id = cmd[cmd.index('--job') + 1]
        data = study.read(path)
        job = next(j for j in data['jobs'] if j['id'] == job_id)
        calls.append(('job', job_id))
        if '_evaluate' in cmd:
            import csv
            import numpy as np
            from tests.test_policy_recovery_metrics import rows
            from mimicx.visualization.replay_package import write_replay_package
            output = Path(cmd[cmd.index('--output') + 1])
            output.mkdir(parents=True)
            n = loop.horizon(data['tasks'][0])
            error = .08
            for slot, value in enumerate((.06, .3, .05, .04, .03, .02)):
                if f'slot{slot:02d}' in str(output):
                    error = value
            values = rows(n, error=error)
            for row in values:
                row['body_pos_error_left_ankle_roll_link'] = error
            with (output / 'steps.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=values[0])
                writer.writeheader()
                writer.writerows(values)
            study.save_json(output / 'metrics.json', dict(steps=n, done_any_count=0))
            arrays = {'timestamps': np.arange(1, n + 1) / 50.}
            for prefix in ('robot', 'reference'):
                arrays[prefix + '_root_pose'] = np.zeros((n, 7))
                arrays[prefix + '_joint_positions'] = np.zeros((n, 29))
                arrays[prefix + '_body_pose'] = np.zeros((n, 1, 7))
            write_replay_package(output / 'replay', arrays=arrays, metadata={'fps': 50.},
                                 body_names=['pelvis'], joint_names=[f'j{i}' for i in range(29)])
        elif job['method'] == 'warmstart':
            study.train_native(data, job, path.parent / 'jobs' / job_id)
        else:
            loop.execute_arm(path, data, job, path.parent / 'jobs' / job_id)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(study.subprocess, 'run', run)
    return calls, executed


def test_scratch_full_dependency_resume_slots_and_costs(campaign, monkeypatch):
    calls, executed = fake_runtime(monkeypatch, interrupt=True)
    job_id = 'walk/mjlab/seed101/beyond_mimic'
    with pytest.raises(RuntimeError, match='interruption'):
        study.run_job(campaign, job_id)
    result = study.run_job(campaign, job_id)
    assert result['status'] == 'completed'
    assert executed == list(range(6000))
    folder = campaign.parent / 'jobs' / job_id
    summary = study.read(folder / 'training.json')
    assert summary['completed_updates'] == 3000
    assert summary['transitions'] == 3000 * 2 * 24
    assert summary['selection_checkpoints'] == [f'model_{n}.pt' for n in range(3499, 6000, 500)]
    assert ('load', str(folder.parent / 'warmstart/final.pt')) in calls
    count = len(calls)
    study.run_job(campaign, job_id)
    assert len(calls) == count
    attempts = [study.read(p) for p in campaign.parent.rglob('attempts/*.json')]
    arm_attempts = [a for a in attempts if 'accounting_scope' not in a]
    assert len(arm_attempts) == 3
    assert any(a['status'] == 'failed' for a in attempts)
    assert all(a['wall_seconds'] >= 0 and a['assigned_gpu_wall_seconds'] == 0 for a in attempts)
    (folder / summary['selection_checkpoints'][0]).write_text('corrupt')
    with pytest.raises(RuntimeError, match='Corrupt'):
        study.run_job(campaign, job_id)


def test_zero_exit_missing_artifacts_never_means_success(campaign, monkeypatch):
    monkeypatch.setattr(study.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0))
    with pytest.raises(FileNotFoundError):
        study.run_job(campaign, 'walk/mjlab/seed101/warmstart')
    assert not list(campaign.parent.rglob('done.json'))
    attempt = study.read(next(campaign.parent.rglob('attempts/*.json')))
    assert attempt['status'] == 'failed'


def test_backend_source_mutation_is_rejected(campaign):
    data = study.read(campaign)
    (Path(data['runtime']['backend_root']) / 'src/task.py').write_text('# changed')
    with pytest.raises(RuntimeError, match='Frozen source'):
        study.run_job(campaign, 'walk/mjlab/seed101/warmstart')


@pytest.mark.parametrize('method', ['static_search', 'mimicx'])
def test_full_search_feedback_budget_validation_gate_and_holdout(campaign, monkeypatch, method):
    calls, executed = fake_runtime(monkeypatch)
    result = study.run_job(campaign, f'walk/mjlab/seed101/{method}')
    folder = campaign.parent / 'jobs' / f'walk/mjlab/seed101/{method}'
    assert result['status'] == 'completed'
    assert len(executed) == 6000
    selection = study.read(folder / 'selection.json')
    assert len(selection['history']) == 6
    assert selection['history'][0]['accepted']
    assert not selection['history'][1]['accepted']
    assert selection['history'][1]['reason'] == 'Tracking guard'
    assert selection['history'][2]['incumbent'] == selection['history'][0]['candidate']
    assert len(list((folder / 'validation').glob('slot*/seed*/done.json'))) == 18
    assert len(list((folder / 'holdout').glob('*/seed*/done.json'))) == 6
    specs = [study.read(folder / name / 'stage.json') for name in (
        ('branch0', 'branch1') if method == 'static_search' else ('stage0', 'stage1'))]
    if method == 'static_search':
        assert specs[0]['parent'] == specs[1]['parent']
        assert all('warmstart/final.pt' in s['parent'] for s in specs)
        job = next(j for j in study.read(campaign)['jobs'] if j['id'].endswith('seed101/static_search'))
        assert [s['patch'] for s in specs] == [s['patch'] for s in job['stages']]
    else:
        assert specs[1]['parent'] == selection['history'][2]['candidate']
        assert (folder / 'stage0/diagnosis/failure_report.json').is_file()
        assert (folder / 'stage1/diagnosis/context.json').is_file()
    assert [s['updates'] for s in specs] == [1500, 1500]
    assert result['training']['transitions'] == 3000 * 2 * 24
    count = len(calls)
    study.run_job(campaign, f'walk/mjlab/seed101/{method}')
    assert len(calls) == count


def test_rejected_candidate_still_consumes_budget_and_holdout_never_controls_selection(campaign, monkeypatch):
    from mimicx.experiments import motion_study_loop as loop
    calls, executed = fake_runtime(monkeypatch)
    original = loop.evaluate

    def evaluate(*args, **kwargs):
        values = original(*args, **kwargs)
        if 'holdout' in str(args[4]):
            for repeat in values:
                repeat[0]['done_any'] = 'True'
        return values

    monkeypatch.setattr(loop, 'evaluate', evaluate)
    result = study.run_job(campaign, 'walk/mjlab/seed101/mimicx')
    assert len(executed) == 6000 and not result['training']['qualified']
    report = study.read(campaign.parent / 'jobs/walk/mjlab/seed101/mimicx/report.json')
    assert report['completed_updates'] == 3000
    assert 'stage1' in report['selected_checkpoint']
    assert all(not r['complete'] for r in report['holdout']['nominal'])


def test_common_validation_seed_contract():
    from mimicx.experiments.motion_study_loop import validation_protocol
    with pytest.raises(ValueError, match='three'):
        validation_protocol({'selection_seeds': [1, 2]})
    with pytest.raises(ValueError, match='disjoint'):
        validation_protocol({'selection_seeds': [2001, 1, 2]})


def test_adopt_warmstart_across_coordinator_versions(campaign, monkeypatch, tmp_path):
    calls, executed = fake_runtime(monkeypatch)
    study.run_job(campaign, 'walk/mjlab/seed101/warmstart')
    donor = campaign.parent / 'jobs/walk/mjlab/seed101/warmstart'
    other = tmp_path / 'version2'
    study.prepare(tmp_path / 'manifest.json', tmp_path / 'runtime.json', other)
    result = study.adopt_warmstart(other / 'campaign.json', 'walk/mjlab/seed101/warmstart', donor)
    assert result['status'] == 'warmstart_adopted'
    assert len(executed) == 3000
    assert study.digest(other / 'jobs/walk/mjlab/seed101/warmstart/final.pt') == study.digest(donor / 'final.pt')
    with pytest.raises(ValueError, match='seed'):
        study.adopt_warmstart(other / 'campaign.json', 'walk/mjlab/seed202/warmstart', donor)


@pytest.mark.parametrize('method', ['static_search', 'mimicx'])
def test_candidate_training_resume_counts_saved_updates_once(campaign, monkeypatch, method):
    calls, executed = fake_runtime(monkeypatch, interrupt=True, interrupt_at=3499)
    job_id = f'walk/mjlab/seed101/{method}'
    with pytest.raises(RuntimeError, match='interruption'):
        study.run_job(campaign, job_id)
    assert len(executed) == 3500
    result = study.run_job(campaign, job_id)
    assert result['status'] == 'completed'
    assert len(executed) == 6000
    folder = campaign.parent / 'jobs' / job_id
    stage = 'branch0' if method == 'static_search' else 'stage0'
    training = study.read(folder / stage / 'training.json')
    assert training['resume']['completed_updates'] == 500
    assert training['executed_updates_this_attempt'] == 1000
    assert len(list((folder / 'gates').glob('*.json'))) == 6


def test_failed_evaluation_keeps_attempt_and_reuses_completed_seeds(campaign, monkeypatch):
    calls, executed = fake_runtime(monkeypatch)
    normal_run = study.subprocess.run
    fail = [True]

    def run(command, **kwargs):
        result = normal_run(command, **kwargs)
        if '_evaluate' in command and 'slot00/seed1002' in ' '.join(command) and fail[0]:
            fail[0] = False
            return SimpleNamespace(returncode=1)
        return result

    monkeypatch.setattr(study.subprocess, 'run', run)
    job_id = 'walk/mjlab/seed101/static_search'
    with pytest.raises(RuntimeError, match='evaluation failed'):
        study.run_job(campaign, job_id)
    folder = campaign.parent / 'jobs' / job_id
    assert not (folder / 'done.json').exists()
    done = folder / 'validation/slot00/seed1001/done.json'
    old = done.read_bytes()
    assert study.run_job(campaign, job_id)['status'] == 'completed'
    assert len(executed) == 6000
    assert done.read_bytes() == old
    assert list((folder / 'validation/slot00/seed1002').glob('incomplete_rollout_*'))
    attempts = [study.read(p) for p in (folder / 'validation/slot00/seed1002/attempts').glob('*.json')]
    assert {a['status'] for a in attempts} == {'failed', 'completed'}


def test_evaluation_zero_exit_without_measurements_fails_closed(campaign, monkeypatch):
    from mimicx.experiments import motion_study_loop as loop
    data = study.read(campaign)
    job = data['jobs'][0]
    checkpoint = campaign.parent / 'checkpoint.pt'
    checkpoint.write_bytes(b'trusted checkpoint fixture')
    monkeypatch.setattr(loop.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0))
    with pytest.raises(FileNotFoundError):
        loop.evaluate(campaign, data, job, checkpoint, campaign.parent / 'evaluation', [1001, 1002, 1003], .005)
    assert not list((campaign.parent / 'evaluation').rglob('done.json'))


def test_startup_failure_diagnosis_never_invents_post_reset_body_evidence(campaign):
    from mimicx.experiments import motion_study_loop as loop
    from tests.test_policy_recovery_metrics import rows
    data = study.read(campaign)
    task = data['tasks'][0]
    repeats = [rows(loop.horizon(task), failure=1, error=9.) for _ in range(3)]
    context = loop.diagnose(data, task, repeats, campaign.parent / 'startup-diagnosis')
    assert context.dominant_bodies == () and context.top_metrics == ()
    assert context.first_failure_step == 1
    assert loop.accept(repeats, [rows(loop.horizon(task)) for _ in range(3)], loop.horizon(task))['accepted']
    from mimicx.policy_recovery.metrics import accept as legacy_accept
    assert not legacy_accept(repeats, [rows(loop.horizon(task)) for _ in range(3)], loop.horizon(task))['accepted']


def test_rejected_stage_does_not_repeat_identical_patch_parent_and_seed(campaign, monkeypatch):
    from mimicx.experiments import motion_study_loop as loop
    calls, executed = fake_runtime(monkeypatch)
    gate = loop.accept
    count = [0]

    def reject_first_stage(old, new, n):
        result = gate(old, new, n)
        count[0] += 1
        if count[0] <= 3:
            result.update(accepted=False, reason='Controlled rejection for history test')
        return result

    monkeypatch.setattr(loop, 'accept', reject_first_stage)
    study.run_job(campaign, 'walk/mjlab/seed101/mimicx')
    folder = campaign.parent / 'jobs/walk/mjlab/seed101/mimicx'
    first, second = (study.read(folder / stage / 'stage.json') for stage in ('stage0', 'stage1'))
    assert first['parent'] == second['parent']
    assert first['patch'] != second['patch']
    assert first['patch']['sampling']['ratio'] == .45
    assert second['patch']['sampling']['ratio'] == .65
    assert len(executed) == 6000
    history = study.read(folder / 'stage1/proposal_history.json')
    assert history['tried'] == [first['patch']]


def test_untried_candidate_signature_does_not_depend_on_id():
    from mimicx.experiments.motion_study_loop import untried_patch
    from mimicx.refinement.proposals import ProposalContext
    context = ProposalContext('walk', 300, 1, 20, (1, 50), (), (), (), ())
    first = untried_patch(context, [])
    renamed = {**first, 'id': 'renamed_but_same_configuration'}
    second = untried_patch(context, [renamed])
    assert first['sampling'] != second['sampling']


def test_intermediate_checkpoint_diagnostic_receipts_do_not_complete_warmstart(campaign, monkeypatch):
    from mimicx.experiments.motion_study_loop import validate_checkpoint
    calls, executed = fake_runtime(monkeypatch)
    checkpoint = campaign.parent / 'model_499.pt'
    checkpoint.write_bytes(b'intermediate trusted weight fixture')
    output = campaign.parent / 'diagnostics/warm500'
    result = validate_checkpoint(campaign, 'walk/mjlab/seed101/warmstart', checkpoint, output)
    assert result['status'] == 'diagnostic_evaluated_not_selected'
    assert not executed
    assert len(list((output / 'evaluation').glob('seed*/done.json'))) == 3
    assert not list((campaign.parent / 'jobs').rglob('done.json'))
    count = len(calls)
    validate_checkpoint(campaign, 'walk/mjlab/seed101/warmstart', checkpoint, output)
    assert len(calls) == count
    with pytest.raises(ValueError, match='separate'):
        validate_checkpoint(campaign, 'walk/mjlab/seed101/warmstart', checkpoint, campaign.parent / 'jobs/diagnostic')


def test_full_protocol_hash_bound_to_ground_contact_tasks(campaign, tmp_path):
    from mimicx.experiments.motion_study_scoring import common_protocol
    import hashlib
    manifest = study.read(tmp_path / 'manifest.json')
    protocol = {**common_protocol(), 'protocol_id': 'prospective-ground-v1', 'absolute_height_guard': False}
    sha = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    manifest.update(protocol=protocol, protocol_sha256=sha)
    manifest['tasks'][0].update(requires_ground_contact_protocol=True, protocol_id=protocol['protocol_id'], protocol_sha256=sha)
    study.save_json(tmp_path / 'manifest.json', manifest)
    data = study.prepare(tmp_path / 'manifest.json', tmp_path / 'runtime.json', tmp_path / 'prospective')
    assert data['common_protocol_sha256'] == sha
    assert data['runtime']['common_protocol'] == protocol
    manifest['tasks'][0]['protocol_sha256'] = 'unfrozen'
    study.save_json(tmp_path / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='Ground-contact'):
        study.prepare(tmp_path / 'manifest.json', tmp_path / 'runtime.json', tmp_path / 'bad-protocol')


def test_diagnosis_keeps_true_failure_timing_and_excludes_reset_coordinates(campaign):
    from mimicx.experiments import motion_study_loop as loop
    from tests.test_policy_recovery_metrics import rows
    data = study.read(campaign)
    task = data['tasks'][0]
    repeats = [rows(loop.horizon(task), failure=20) for _ in range(3)]
    for repeat in repeats:
        for row in repeat[19:]:
            row['body_pos_error_max'] = 9.
        for row in repeat:
            row['robot_anchor_z'] = 100.
            row['ref_anchor_z'] = 100.
    output = campaign.parent / 'diagnosis-valid-prefix'
    context = loop.diagnose(data, task, repeats, output)
    assert context.first_failure_step == 20
    assert all(item['max'] < 1. for item in context.top_metrics)
    with (output / 'diagnosis_steps.csv').open() as stream:
        header = stream.readline()
    assert 'robot_anchor_z' not in header and 'ref_anchor_z' not in header


def test_aligned_legacy_v2_donor_without_new_contract_field_can_be_adopted(campaign, monkeypatch, tmp_path):
    fake_runtime(monkeypatch)
    study.run_job(campaign, 'walk/mjlab/seed101/warmstart')
    donor = campaign.parent / 'jobs/walk/mjlab/seed101/warmstart'
    effective = study.read(donor / 'effective_training.json')
    effective.pop('initial_reference_refresh')
    study.save_json(donor / 'effective_training.json', effective)
    receipt = study.read(donor / 'done.json')
    receipt['artifacts']['effective_training.json'] = study.digest(donor / 'effective_training.json')
    study.save_json(donor / 'done.json', receipt)
    target = tmp_path / 'new-frozen-coordinator'
    study.prepare(tmp_path / 'manifest.json', tmp_path / 'runtime.json', target)
    study.adopt_warmstart(target / 'campaign.json', 'walk/mjlab/seed101/warmstart', donor)
    adoption = study.read(target / 'jobs/walk/mjlab/seed101/warmstart/adoption.json')
    assert adoption['legacy_refresh_source_audit']['initial_reference_refresh']
    assert adoption['training_cost_reused_not_zero']


@pytest.mark.parametrize('code', [None, 0, 1, 2])
def test_rollout_cli_zero_exit_allows_post_rollout_scoring(monkeypatch, code):
    from mimicx.experiments import motion_study_loop as loop
    def run(*args, **kwargs):
        raise SystemExit(code)
    monkeypatch.setattr(loop.runpy, 'run_path', run)
    scored = []
    if code in (None, 0):
        loop.run_rollout_cli('rollout.py')
        scored.append('post_rollout_common_metrics')
        assert scored
    else:
        with pytest.raises(SystemExit):
            loop.run_rollout_cli('rollout.py')
        assert not scored


def test_zero_exit_rollout_requires_captured_states_and_score_artifact(campaign, monkeypatch):
    from mimicx.experiments import motion_study_loop as loop, motion_study_scoring as scoring
    data = study.read(campaign)
    data['reference_contract'] = {'joint_names': ['joint'], 'body_names': ['pelvis']}
    job = data['jobs'][0]
    cfg = SimpleNamespace(commands={'motion': SimpleNamespace(anchor_body_name='torso_link')}, viewer=SimpleNamespace())
    envs = ModuleType('mjlab.envs')
    envs.ManagerBasedRlEnv = type('Env', (), {})
    package = ModuleType('mjlab')
    package.envs = envs
    registry = ModuleType('mjlab.tasks.registry')
    registry.load_env_cfg = lambda *args, **kwargs: cfg
    registry.load_rl_cfg = lambda *args: SimpleNamespace()
    registry.register_mjlab_task = lambda *args: None
    monkeypatch.setitem(sys.modules, 'mjlab', package)
    monkeypatch.setitem(sys.modules, 'mjlab.envs', envs)
    monkeypatch.setitem(sys.modules, 'mjlab.tasks.registry', registry)
    monkeypatch.setattr(study, 'native_configs', lambda *args: None)
    monkeypatch.setattr('mimicx.parkour.environment.install_initial_reference_refresh', lambda: None)
    def capture(base, contract, samples):
        samples.extend([({'frame': i}, False) for i in range(data['tasks'][0]['frames'])])
        return base
    monkeypatch.setattr(scoring, 'capture_class', capture)
    def run(*args, **kwargs):
        raise SystemExit(0)
    monkeypatch.setattr(loop.runpy, 'run_path', run)
    calls = []
    def score(actual_campaign, task, states, output, **kwargs):
        assert len(states) == task['frames'] and states[0][0]['frame'] == 0
        assert states[-1][0]['frame'] == task['frames'] - 1
        assert kwargs == dict(align_rotation=True, anchor_body='torso_link')
        calls.append('score_capture')
        study.save_json(output / 'common_metrics.json', {'frames': len(states), 'source': 'mocked control-flow regression'})
    monkeypatch.setattr(scoring, 'score_capture', score)
    args = SimpleNamespace(checkpoint=campaign.parent / 'weights.pt', output=campaign.parent / 'post-rollout', seed=1001, noise=.005)
    loop.evaluate_native(data, job, args)
    assert calls == ['score_capture']
    assert study.read(args.output / 'common_metrics.json')['frames'] == data['tasks'][0]['frames']


@pytest.mark.parametrize('dependency', ['sonic_study.py', 'sonic_helpers.py'])
def test_common_scorer_and_helper_are_frozen_and_drift_fails_closed(campaign, tmp_path, monkeypatch, dependency):
    root = tmp_path / 'code-snapshot'
    files = ['mimicx/refinement/proposals.py', 'scripts/autorefine/mine_failures.py',
             'scripts/rollout_mjlab_motion_smoke.py', 'mimicx/parkour/environment.py',
             'mimicx/visualization/replay_package.py',
             'mimicx/experiments/sonic_study.py', 'mimicx/experiments/sonic_helpers.py']
    for name in files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# immutable source fixture\n')
    model = tmp_path / 'canonical.xml'
    model.write_text('<mujoco/>')
    monkeypatch.setattr(study, 'ROOT', root)
    data = study.read(campaign)
    data['runtime']['common_model_xml'] = str(model)
    data['sources'] = study.source_identity(data['runtime'])
    for name in ('sonic_study.py', 'sonic_helpers.py'):
        path = root / 'mimicx/experiments' / name
        assert data['sources'][str(path)] == study.digest(path)
    study.verify_campaign(data)
    changed = root / 'mimicx/experiments' / dependency
    changed.write_text('# changed scorer dependency\n')
    with pytest.raises(RuntimeError, match=dependency):
        study.verify_campaign(data)


def test_common_scoring_refuses_missing_helper_source(campaign, tmp_path, monkeypatch):
    original = study.ROOT
    root = tmp_path / 'incomplete-code-snapshot'
    for name in ('mimicx/refinement/proposals.py', 'scripts/autorefine/mine_failures.py',
                 'scripts/rollout_mjlab_motion_smoke.py', 'mimicx/parkour/environment.py',
                 'mimicx/visualization/replay_package.py',
                 'mimicx/experiments/sonic_study.py'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((original / name).read_bytes())
    monkeypatch.setattr(study, 'ROOT', root)
    runtime = {**study.read(campaign)['runtime'], 'common_model_xml': str(tmp_path / 'model.xml')}
    Path(runtime['common_model_xml']).write_text('<mujoco/>')
    with pytest.raises(FileNotFoundError, match='sonic_helpers.py'):
        study.source_identity(runtime)


@pytest.mark.parametrize('dependency', ['replay_package.py', 'parts.xml', 'robot.stl', 'skin.png'])
def test_replay_and_model_asset_closure_are_frozen(campaign, tmp_path, monkeypatch, dependency):
    root = tmp_path / 'code'
    replay = root / 'mimicx/visualization/replay_package.py'
    replay.parent.mkdir(parents=True)
    replay.write_text('# frozen replay exporter\n')
    model = tmp_path / 'model.xml'
    model.write_text('<mujoco><compiler meshdir="meshes" texturedir="textures"/><include file="parts.xml"/></mujoco>')
    include = tmp_path / 'parts.xml'
    include.write_text('<mujoco><asset><mesh file="robot.stl"/><texture file="skin.png"/></asset></mujoco>')
    mesh, texture = tmp_path / 'meshes/robot.stl', tmp_path / 'textures/skin.png'
    for path in (mesh, texture):
        path.parent.mkdir()
        path.write_bytes(b'frozen asset fixture')
    # Copy only source fixtures; source_identity never imports these fake files.
    for name in ('mimicx/refinement/proposals.py', 'scripts/autorefine/mine_failures.py',
                 'scripts/rollout_mjlab_motion_smoke.py', 'mimicx/parkour/environment.py',
                 'mimicx/experiments/sonic_study.py', 'mimicx/experiments/sonic_helpers.py'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((study.ROOT / name).read_bytes())
    monkeypatch.setattr(study, 'ROOT', root)
    data = study.read(campaign)
    data['runtime']['common_model_xml'] = str(model)
    data['sources'] = study.source_identity(data['runtime'])
    for path in (replay, model, include, mesh, texture):
        assert data['sources'][str(path)] == study.digest(path)
    study.verify_campaign(data)
    changed = {p.name: p for p in (replay, include, mesh, texture)}[dependency]
    changed.write_bytes(changed.read_bytes() + b'changed')
    with pytest.raises(RuntimeError, match='Frozen source'):
        study.verify_campaign(data)


def test_expanded_asset_preflight_before_training_and_adoption_receipt(campaign, tmp_path, monkeypatch):
    from mimicx.experiments.motion_study_adoption import build_asset_proof
    old = study.read(campaign)
    backend = Path(old['runtime']['backend_root'])
    model, mesh = backend / 'src/assets/g1.xml', backend / 'src/assets/mesh.stl'
    model.parent.mkdir()
    model.write_text('<mujoco><asset><mesh file="mesh.stl"/></asset></mujoco>')
    mesh.write_bytes(b'immutable tracked mesh fixture')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(backend), *args], stderr=subprocess.DEVNULL)
    git('init')
    git('add', '.')
    git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', '-c', 'commit.gpgsign=false',
        'commit', '-m', 'Synthetic legacy source')
    commit = git('rev-parse', 'HEAD').decode().strip()
    proof_path = tmp_path / 'legacy-asset-proof.json'
    proof = build_asset_proof(backend, model, commit, proof_path)
    old['runtime']['common_model_xml'] = str(model)
    old['sources'] = study.source_identity(old['runtime'])
    old['sources'].pop(str(mesh))  # Reproduce the old v2 XML-only identity schema.
    study.save_json(campaign, old)
    runtime_path = tmp_path / 'runtime.json'
    runtime = study.read(runtime_path)
    runtime.update(common_model_xml=str(model), legacy_backend_asset_proof=str(proof_path),
                   legacy_backend_asset_proof_sha256=proof['sha256'], legacy_backend_asset_commit=commit)
    study.save_json(runtime_path, runtime)
    target = tmp_path / 'expanded-identity-campaign'
    study.prepare(tmp_path / 'manifest.json', runtime_path, target)
    preflight = tmp_path / 'adoption-preflight.json'
    study.main(['check-adoption', '--campaign', str(target / 'campaign.json'),
                '--donor-campaign', str(campaign), '--output', str(preflight)])
    assert study.read(preflight)['status'] == 'backend_compatible'
    assert not list(campaign.parent.rglob('*.pt'))
    fake_runtime(monkeypatch)
    study.run_job(campaign, 'walk/mjlab/seed101/warmstart')
    donor = campaign.parent / 'jobs/walk/mjlab/seed101/warmstart'
    study.adopt_warmstart(target / 'campaign.json', 'walk/mjlab/seed101/warmstart', donor)
    audit = study.read(target / 'jobs/walk/mjlab/seed101/warmstart/adoption.json')['backend_identity_audit']
    assert audit['newly_recorded_paths'] == ['src/assets/mesh.stl']
    assert audit['mode'] == 'expanded_asset_coverage_with_pinned_git_proof'
    assert audit['proof_sha256'] == proof['sha256']
    assert not audit['historical_donor_asset_hashes_claimed']
