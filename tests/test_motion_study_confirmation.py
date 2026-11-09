"""Reduced confirmation contracts and CPU-fake dispatch, not physical evidence."""
import copy
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mimicx.experiments import motion_study as study
from mimicx.experiments import motion_study_confirmation as confirmation
from mimicx.experiments import motion_study_loop as loop
from tests.test_motion_study_campaign import campaign
from tests.test_motion_study_v4 import evidence, repeats


@pytest.fixture
def inputs(campaign, tmp_path):
    manifest_path, runtime_path = tmp_path / 'manifest.json', tmp_path / 'runtime.json'
    manifest, runtime = study.read(manifest_path), study.read(runtime_path)
    model = tmp_path / 'model.xml'
    model.write_text('<mujoco/>')
    manifest['canonical_g1'] = dict(joint_names=[f'joint{i}' for i in range(29)],
                                   body_names=[f'body{i}' for i in range(30)], xml_sha256=study.digest(model))
    manifest['tasks'] = [dict(manifest['tasks'][0], task_id=f'unseen_{i}', split='confirmation') for i in range(12)]
    runtime.update(refinement_strategy=confirmation.CONFIRMATION_STRATEGY, common_model_xml=str(model),
                   validation_protocol=dict(holdout={'confirmation': copy.deepcopy(confirmation.HOLDOUT_SEEDS)}))
    study.save_json(manifest_path, manifest)
    study.save_json(runtime_path, runtime)
    return manifest_path, runtime_path


def freeze(inputs, tmp_path):
    manifest, runtime_path = inputs
    contract_path = tmp_path / 'freeze.json'
    record = confirmation.build_confirmation_contract(manifest, runtime_path, contract_path)
    runtime = study.read(runtime_path)
    runtime.update(confirmation_contract=contract_path.name, confirmation_contract_sha256=study.digest(contract_path))
    study.save_json(runtime_path, runtime)
    output = tmp_path / 'confirmation'
    data = study.prepare(manifest, runtime_path, output)
    return output / 'campaign.json', data, record


def test_prepare_roundtrip_and_approved_matrix(inputs, tmp_path):
    with pytest.raises(ValueError, match='confirmation_contract'):
        study.prepare(*inputs, tmp_path / 'missing-freeze')
    assert not (tmp_path / 'missing-freeze').exists()
    path, data, record = freeze(inputs, tmp_path)
    assert study.prepare(*inputs, path.parent) == data
    assert confirmation.build_confirmation_contract(*inputs, tmp_path / 'freeze.json') == record
    study.verify_campaign(study.read(path))
    assert record['algorithm'] == study.V6_STRATEGY
    assert record['gate'] == study.V4_STRATEGY
    assert len(record['task_ids']) == 12
    assert len(data['jobs']) == 96
    assert sum(j['updates'] for j in data['jobs']) == 288000
    assert sum(j['method'] == 'warmstart' for j in data['jobs']) == 24
    assert {t['split'] for t in data['tasks']} == {'confirmation'}
    assert {j['seed'] for j in data['jobs']} == {101, 202}
    assert data['runtime']['validation_protocol']['protocol_version'] == confirmation.PROTOCOL_VERSION
    assert data['runtime']['validation_protocol']['holdout']['confirmation'] == confirmation.HOLDOUT_SEEDS
    assert str(Path(confirmation.__file__).resolve()) in record['sources']
    assert not list(path.parent.rglob('*.pt'))
    assert not list(path.parent.rglob('done.json'))


@pytest.mark.parametrize('fault', [
    'third_training_seed', 'float_training_seed', 'duplicate_training_seed', 'selection_seed',
    'holdout_overlap', 'holdout_duplicate', 'old_holdout', 'missing_holdout', 'float_holdout',
    'noise', 'version', 'few_tasks', 'duplicate_task', 'development', 'blocked', 'canonical',
    'canonical_names', 'strategy', 'backend', 'physical',
])
def test_builder_rejects_unapproved_scope(inputs, tmp_path, fault):
    manifest_path, runtime_path = inputs
    manifest, runtime = study.read(manifest_path), study.read(runtime_path)
    p = runtime['validation_protocol']
    if fault == 'third_training_seed':
        runtime['seeds'] = {'confirmation': [101, 202, 303]}
    elif fault == 'float_training_seed':
        runtime['seeds'] = {'confirmation': [101., 202]}
    elif fault == 'duplicate_training_seed':
        runtime['seeds'] = {'confirmation': [101, 101]}
    elif fault == 'selection_seed':
        p['selection_seeds'] = [1001, 1002, 1004]
    elif fault == 'holdout_overlap':
        p['holdout']['confirmation']['nominal'][0] = 1001
    elif fault == 'holdout_duplicate':
        p['holdout']['confirmation']['nominal'][1] = 21001
    elif fault == 'old_holdout':
        p['holdout']['confirmation'] = loop.validation_protocol()['holdout']['confirmation']
    elif fault == 'missing_holdout':
        del p['holdout']
    elif fault == 'float_holdout':
        p['holdout']['confirmation']['nominal'][0] = 21001.
    elif fault == 'noise':
        p['perturbed_noise'] = .02
    elif fault == 'version':
        p['protocol_version'] = 'old'
    elif fault == 'few_tasks':
        manifest['tasks'].pop()
    elif fault == 'duplicate_task':
        manifest['tasks'][1]['task_id'] = manifest['tasks'][0]['task_id']
    elif fault == 'development':
        manifest['tasks'][0]['split'] = 'development'
    elif fault == 'blocked':
        manifest['tasks'][0]['quality_status'] = 'blocked'
    elif fault == 'canonical':
        del manifest['canonical_g1']
    elif fault == 'canonical_names':
        manifest['canonical_g1']['body_names'].pop()
    elif fault == 'strategy':
        runtime['refinement_strategy'] = 'joint_adaptive_precision_v7'
    elif fault == 'backend':
        runtime['backends'] = ['mjlab', 'isaaclab']
    elif fault == 'physical':
        manifest['protocol'] = {'protocol_id': 'changed'}
    study.save_json(manifest_path, manifest)
    study.save_json(runtime_path, runtime)
    with pytest.raises(ValueError):
        confirmation.build_confirmation_contract(*inputs, tmp_path / 'invalid-freeze.json')
    assert not (tmp_path / 'invalid-freeze.json').exists()


@pytest.mark.parametrize('fault', [
    'task_id', 'split', 'frames', 'training_seed', 'selection_seed', 'holdout_seed', 'noise',
    'budget', 'updates', 'slots', 'stage_budget', 'stage_patch', 'job_removed', 'job_seed',
    'reference', 'physical', 'runtime', 'strategy', 'source_removed', 'source_changed',
    'source_added', 'motion_changed', 'freeze_file', 'freeze_rehashed', 'missing_pin',
])
def test_tamper_rejected_before_run(inputs, tmp_path, monkeypatch, fault):
    path, data, _ = freeze(inputs, tmp_path)
    arm = next(j for j in data['jobs'] if j['method'] == 'mimicx')
    p = data['runtime']['validation_protocol']
    if fault == 'task_id':
        data['tasks'][0]['task_id'] = 'replacement'
    elif fault == 'split':
        data['tasks'][0]['split'] = 'development'
    elif fault == 'frames':
        data['tasks'][0]['frames'] += 50
    elif fault == 'training_seed':
        data['runtime']['seeds']['confirmation'] = [101, 202, 303]
    elif fault == 'selection_seed':
        p['selection_seeds'][0] = 21001
    elif fault == 'holdout_seed':
        p['holdout']['confirmation']['nominal'][0] = 1001
    elif fault == 'noise':
        p['nominal_noise'] = .01
    elif fault == 'budget':
        data['budget']['per_arm'] = 3500
    elif fault == 'updates':
        arm['updates'] = 3500
    elif fault == 'slots':
        arm['selection_slots'] = 7
    elif fault == 'stage_budget':
        arm['stages'][0]['updates'] = 1000
    elif fault == 'stage_patch':
        arm['stages'][0]['patch'] = {'id': 'tuned'}
    elif fault == 'job_removed':
        data['jobs'].pop()
    elif fault == 'job_seed':
        arm['seed'] = 303
    elif fault == 'reference':
        data['reference_contract']['body_names'][0] = 'different'
    elif fault == 'physical':
        data['runtime']['common_protocol']['root_height_m'] = .5
    elif fault == 'runtime':
        data['runtime']['num_envs'] = 4
    elif fault == 'strategy':
        data['runtime']['refinement_strategy'] = 'legacy_v3'
    elif fault == 'source_removed':
        data['sources'].pop(str(Path(confirmation.__file__).resolve()))
    elif fault == 'source_changed':
        (tmp_path / 'backend/src/task.py').write_text('# altered')
    elif fault == 'source_added':
        (tmp_path / 'backend/src/extra.py').write_text('# new source')
    elif fault == 'motion_changed':
        (tmp_path / 'motion.npz').write_bytes(b'changed')
    elif fault in ('freeze_file', 'freeze_rehashed'):
        record = study.read(tmp_path / 'freeze.json')
        record['algorithm'] = 'v7'
        study.save_json(tmp_path / 'freeze.json', record)
        if fault == 'freeze_rehashed':
            data['runtime']['confirmation_contract_sha256'] = study.digest(tmp_path / 'freeze.json')
    elif fault == 'missing_pin':
        del data['runtime']['confirmation_contract_sha256']
    study.save_json(path, data)
    monkeypatch.setattr(study, '_execute', lambda *a: pytest.fail('Must reject before execution'))
    with pytest.raises((ValueError, RuntimeError)):
        study.run_job(path, arm['id'])
    assert not list(path.parent.rglob('attempts/*.json'))


def test_freeze_immutable_and_preparation_drift(inputs, tmp_path):
    _, _, _ = freeze(inputs, tmp_path)
    runtime = study.read(inputs[1])
    runtime['num_envs'] = 4
    study.save_json(inputs[1], runtime)
    with pytest.raises(ValueError, match='Immutable'):
        confirmation.build_confirmation_contract(*inputs, tmp_path / 'freeze.json')
    with pytest.raises(ValueError, match='does not match'):
        study.prepare(*inputs, tmp_path / 'changed')
    assert not (tmp_path / 'changed').exists()


@pytest.mark.parametrize('method', ['warmstart', 'mimicx'])
def test_worker_rejects_missing_freeze_before_training(inputs, tmp_path, monkeypatch, method):
    path, data, _ = freeze(inputs, tmp_path)
    del data['runtime']['confirmation_contract_sha256']
    study.save_json(path, data)
    job = next(j for j in data['jobs'] if j['method'] == method)
    monkeypatch.setattr(study, 'train_native', lambda *a: pytest.fail('Must validate before training'))
    monkeypatch.setattr(loop, 'evaluate', lambda *a: pytest.fail('Must validate before evaluation'))
    with pytest.raises(ValueError, match='confirmation_contract'):
        study.main(['_worker', '--campaign', str(path), '--job', job['id']])
    if method == 'mimicx':
        with pytest.raises(ValueError, match='confirmation_contract'):
            study.execute_arm_v4(path, data, job, path.parent / 'direct')


def test_confirmation_requires_fresh_warmstarts(inputs, tmp_path, monkeypatch):
    path, data, _ = freeze(inputs, tmp_path)
    job = next(j for j in data['jobs'] if j['method'] == 'warmstart')
    monkeypatch.setattr(study, 'verified_receipt', lambda *a: pytest.fail('Must not consume donor'))
    with pytest.raises(ValueError, match='fresh scratch warmstarts'):
        study.adopt_warmstart(path, job['id'], tmp_path / 'unused-donor')


@pytest.mark.parametrize('name,expected', [
    ('refinement_v4', '0f77ae24910f2e18e47a66e7ee517dd7005824c329cfb174fd7b9366d8c82f18'),
    ('refinement_v5', '1fbb31573378f78463a88ebf46eb123ef410ba0192e53003a0a8571409441440'),
    ('native_replay_needed_v6', '053a700b1c3224d285addba0e5bb275bc0401019a512a7522b2579e83cdefbb8'),
    ('refinement_v6', '4b5ac93fa84a0d2fbd3e6b359363a6a20d52875333ad80b1b9d349ddc1f51720'),
    ('accept_v4', 'a7cc0d414c7ef4f482395ee91796001975b3eb648a6d5e8cec805d4d646aa910'),
])
def test_original_v6_proposal_and_gate_source_bytes(name, expected):
    # Pins from the approved v6 jog snapshot; source text is independent of AST version.
    assert hashlib.sha256(inspect.getsource(getattr(study, name)).encode()).hexdigest() == expected


@pytest.mark.parametrize('method', ['mimicx', 'static_search', 'beyond_mimic'])
@pytest.mark.parametrize('bad_holdout', [False, True])
def test_real_run_and_worker_dispatch_v6_and_holdout_isolation(inputs, tmp_path, monkeypatch, method, bad_holdout):
    path, data, _ = freeze(inputs, tmp_path)
    job = next(j for j in data['jobs'] if j['method'] == method)
    task = data['tasks'][0]
    n = loop.horizon(task)
    folder = path.parent / 'jobs' / job['id']
    warm = folder.parent / 'warmstart/final.pt'
    training, proposals, evaluations, held_hashes = [], [], [], []
    actual_v6 = study.refinement_v6

    def propose(*args):
        assert not held_hashes
        assert args[1]['split'] == 'confirmation'
        result = actual_v6(*args)
        proposals.append(copy.deepcopy(result))
        return result

    def train(c, j, output, *, stage=None):
        updates = stage['updates'] if stage else j['updates']
        if stage:
            assert stage['parent'] == str(warm)
        training.append((j['method'], updates))
        output.mkdir(parents=True, exist_ok=True)
        slots = [f'model_{i}.pt' for i in range(499, updates, 500)]
        for name in slots + ['final.pt']:
            (output / name).write_bytes(str(output / name).encode())
        study.save_json(output / 'effective_training.json', {'cpu_test': True})
        summary = dict(status='training_complete_selection_pending', completed_updates=updates,
                       transitions=updates * c['runtime']['num_envs'] * 24,
                       selection_checkpoints=[] if j['method'] == 'warmstart' else slots)
        study.save_json(output / 'training.json', summary)
        return summary

    def evaluate(campaign_path, c, j, checkpoint, output, seeds, noise):
        is_holdout = 'holdout' in output.parts
        evaluations.append((is_holdout, list(seeds)))
        if is_holdout:
            assert len(evaluations) >= 8
            selected = study.read(folder / 'selection.json')
            assert selected['checkpoint'] == str(checkpoint)
            assert selected['uses_holdout'] is False
            assert seeds == confirmation.HOLDOUT_SEEDS[output.name]
            held_hashes.append(study.digest(folder / 'selection.json'))
        else:
            assert not held_hashes
            assert seeds == confirmation.SELECTION_SEEDS
        candidate = checkpoint != warm
        first_precision = 'stage0' in checkpoint.parts or method != 'mimicx'
        sacrifice = candidate and first_precision
        ev = evidence(.03 if candidate else .05, (.025 if sacrifice else .01) if candidate else .02, n)
        for seed, item in zip(seeds, ev):
            target = output / f'seed{seed}' / 'rollout'
            target.mkdir(parents=True, exist_ok=True)
            np.savez(target / 'common_steps.npz', **{k: v for k, v in item.items() if k != 'pre_failure_frames'})
        failed = is_holdout and bad_holdout
        study.save_json(output / 'summary.json', dict(common_metrics=[
            dict(pre_failure_frames=20 if failed else n + 1, complete_motion=not failed)] * 3))
        return repeats(n=n, failure=20) if failed else repeats(n=n)

    def child(command, **kwargs):
        assert command[1] == str(study.ENTRY)
        assert command[2] == '_worker'
        study.main(command[2:])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(study, 'train_native', train)
    monkeypatch.setattr(study, 'refinement_v6', propose)
    monkeypatch.setattr(loop, 'evaluate', evaluate)
    monkeypatch.setattr(loop, 'execute_arm', lambda *a: pytest.fail('Legacy dispatcher'))
    monkeypatch.setattr(study.subprocess, 'run', child)
    study.main(['run', '--campaign', str(path), '--task', task['task_id'], '--seed', '101', '--method', method])
    report = study.read(folder / 'report.json')
    selection = study.read(folder / 'selection.json')
    assert training[0] == ('warmstart', 3000)
    assert sum(updates for name, updates in training if name == method) == 3000
    assert len(evaluations) == 9
    assert held_hashes == [study.digest(folder / 'selection.json')] * 2
    assert len(selection['history']) == 6
    assert selection['criterion'] == confirmation.CONFIRMATION_STRATEGY
    assert all(g['gate_version'] == study.V4_STRATEGY and not g['uses_holdout'] for g in selection['history'])
    assert report['refinement_strategy'] == confirmation.CONFIRMATION_STRATEGY
    assert report['metrics_contract'] == study.V4_STRATEGY
    assert report['budget_accounting']['independent_holdout_rollouts'] == 6
    assert report['budget_accounting']['candidate_selection_rollouts'] == 18
    assert report['common_physical_qualified'] is not bad_holdout
    if bad_holdout:
        assert not report['qualified']
    if method == 'mimicx':
        assert proposals[0] == loop.static_patches(task)[0]
        assert proposals[1] == study.refinement_v4(data, task, repeats(n=n), tmp_path, proposals[:1])
        assert proposals[1]['id'] == 'native_full_replay_45'
    else:
        assert not proposals
    assert study.verified_receipt(folder)['status'] == 'completed'
    calls = len(training), len(evaluations), len(proposals)
    study.run_job(path, job['id'])
    assert (len(training), len(evaluations), len(proposals)) == calls


def test_old_protocol_and_new_training_defaults_remain_separate(inputs):
    old = loop.validation_protocol()
    assert len(old['holdout']['confirmation']['nominal']) == 10
    with pytest.raises(ValueError, match='Insufficient holdout'):
        loop.validation_protocol({'holdout': dict(old['holdout'], confirmation=confirmation.HOLDOUT_SEEDS)})
    task = dict(task_id='unseen', split='confirmation', training_eligible=True, frames=301)
    assert {j['seed'] for j in study.make_jobs([task], {})} == {101, 202, 303}
    assert {j['seed'] for j in study.make_jobs([task], {'refinement_strategy': confirmation.CONFIRMATION_STRATEGY})} == {101, 202}


def test_base_runtime_schema_with_explicit_seed_replacement():
    raw = loop.validation_protocol()
    with pytest.raises(ValueError, match=r'3\+3'):
        loop.validation_protocol(raw, strategy=confirmation.CONFIRMATION_STRATEGY)
    raw['holdout']['confirmation'] = copy.deepcopy(confirmation.HOLDOUT_SEEDS)
    normalized = loop.validation_protocol(raw, strategy=confirmation.CONFIRMATION_STRATEGY)
    assert normalized['holdout']['development'] == raw['holdout']['development']
    assert normalized['protocol_version'] == confirmation.PROTOCOL_VERSION
