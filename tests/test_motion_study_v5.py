"""CPU contracts for the private development proposal; not performance evidence."""
import copy

import numpy as np
import pytest

from mimicx.experiments import motion_study as study
from mimicx.experiments import motion_study_loop as loop
from tests.test_motion_study_campaign import campaign
from tests.test_motion_study_v4 import evidence, repeats


V5 = 'joint_precision_v5'


@pytest.mark.parametrize('error', [.05, .8])
def test_saturated_uses_exact_static_pair_even_when_inaccurate(tmp_path, monkeypatch, error):
    task = dict(task_id='walk', split='development', frames=41)
    before = repeats(error=error)
    original = copy.deepcopy(before)
    monkeypatch.setattr(loop, 'diagnose', lambda *a: pytest.fail('Saturated: no diagnosis'))
    first = study.refinement_v5({}, task, before, tmp_path, [])
    second = study.refinement_v5({}, task, before, tmp_path, [{**first, 'id': 'renamed'}])
    assert [first, second] == loop.static_patches(task)
    assert before == original
    with pytest.raises(ValueError, match='untried'):
        study.refinement_v5({}, task, before, tmp_path, [first, second])


def test_one_incomplete_repeat_keeps_existing_diagnosis(tmp_path, monkeypatch):
    from mimicx.refinement.proposals import ProposalContext
    task = dict(task_id='walk', split='development', frames=41)
    context = ProposalContext('walk', 40, 1, 20, (5, 30), (), (), (), ())
    calls = []
    monkeypatch.setattr(loop, 'diagnose', lambda *a: calls.append(a) or context)
    data = repeats()
    data[2] = repeats(failure=20)[0]
    tried = [loop.untried_patch(context, [])]
    assert study.refinement_v5({}, task, data, tmp_path, tried) == loop.untried_patch(context, tried)
    assert calls == [({}, task, data, tmp_path)]


@pytest.mark.parametrize('split', ['development', 'confirmation'])
def test_prepare_v5_development_only(campaign, tmp_path, monkeypatch, split):
    runtime = study.read(tmp_path / 'runtime.json')
    runtime['refinement_strategy'] = V5
    model = tmp_path / 'model.xml'
    model.write_text('<mujoco/>')
    runtime['common_model_xml'] = str(model)
    manifest = study.read(tmp_path / 'manifest.json')
    manifest['tasks'][0]['split'] = split
    manifest['canonical_g1'] = dict(joint_names=[], body_names=[], xml_sha256=study.digest(model))
    study.save_json(tmp_path / 'manifest.json', manifest)
    study.save_json(tmp_path / 'runtime.json', runtime)
    monkeypatch.setattr(study, 'source_identity', lambda runtime: {})
    if split == 'confirmation':
        with pytest.raises(ValueError, match='development-only'):
            study.prepare(tmp_path / 'manifest.json', tmp_path / 'runtime.json', tmp_path / 'v5')
        assert not (tmp_path / 'v5').exists()
    else:
        result = study.prepare(tmp_path / 'manifest.json', tmp_path / 'runtime.json', tmp_path / 'v5')
        assert result['runtime']['refinement_strategy'] == study.V5_STRATEGY == V5
        assert result['budget'] == study.read(campaign)['budget']
        assert all(j['updates'] == 3000 for j in result['jobs'])


def test_v5_requires_canonical_capture(campaign, tmp_path):
    runtime = study.read(tmp_path / 'runtime.json')
    runtime['refinement_strategy'] = V5
    study.save_json(tmp_path / 'runtime.json', runtime)
    with pytest.raises(ValueError, match='canonical'):
        study.prepare(tmp_path / 'manifest.json', tmp_path / 'runtime.json', tmp_path / 'v5')


@pytest.mark.parametrize('invalid', ['confirmation', 'missing_fk', 'unknown_version'])
def test_executor_guard_before_any_evaluation(campaign, monkeypatch, invalid):
    data = study.read(campaign)
    data['runtime']['refinement_strategy'] = V5
    data['reference_contract'] = {'fixture': True}
    if invalid == 'confirmation':
        data['tasks'][0]['split'] = 'confirmation'
    elif invalid == 'missing_fk':
        data['reference_contract'] = None
    else:
        data['runtime']['refinement_strategy'] = 'joint_precision_v6'
    job = next(j for j in data['jobs'] if j['method'] == 'mimicx')
    monkeypatch.setattr(loop, 'evaluate', lambda *a: pytest.fail('Guard must precede evaluation'))
    with pytest.raises(ValueError, match='development|strategy'):
        study.execute_arm_v4(campaign, data, job, campaign.parent / 'guard')


def test_worker_dispatches_v5_to_joint_executor(campaign, monkeypatch):
    data = study.read(campaign)
    data['runtime']['refinement_strategy'] = V5
    study.save_json(campaign, data)
    job = next(j for j in data['jobs'] if j['method'] == 'mimicx')
    calls = []
    monkeypatch.setattr(study, 'verify_campaign', lambda *a: None)
    monkeypatch.setattr(study, 'check_execution_environment', lambda *a: None)
    monkeypatch.setattr(study, 'execute_arm_v4', lambda *a: calls.append(a))
    monkeypatch.setattr(loop, 'execute_arm', lambda *a: pytest.fail('Not legacy executor'))
    study.main(['_worker', '--campaign', str(campaign), '--job', job['id']])
    assert len(calls) == 1


@pytest.mark.parametrize('fault', ['none', 'sacrifice', 'bad_parent', 'updates', 'slots', 'holdout_failure'])
@pytest.mark.parametrize('method', ['mimicx', 'static_search', 'beyond_mimic'])
def test_v5_execution_contracts(campaign, monkeypatch, method, fault):
    data = study.read(campaign)
    data['runtime']['refinement_strategy'] = V5
    data['reference_contract'] = {'fixture': True}
    study.save_json(campaign, data)
    task = data['tasks'][0]
    job = next(j for j in data['jobs'] if j['method'] == method)
    folder = campaign.parent / 'jobs' / job['id']
    warm = folder.parent / 'warmstart/final.pt'
    warm.parent.mkdir(parents=True, exist_ok=True)
    warm.write_bytes(b'protected donor')
    if fault == 'bad_parent' and method != 'beyond_mimic':
        study.save_json(folder / job['stages'][0]['name'] / 'stage.json',
                        dict(parent=str(warm), parent_sha256='wrong'))
    n = task['frames'] - 1
    updates, patches, holdouts, evaluations = [], [], [], []

    def train(actual_campaign, actual_job, output, *, stage=None):
        budget = stage['updates'] if stage else 3000
        updates.append(budget)
        if stage:
            patches.append(stage['patch'])
            expected_parent = warm if method == 'static_search' or len(updates) == 1 or fault == 'sacrifice' else folder / 'stage0/model_499.pt'
            assert stage['parent'] == str(expected_parent)
            assert stage['parent_sha256'] == study.digest(expected_parent)
        output.mkdir(parents=True, exist_ok=True)
        slots = [f'model_{i}.pt' for i in range(499, budget, 500)]
        if fault == 'slots':
            slots = slots[:-1]
        for name in slots + ['final.pt']:
            (output / name).write_bytes(b'candidate')
        study.save_json(output / 'effective_training.json', {'fixture': True})
        summary = dict(completed_updates=budget - (fault == 'updates'), transitions=budget * 2 * 24,
                       selection_checkpoints=slots)
        study.save_json(output / 'training.json', summary)
        return summary

    def evaluate(path, actual_campaign, actual_job, checkpoint, output, seeds, noise):
        evaluations.append(output)
        if 'holdout' in output.parts:
            selected = study.read(folder / 'selection.json')
            assert selected['checkpoint'] == str(checkpoint)
            assert not selected['uses_holdout']
            assert seeds == data['runtime']['validation_protocol']['holdout']['development'][output.name]
            holdouts.append(study.digest(folder / 'selection.json'))
            if fault == 'holdout_failure':
                raise RuntimeError('Independent holdout interrupted')
        else:
            assert seeds == data['runtime']['validation_protocol']['selection_seeds']
        candidate = checkpoint != warm
        metrics = evidence(.04 if candidate else .05,
                           (.03 if fault == 'sacrifice' else .01) if candidate else .02, n)
        for seed, item in zip(seeds, metrics):
            target = output / f'seed{seed}' / 'rollout'
            target.mkdir(parents=True, exist_ok=True)
            np.savez(target / 'common_steps.npz', **{k: v for k, v in item.items() if k != 'pre_failure_frames'})
        study.save_json(output / 'summary.json', dict(common_metrics=[dict(pre_failure_frames=n + 1, complete_motion=True)] * 3))
        return repeats(n=n)

    monkeypatch.setattr(study, 'train_native', train)
    monkeypatch.setattr(loop, 'evaluate', evaluate)
    monkeypatch.setattr(study, 'refinement_v4', lambda *a: pytest.fail('v5 must not use native replay'))
    if fault == 'bad_parent' and method != 'beyond_mimic':
        with pytest.raises(ValueError, match='protected-parent'):
            study.execute_arm_v4(campaign, data, job, folder)
        assert not updates
        return
    if fault in ('updates', 'slots', 'holdout_failure'):
        with pytest.raises(RuntimeError, match='3000 updates|Independent holdout'):
            study.execute_arm_v4(campaign, data, job, folder)
        assert not (folder / 'report.json').exists()
        assert (folder / 'selection.json').exists() == (fault == 'holdout_failure')
        return
    study.execute_arm_v4(campaign, data, job, folder)
    selection = study.read(folder / 'selection.json')
    assert sum(updates) == 3000
    assert selection['criterion'] == V5
    assert len(selection['history']) == selection['slots'] == 6
    assert all(h['gate_version'] == study.V4_STRATEGY for h in selection['history'])
    assert selection['checkpoint'] == str(warm) if fault == 'sacrifice' else selection['checkpoint'] != str(warm)
    if method != 'beyond_mimic':
        assert patches == loop.static_patches(task)
    assert evaluations[0] == folder.parent / 'shared_validation/warmstart'
    assert holdouts == [study.digest(folder / 'selection.json')] * 2
    report = study.read(folder / 'report.json')
    assert report['metrics_contract'] == study.V4_STRATEGY
    assert report['refinement_strategy'] == V5
    assert report['budget_accounting']['continuation_updates'] == 3000
    assert report['budget_accounting']['shared_three_arm_updates'] == 12000
    before = copy.deepcopy(selection)
    updates.clear()
    patches.clear()
    study.execute_arm_v4(campaign, data, job, folder)
    assert study.read(folder / 'selection.json') == before


@pytest.mark.parametrize('mismatch', ['campaign_sha256', 'checkpoint_sha256', 'motion_sha256', 'seed', 'noise', 'horizon'])
def test_shared_warm_validation_rejects_mismatched_contract(campaign, tmp_path, monkeypatch, mismatch):
    data = study.read(campaign)
    job = next(j for j in data['jobs'] if j['method'] == 'warmstart')
    task = data['tasks'][0]
    checkpoint = tmp_path / 'donor.pt'
    checkpoint.write_bytes(b'protected donor')
    output = tmp_path / 'shared_validation/warmstart'
    signature = dict(campaign_sha256=study.digest(campaign), checkpoint_sha256=study.digest(checkpoint),
                     motion_sha256=task['sha256'], seed=1001, noise=.005, horizon=task['frames'] - 1)
    signature[mismatch] = 'old mismatched value'
    study.save_json(output / 'seed1001/contract.json', signature)
    monkeypatch.setattr(loop, 'checked_process', lambda *a: pytest.fail('Must fail before launch'))
    with pytest.raises(ValueError, match='changed'):
        loop.evaluate(campaign, data, job, checkpoint, output, [1001], .005)
