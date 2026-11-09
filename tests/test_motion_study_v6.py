"""CPU proposal/selection contracts; no physical-performance claim."""
import copy

import numpy as np
import pytest

from mimicx.experiments import motion_study as study
from mimicx.experiments import motion_study_loop as loop
from tests.test_motion_study_campaign import campaign
from tests.test_motion_study_v4 import evidence, repeats


def history(n=40):
    gates = []
    for slot in range(3):
        gate = study.accept_v4(repeats(n=n), repeats(n=n), n,
                               evidence(n=n), evidence(.03, .025, n))
        gate.update(slot=slot, stage=0, incumbent_sha256='same-donor')
        gates.append(gate)
    return gates


def test_tradeoff_switches_only_second_proposal_without_mutation(tmp_path):
    task = dict(task_id='unseen_motion', frames=41)
    first = loop.static_patches(task)[0]
    tried, gates, rows = [{**first, 'id': 'renamed'}], history(), repeats()
    before = copy.deepcopy((tried, gates, rows))
    assert study.refinement_v6({}, task, rows, tmp_path, [], []) == first
    result = study.refinement_v6({}, task, rows, tmp_path, tried, gates)
    assert result == study.refinement_v4({}, task, rows, tmp_path, tried)
    assert result['id'] == 'native_full_replay_45'
    assert result['reward_overrides'] == {} and result['body_pos_rewards'] == []
    assert (tried, gates, rows) == before
    with pytest.raises(ValueError, match='untried'):
        study.refinement_v6({}, task, rows, tmp_path, loop.static_patches(task), gates * 2)


@pytest.mark.parametrize('fault', ['accepted', 'world_mean', 'world_p95', 'p95_only',
    'native_guard', 'incomplete', 'frontier', 'holdout', 'wrong_stage', 'wrong_slot',
    'missing_repeat', 'duplicate_repeat', 'short_prefix', 'nan', 'negative', 'wrong_gate',
    'missing_slot', 'wrong_first_patch', 'changed_donor'])
def test_other_feedback_keeps_v5(tmp_path, fault):
    task = dict(task_id='walk', frames=41)
    tried, gates = [loop.static_patches(task)[0]], history()
    g = gates[1]
    if fault == 'accepted':
        g['accepted'] = True
    elif fault in ('world_mean', 'world_p95'):
        g['joint_checks'][0]['measured']['world_body_error_m']['new'][fault == 'world_p95'] = .2
    elif fault == 'p95_only':
        for check in g['joint_checks']:
            check['measured']['root_local_fk_body_error_m']['new'][0] = .01
    elif fault == 'native_guard':
        g['reason'] = 'Tracking guard'
    elif fault == 'incomplete':
        g['new'][0]['complete'] = False
    elif fault == 'frontier':
        g['joint_checks'][0]['frontier_guard'] = False
    elif fault == 'holdout':
        g['uses_holdout'] = True
    elif fault == 'wrong_stage':
        g['stage'] = 1
    elif fault == 'wrong_slot':
        g['slot'] = 5
    elif fault == 'missing_repeat':
        g['joint_checks'].pop()
    elif fault == 'duplicate_repeat':
        g['joint_checks'][1]['seed_index'] = 0
    elif fault == 'short_prefix':
        g['joint_checks'][0]['common_valid_steps'] -= 1
    elif fault in ('nan', 'negative'):
        g['joint_checks'][0]['measured']['root_local_fk_body_error_m']['new'][0] = float('nan') if fault=='nan' else -1.
    elif fault == 'wrong_gate':
        g['gate_version'] = 'unknown'
    elif fault == 'missing_slot':
        gates.pop()
    elif fault == 'wrong_first_patch':
        tried[0]['sampling']['ratio'] = .6
    elif fault == 'changed_donor':
        g['incumbent_sha256'] = 'other-donor'
    assert study.refinement_v6({}, task, repeats(), tmp_path, tried, gates) == study.refinement_v5({}, task, repeats(), tmp_path, tried)


def test_incomplete_incumbent_keeps_existing_diagnosis(tmp_path, monkeypatch):
    task = dict(task_id='walk', frames=41)
    sentinel = object()
    calls = []
    monkeypatch.setattr(study, 'refinement_v5', lambda *a: calls.append(a) or sentinel)
    rows = repeats()
    rows[2] = repeats(failure=20)[0]
    tried = [loop.static_patches(task)[0]]
    assert study.refinement_v6({}, task, rows, tmp_path, tried, history()) is sentinel
    assert calls == [({}, task, rows, tmp_path, tried)]


@pytest.mark.parametrize('invalid', ['confirmation', 'canonical', 'unknown'])
def test_executor_rejects_before_evaluation(campaign, monkeypatch, invalid):
    data = study.read(campaign)
    data['runtime']['refinement_strategy'] = study.V6_STRATEGY
    data['reference_contract'] = {'fixture': True}
    if invalid == 'confirmation':
        data['tasks'][0]['split'] = 'confirmation'
    elif invalid == 'canonical':
        data['reference_contract'] = None
    else:
        data['runtime']['refinement_strategy'] = 'unknown'
    monkeypatch.setattr(loop, 'evaluate', lambda *a: pytest.fail('Guard before evaluation'))
    job = next(j for j in data['jobs'] if j['method']=='mimicx')
    with pytest.raises(ValueError, match='development|strategy'):
        study.execute_arm_v4(campaign, data, job, campaign.parent/'rejected')


@pytest.mark.parametrize('method', ['mimicx', 'static_search', 'beyond_mimic'])
@pytest.mark.parametrize('outcome', ['tradeoff', 'accepted', 'holdout_failure', 'bad_budget'])
def test_execution_budget_parent_resume_and_holdout_isolation(campaign, monkeypatch, method, outcome):
    data = study.read(campaign)
    data['runtime']['refinement_strategy'] = study.V6_STRATEGY
    data['reference_contract'] = {'fixture': True}
    job = next(j for j in data['jobs'] if j['method']==method)
    task = data['tasks'][0]
    n = task['frames']-1
    folder = campaign.parent/'jobs'/job['id']
    warm = folder.parent/'warmstart/final.pt'
    warm.parent.mkdir(parents=True, exist_ok=True)
    warm.write_bytes(b'donor')
    stages, held = [], []

    def train(c, j, output, *, stage=None):
        updates = stage['updates'] if stage else 3000
        if stage:
            expected = warm
            if method=='mimicx' and stages and outcome=='accepted':
                expected = folder/'stage0/model_499.pt'
            assert stage['parent'] == str(expected)
            assert stage['parent_sha256'] == study.digest(expected)
            stages.append(stage)
        output.mkdir(parents=True, exist_ok=True)
        slots = [f'model_{i}.pt' for i in range(499, updates, 500)]
        for name in slots:
            (output/name).write_bytes(str(output/name).encode())
        result = dict(completed_updates=updates-(outcome=='bad_budget'),
                      transitions=updates*2*24, selection_checkpoints=slots)
        study.save_json(output/'training.json', result)
        return result

    def evaluate(path, c, j, checkpoint, output, seeds, noise):
        if 'holdout' in output.parts:
            selection = study.read(folder/'selection.json')
            assert selection['checkpoint'] == str(checkpoint)
            assert not selection['uses_holdout']
            assert seeds == data['runtime']['validation_protocol']['holdout']['development'][output.name]
            held.append(study.digest(folder/'selection.json'))
            if outcome=='holdout_failure':
                raise RuntimeError('holdout interrupted')
        else:
            assert seeds == [1001, 1002, 1003]
        is_candidate = checkpoint != warm
        is_precision = 'stage0' in checkpoint.parts or method != 'mimicx'
        sacrifice = outcome!='accepted' and is_precision and is_candidate
        metrics = evidence(.03 if is_candidate else .05,
                           (.025 if sacrifice else .01) if is_candidate else .02, n)
        for seed, item in zip(seeds, metrics):
            target = output/f'seed{seed}'/'rollout'
            target.mkdir(parents=True, exist_ok=True)
            np.savez(target/'common_steps.npz', **{k:v for k,v in item.items() if k!='pre_failure_frames'})
        study.save_json(output/'summary.json', dict(common_metrics=[dict(pre_failure_frames=n+1, complete_motion=True)]*3))
        return repeats(n=n)

    monkeypatch.setattr(study, 'train_native', train)
    monkeypatch.setattr(loop, 'evaluate', evaluate)
    if outcome in ('bad_budget', 'holdout_failure'):
        with pytest.raises(RuntimeError, match='3000 updates|holdout interrupted'):
            study.execute_arm_v4(campaign, data, job, folder)
        assert (folder/'selection.json').exists() == (outcome=='holdout_failure')
        assert not (folder/'report.json').exists()
        return
    study.execute_arm_v4(campaign, data, job, folder)
    selected = study.read(folder/'selection.json')
    assert selected['criterion'] == study.V6_STRATEGY
    assert selected['slots'] == len(selected['history']) == 6
    assert all(g['gate_version']==study.V4_STRATEGY for g in selected['history'])
    assert held == [study.digest(folder/'selection.json')]*2
    if method != 'beyond_mimic':
        assert sum(s['updates'] for s in stages) == 3000
        assert stages[0]['patch'] == loop.static_patches(task)[0]
        expected = (study.refinement_v4({}, task, repeats(n=n), folder, [stages[0]['patch']])
                    if method=='mimicx' and outcome=='tradeoff' else loop.static_patches(task)[1])
        assert stages[1]['patch'] == expected
    report = study.read(folder/'report.json')
    assert report['refinement_strategy'] == study.V6_STRATEGY
    assert report['metrics_contract'] == study.V4_STRATEGY
    assert report['completed_updates'] == report['budget_accounting']['continuation_updates'] == 3000
    stages.clear()
    held.clear()
    study.execute_arm_v4(campaign, data, job, folder)
    assert study.read(folder/'selection.json') == selected


@pytest.mark.parametrize('split', ['development', 'confirmation'])
def test_prepare_is_development_only(campaign, tmp_path, monkeypatch, split):
    runtime = study.read(tmp_path/'runtime.json')
    runtime['refinement_strategy'] = study.V6_STRATEGY
    model = tmp_path/'model.xml'
    model.write_text('<mujoco/>')
    runtime['common_model_xml'] = str(model)
    manifest = study.read(tmp_path/'manifest.json')
    manifest['tasks'][0]['split'] = split
    manifest['canonical_g1'] = dict(joint_names=[], body_names=[], xml_sha256=study.digest(model))
    study.save_json(tmp_path/'manifest.json', manifest)
    study.save_json(tmp_path/'runtime.json', runtime)
    monkeypatch.setattr(study, 'source_identity', lambda *a: {})
    if split=='confirmation':
        with pytest.raises(ValueError, match='development-only'):
            study.prepare(tmp_path/'manifest.json', tmp_path/'runtime.json', tmp_path/'v6')
        assert not (tmp_path/'v6').exists()
    else:
        result = study.prepare(tmp_path/'manifest.json', tmp_path/'runtime.json', tmp_path/'v6')
        assert result['budget'] == study.read(campaign)['budget']


def test_worker_dispatches_v6(campaign, monkeypatch):
    data = study.read(campaign)
    data['runtime']['refinement_strategy'] = study.V6_STRATEGY
    study.save_json(campaign, data)
    job = next(j for j in data['jobs'] if j['method']=='mimicx')
    calls = []
    monkeypatch.setattr(study, 'verify_campaign', lambda *a: None)
    monkeypatch.setattr(study, 'check_execution_environment', lambda *a: None)
    monkeypatch.setattr(study, 'execute_arm_v4', lambda *a: calls.append(a))
    monkeypatch.setattr(loop, 'execute_arm', lambda *a: pytest.fail('Legacy dispatcher'))
    study.main(['_worker', '--campaign', str(campaign), '--job', job['id']])
    assert len(calls)==1
