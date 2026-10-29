"""CPU-only v4 contracts, not physical performance evidence."""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mimicx.experiments import motion_study as study
from mimicx.experiments import motion_study_loop as loop
from tests.test_motion_study_campaign import campaign
from tests.test_policy_recovery_metrics import rows


def repeats(n=40, error=.05, failure=None):
    return [rows(n, error=error, failure=failure) for _ in range(3)]


def evidence(world=.05, local=.02, n=40, frontier=None):
    return [dict(world_body_error_m=np.full(n + 1, world),
                 root_local_fk_body_error_m=np.full(n + 1, local),
                 pre_failure_frames=n + 1 if frontier is None else frontier)
            for _ in range(3)]


def test_saturated_fallback_preserves_native_rewards_and_avoids_duplicate_search(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, 'diagnose', lambda *a: pytest.fail('No failure to diagnose'))
    task = dict(task_id='walk', frames=41)
    before = repeats()
    first = study.refinement_v4({}, task, before, tmp_path / 'stage0', [])
    renamed = {**first, 'id': 'same_configuration'}
    second = study.refinement_v4({}, task, before, tmp_path / 'stage1', [renamed])
    assert first['sampling'] == dict(mode='window', start=0, end=40, ratio=.45)
    assert second['sampling'] == dict(mode='window', start=0, end=40, ratio=.65)
    for patch in (first, second):
        assert patch['reward_overrides'] == {}
        assert all(patch[k] == [] for k in ('body_pos_rewards', 'body_z_rewards',
                                          'body_lin_vel_rewards', 'body_ang_vel_rewards'))
    assert before == repeats()
    with pytest.raises(ValueError, match='untried'):
        study.refinement_v4({}, task, before, tmp_path / 'stage2', [first, second])


def test_unsaturated_refinement_keeps_existing_failure_miner(tmp_path, monkeypatch):
    from mimicx.refinement.proposals import ProposalContext
    context = ProposalContext('walk', 40, 1, 20, (5, 30), (), (), (), ())
    monkeypatch.setattr(loop, 'diagnose', lambda *a: context)
    patch = study.refinement_v4({}, dict(task_id='walk', frames=41), repeats(failure=20), tmp_path, [])
    assert patch['sampling'] == dict(mode='window', start=5, end=30, ratio=.45)
    assert patch['reward_overrides']['motion_global_root_pos']['weight'] == 5.2


@pytest.mark.parametrize('world,local', [(.03, .021), (.051, .01)])
def test_joint_guard_rejects_local_or_global_sacrifice(world, local):
    result = study.accept_v4(repeats(), repeats(error=.03), 40, evidence(), evidence(world, local))
    assert not result['accepted']
    assert result['reason'] == 'Joint local/global guard'


def test_joint_improvement_can_win_when_legacy_proxy_score_is_flat():
    result = study.accept_v4(repeats(), repeats(), 40, evidence(), evidence(.04, .018))
    assert result['accepted']
    assert not loop.accept(repeats(), repeats(), 40)['accepted']


def test_one_bad_seed_or_p95_excursion_cannot_hide_in_average():
    old, new = evidence(), evidence(.03, .01)
    new[2]['root_local_fk_body_error_m'][:] = .021
    assert not study.accept_v4(repeats(), repeats(error=.03), 40, old, new)['accepted']
    new = evidence(.03, .01)
    new[0]['root_local_fk_body_error_m'][1:5] = .03
    assert not study.accept_v4(repeats(), repeats(error=.03), 40, old, new)['accepted']


def test_joint_guard_rejects_physical_frontier_regression():
    result = study.accept_v4(repeats(), repeats(error=.03), 40,
                             evidence(), evidence(.03, .01, frontier=30))
    assert not result['accepted']


def test_common_prefix_ignores_phase_zero_terminal_and_postreset_values():
    old, new = evidence(n=40, frontier=20), evidence(.04, .018, n=40, frontier=30)
    expected = study.accept_v4(repeats(failure=20), repeats(failure=30), 40, old, new)
    assert expected['accepted']
    for group in (old, new):
        for item in group:
            for key in ('world_body_error_m', 'root_local_fk_body_error_m'):
                item[key][0] = 100.
                item[key][20:] = 100.
    assert study.accept_v4(repeats(failure=20), repeats(failure=30), 40, old, new) == expected


@pytest.mark.parametrize('invalid', ['missing', 'nan', 'negative', 'shape'])
def test_joint_evidence_fails_closed(invalid):
    new = evidence(.04, .018)
    if invalid == 'missing':
        new.pop()
    elif invalid == 'nan':
        new[0]['world_body_error_m'][1] = np.nan
    elif invalid == 'negative':
        new[0]['world_body_error_m'][1] = -.1
    else:
        new[0]['world_body_error_m'] = np.zeros(2)
    with pytest.raises(ValueError):
        study.accept_v4(repeats(), repeats(error=.03), 40, evidence(), new)


def test_legacy_default_and_opt_in_validation(campaign, tmp_path):
    original = study.read(campaign)
    assert 'refinement_strategy' not in original['runtime']
    runtime_path = tmp_path / 'runtime.json'
    runtime = study.read(runtime_path)
    runtime['refinement_strategy'] = 'unknown'
    study.save_json(runtime_path, runtime)
    with pytest.raises(ValueError, match='strategy'):
        study.prepare(tmp_path / 'manifest.json', runtime_path, tmp_path / 'bad')
    runtime['refinement_strategy'] = 'joint_neutral_v4'
    study.save_json(runtime_path, runtime)
    with pytest.raises(ValueError, match='canonical'):
        study.prepare(tmp_path / 'manifest.json', runtime_path, tmp_path / 'no-fk')


@pytest.mark.parametrize('method', ['beyond_mimic', 'static_search', 'mimicx'])
@pytest.mark.parametrize('candidate_kind', ['local_sacrifice', 'joint_improvement', 'failed_holdout'])
def test_v4_execution_freezes_selection_before_independent_holdouts(campaign, monkeypatch, method, candidate_kind):
    data = study.read(campaign)
    data['runtime']['refinement_strategy'] = 'joint_neutral_v4'
    data['reference_contract'] = {'fixture': True}
    job = next(j for j in data['jobs'] if j['method'] == method)
    folder = campaign.parent / 'jobs' / job['id']
    warm = folder.parent / 'warmstart/final.pt'
    warm.parent.mkdir(parents=True, exist_ok=True)
    warm.write_bytes(b'protected warmstart')
    n = data['tasks'][0]['frames'] - 1
    updates = []
    frozen_selection = []

    def train(actual_campaign, actual_job, output, *, stage=None):
        budget = stage['updates'] if stage else 3000
        updates.append(budget)
        output.mkdir(parents=True, exist_ok=True)
        slots = [f'model_{i}.pt' for i in range(499, budget, 500)]
        for name in slots:
            (output / name).write_bytes(b'candidate')
        (output / 'final.pt').write_bytes(b'last candidate')
        study.save_json(output / 'effective_training.json', {'test_fixture': True})
        summary = dict(completed_updates=budget, transitions=budget * 2 * 24, selection_checkpoints=slots)
        study.save_json(output / 'training.json', summary)
        return summary

    def evaluate(path, actual_campaign, actual_job, checkpoint, output, seeds, noise):
        if 'holdout' in output.parts:
            selected = study.read(folder / 'selection.json')
            assert selected['checkpoint'] == str(checkpoint)
            frozen_selection.append(study.digest(folder / 'selection.json'))
            assert seeds == ([2001, 2002, 2003] if output.name == 'nominal' else [3001, 3002, 3003])
            if candidate_kind == 'failed_holdout':
                raise RuntimeError('Independent evaluator interrupted')
        else:
            assert seeds == [1001, 1002, 1003]
        candidate = checkpoint != warm
        local = .01 if candidate_kind == 'joint_improvement' else .03
        metrics = evidence(.03 if candidate else .05, local if candidate else .02, n)
        for seed, item in zip(seeds, metrics):
            target = output / f'seed{seed}' / 'rollout'
            target.mkdir(parents=True, exist_ok=True)
            np.savez(target / 'common_steps.npz', **{k: v for k, v in item.items() if k != 'pre_failure_frames'})
        study.save_json(output / 'summary.json', dict(common_metrics=[dict(pre_failure_frames=n + 1,
                                                                         complete_motion=True)] * 3))
        return repeats(n=n, error=.03 if candidate else .05)

    monkeypatch.setattr(study, 'train_native', train)
    monkeypatch.setattr(loop, 'evaluate', evaluate)
    if candidate_kind == 'failed_holdout':
        with pytest.raises(RuntimeError, match='Independent evaluator'):
            study.execute_arm_v4(campaign, data, job, folder)
        assert (folder / 'selection.json').exists()
        assert not (folder / 'report.json').exists()
        assert not (folder / 'done.json').exists()
        return
    study.execute_arm_v4(campaign, data, job, folder)
    selection = study.read(folder / 'selection.json')
    assert sum(updates) == 3000
    assert len(selection['history']) == selection['slots'] == 6
    if candidate_kind == 'joint_improvement':
        assert selection['checkpoint'] != str(warm)
        assert selection['history'][0]['accepted']
        if method == 'mimicx':
            assert study.read(folder / 'stage1/stage.json')['parent'] == selection['checkpoint']
    else:
        assert selection['checkpoint'] == str(warm)
        assert all(not g['accepted'] for g in selection['history'])
    report = study.read(folder / 'report.json')
    assert set(report['holdout']) == set(report['common_physical_metrics']) == {'nominal', 'perturbed'}
    assert report['common_protocol_sha256'] == data['common_protocol_sha256']
    assert report['native_qualified'] and report['common_physical_qualified'] and report['qualified']
    assert len(report['common_selection_metrics']) == 3
    assert frozen_selection == [study.digest(folder / 'selection.json')] * 2
    assert report['budget_accounting']['arm_updates_including_reused_warmstart'] == 6000
    assert report['budget_accounting']['shared_three_arm_updates'] == 12000
    assert study.read(folder / 'training.json')['completed_updates'] == 3000
    assert (folder / 'holdout').exists()
    before = copy.deepcopy(selection)
    study.execute_arm_v4(campaign, data, job, folder)
    assert study.read(folder / 'selection.json') == before
    monkeypatch.setattr(study.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0))
    receipt = study._execute(campaign, data, job)
    assert study.verified_receipt(folder) == receipt
    assert {'report.json', 'selection.json', 'training.json', 'final.pt',
            'effective_training.json'} <= receipt['artifacts'].keys()
    if method != 'beyond_mimic':
        assert f"{job['stages'][0]['name']}/training.json" in receipt['artifacts']
