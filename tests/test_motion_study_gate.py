"""CPU-only startup selection tests; terminal state is never tracking evidence."""

import copy
import json

import pytest

from mimicx.experiments.motion_study_gate import STARTUP_PRECISION_BOUNDS, accept
from mimicx.policy_recovery.metrics import accept as recovery_accept, qualifies


def rows(horizon=40, failure=None, error=.04):
    return [dict(step=i, done_any=i == failure, robot_anchor_x=error,
                 robot_anchor_y=0., robot_anchor_z=1., ref_anchor_x=0.,
                 ref_anchor_y=0., ref_anchor_z=1., body_pos_error_max=error,
                 error_anchor_rot=error, rew_mean=0.) for i in range(1, horizon + 1)]


def repeats(**kwargs):
    return [rows(**kwargs) for _ in range(3)]


def test_good_full_candidate_recovers_first_step_failure():
    old, new = repeats(failure=1), repeats()
    assert not recovery_accept(old, new, 40)['accepted']
    result = accept(old, new, 40)
    assert result['accepted'] and result['reason'] == 'Accepted'
    assert result['startup_seed_indices'] == [0, 1, 2]
    assert result['common_prefix_old'] == [None] * 3
    assert result['common_prefix_new'] == [None] * 3
    assert all(check['passed'] for check in result['startup_checks'])
    assert result['startup_precision_bounds'] == dict(STARTUP_PRECISION_BOUNDS)
    assert not result['uses_training_reward']
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('failure', [2, 8, 25])
def test_partial_recovery_does_not_require_full_qualification(failure):
    result = accept(repeats(failure=1), repeats(failure=failure), 40)
    assert result['accepted']
    assert all(r['valid_steps'] == failure - 1 for r in result['new'])
    assert not qualifies(result['new'])


def test_mixed_partial_recovery_keeps_shared_prefix_guard():
    old = [rows(failure=1), rows(failure=15), rows(failure=25)]
    new = [rows(failure=8), rows(failure=20), rows(failure=30)]
    result = accept(old, new, 40)
    assert result['accepted']
    assert result['startup_seed_indices'] == [0]
    assert result['common_prefix_old'][0] is None
    assert result['common_prefix_old'][1:] == [(.04, .04, .04)] * 2
    # This error is within startup bounds but violates the stricter relative
    # guard on a seed with an already accurate incumbent.
    for row in new[1]:
        row['robot_anchor_x'] = .07
    result = accept(old, new, 40)
    assert not result['accepted'] and result['reason'] == 'Tracking guard'
    assert result['startup_checks'][0]['passed']


@pytest.mark.parametrize('key,field,value,tail', [
    ('world_anchor_mean', 'robot_anchor_x', .081, False),
    ('world_anchor_p95', 'robot_anchor_x', .151, True),
    ('aligned_body_max_mean', 'body_pos_error_max', .101, False),
    ('aligned_body_max_p95', 'body_pos_error_max', .201, True),
    ('anchor_rotation_mean', 'error_anchor_rot', .201, False),
    ('anchor_rotation_p95', 'error_anchor_rot', .351, True),
])
def test_each_absolute_bound_rejects_precisely(key, field, value, tail):
    new = repeats(error=0.)
    selected = new[1][-10:] if tail else new[1]
    for row in selected:
        row[field] = value
    result = accept(repeats(failure=1), new, 40)
    assert not result['accepted'] and result['reason'] == 'Startup precision guard'
    assert result['startup_checks'][1]['violated_bounds'] == [key]


def test_startup_guard_checks_entire_candidate_prefix_not_only_early_frames():
    new = repeats(failure=30)
    for row in new[0][20:29]:
        row['robot_anchor_x'] = .5
    assert accept(repeats(failure=1), new, 40)['reason'] == 'Startup precision guard'


def test_absolute_mean_limits_are_inclusive():
    new = repeats(error=0.)
    for repeat in new:
        for row in repeat:
            row.update(robot_anchor_x=.08, body_pos_error_max=.10, error_anchor_rot=.20)
    assert accept(repeats(failure=1), new, 40)['accepted']


@pytest.mark.parametrize('startup', [True, False])
def test_any_per_seed_horizon_regression_rejects_despite_aggregate_gain(startup):
    old = [rows(failure=1 if startup else 10), rows(failure=20), rows(failure=30)]
    new = [rows(), rows(), rows(failure=29)]
    result = accept(old, new, 40)
    assert not result['accepted'] and result['reason'] == 'Per-seed horizon regression'
    assert not result['no_seed_frontier_regression']
    assert result['regressed_seed_indices'] == [2]


def test_candidate_zero_evidence_rejected_even_with_other_seed_improvements():
    result = accept(repeats(failure=1), [rows(), rows(failure=1), rows()], 40)
    assert not result['accepted'] and result['reason'] == 'No valid candidate evidence'
    assert result['no_evidence_seed_indices'] == [1]


def test_terminal_and_postreset_rows_cannot_supply_or_destroy_precision_evidence():
    old, new = repeats(failure=1), repeats(failure=8)
    expected = accept(old, new, 40)
    for repeat in old:
        for row in repeat:
            row.update(robot_anchor_x=900., body_pos_error_max=900., error_anchor_rot=900.)
    for repeat in new:
        for row in repeat[7:]:
            row.update(robot_anchor_x=900., body_pos_error_max=900., error_anchor_rot=900.)
    assert accept(old, new, 40) == expected
    # Clean terminal/postreset values cannot rescue a poor valid prefix either.
    for row in new[0][:7]:
        row['robot_anchor_x'] = 1.
    for row in new[0][7:]:
        row.update(robot_anchor_x=0., body_pos_error_max=0., error_anchor_rot=0.)
    assert accept(old, new, 40)['reason'] == 'Startup precision guard'


@pytest.mark.parametrize('old,new', [
    (repeats(), repeats(error=.03)),
    (repeats(), repeats()),
    (repeats(failure=20), repeats(error=.045)),
    (repeats(failure=20), repeats(error=.20)),
])
def test_non_startup_decisions_keep_existing_rule(old, new):
    expected = recovery_accept(old, new, 40)
    result = accept(old, new, 40)
    assert all(result[key] == value for key, value in expected.items())
    assert result['startup_checks'] == []


def test_gate_never_uses_rewards_or_mutates_inputs():
    old, new = repeats(failure=1), repeats(failure=8)
    before = copy.deepcopy((old, new))
    expected = accept(old, new, 40)
    assert (old, new) == before
    for repeat in old + new:
        for row in repeat:
            row['rew_mean'] = 'not even a number'
    assert accept(old, new, 40) == expected
    for repeat in old + new:
        for row in repeat:
            del row['rew_mean']
    assert accept(old, new, 40) == expected


@pytest.mark.parametrize('mutation', ['repeat_count', 'unmatched', 'length', 'step', 'done', 'nan'])
def test_invalid_rollouts_fail_closed(mutation):
    old, new = repeats(failure=1), repeats()
    if mutation == 'repeat_count':
        old.pop()
        new.pop()
    elif mutation == 'unmatched':
        new.pop()
    elif mutation == 'length':
        new[0].pop()
    elif mutation == 'step':
        new[0][1]['step'] = 1
    elif mutation == 'done':
        new[0][0]['done_any'] = 'unknown'
    else:
        new[0][0]['robot_anchor_x'] = float('nan')
    with pytest.raises(ValueError):
        accept(old, new, 40)


def test_bounds_cannot_be_changed_by_a_caller():
    with pytest.raises(TypeError):
        STARTUP_PRECISION_BOUNDS['world_anchor_mean'] = 100.
    result = accept(repeats(failure=1), repeats(), 40)
    result['startup_precision_bounds']['world_anchor_mean'] = 100.
    assert STARTUP_PRECISION_BOUNDS['world_anchor_mean'] == .08
