"""The released continuation gates must ignore terminal/post-reset states."""
import copy

import pytest


def rows(horizon=40, failure=None, error=.04):
    return [dict(step=i, done_any=i == failure, robot_anchor_x=error,
                 robot_anchor_y=0, robot_anchor_z=1, ref_anchor_x=0,
                 ref_anchor_y=0, ref_anchor_z=1, body_pos_error_max=error,
                 error_anchor_rot=error) for i in range(1, horizon + 1)]


def test_terminal_and_post_reset_rows_never_count_as_continuation():
    from mimicx.policy_recovery.metrics import summarize
    data = rows(failure=5)
    for row in data[4:]:
        row['body_pos_error_max'] = 9
    result = summarize(data, 40)
    assert result['valid_steps'] == 4
    assert result['first_failure_step'] == 5
    assert result['aligned_body_max_mean'] == .04
    assert not result['complete']


def test_recovery_selection_uses_common_prefix_and_tracking_guard():
    from mimicx.policy_recovery.metrics import accept
    old = [rows(failure=20) for _ in range(3)]
    good = [rows(error=.045) for _ in range(3)]
    assert accept(old, good, 40)['accepted']
    bad = [rows(error=.20) for _ in range(3)]
    assert accept(old, bad, 40)['reason'] == 'Tracking guard'
    assert not accept(old, bad, 40)['accepted']


def test_holdout_requires_three_complete_quality_repeats():
    from mimicx.policy_recovery.metrics import qualifies, summarize
    good = summarize(rows(), 40)
    assert qualifies([good] * 3)
    assert not qualifies([good] * 2)
    assert not qualifies([good, good, summarize(rows(failure=40), 40)])


@pytest.mark.parametrize('mutation', ['step', 'done', 'nan', 'length'])
def test_invalid_evaluations_are_rejected(mutation):
    from mimicx.policy_recovery.metrics import summarize
    data = copy.deepcopy(rows())
    if mutation == 'step':
        data[1]['step'] = 1
    elif mutation == 'done':
        data[1]['done_any'] = 'unknown'
    elif mutation == 'nan':
        data[1]['error_anchor_rot'] = float('nan')
    else:
        data.pop()
    with pytest.raises(ValueError):
        summarize(data, 40)


def test_failure_window_uses_earliest_failure_and_stays_in_horizon():
    from mimicx.policy_recovery.metrics import failure_window
    assert failure_window([rows(failure=5), rows(failure=20)], 40) == (0, 15)


def test_terrain_gate_protects_every_seed_frontier():
    from mimicx.parkour.metrics import decision, diagnose, practice_windows
    old = [rows(failure=20), rows(failure=30)]
    new = [rows(), rows(failure=29)]
    result = decision(old, new, 40)
    assert not result['accepted']
    assert not result['no_seed_frontier_regression']
    assert not result['uses_training_reward']
    diagnostic = diagnose(old, 40)
    assert diagnostic['frontier'] == 19
    windows = practice_windows('execution_recovery', diagnostic, 40, {})
    assert windows == [[0, 40, .35], [0, 25, .15], [0, 40, .10]]
