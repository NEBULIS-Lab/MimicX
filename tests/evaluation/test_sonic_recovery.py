"""Read-only recovery classification; synthetic journals, no processes."""
import importlib.util
import json

import pytest


def api():
    assert importlib.util.find_spec('mimicx.evaluation.sonic_recovery') is not None
    from mimicx.evaluation.sonic_recovery import classify_attempt
    return classify_attempt


@pytest.fixture
def attempt(tmp_path):
    identity = dict(task_id='task', condition='nominal', seed=21001, row_id='task__nominal__21001')
    row = dict(identity, row_dir=str(tmp_path/'row'))
    (tmp_path/'row').mkdir()
    terminal = dict(identity, execution_started=False, protocol_valid=False, policy_failure=None,
                    classification='prelaunch_invalid', finished_unix=100, wall_seconds=1,
                    error='ValueError: DDS namespace active/unreadable: ' + json.dumps(
                        dict(schema='mimicx.dds-audit.launcher06', conflicts=['synthetic conflict'])))
    audit = dict(identity, protocol_valid=False, classification='prelaunch_invalid')
    path = tmp_path/'row/terminal.json'
    path.write_text(json.dumps(terminal))
    return tmp_path/'queue.json', row, audit, terminal, path


def test_proven_prenative_conflict(attempt):
    queue, row, audit, _, path = attempt
    before = path.read_bytes()
    result = api()(queue, row, audit)
    assert result['kind'] == 'prelaunch_dds'
    assert result['physical_markers'] == []
    assert path.read_bytes() == before


@pytest.mark.parametrize('marker', ['attempt.json', 'result.json', 'receipt.json', 'sim_state.npz',
    'sim.log', 'deploy.log', 'sim_ready.json',
    'executions/launch.json', 'executions/pid.json', 'executions/runner.log',
    'executions/UNRESOLVED_LIVE_CHILD.json'])
def test_any_execution_evidence_blocks_prelaunch_classification(attempt, marker):
    queue, row, audit, _, _ = attempt
    path = (queue.parent/'executions'/row['row_id']/marker.split('/')[-1] if '/' in marker
            else queue.parent/'row'/marker)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    with pytest.raises(ValueError, match='contradict'):
        api()(queue, row, audit)


@pytest.mark.parametrize('change', [dict(error='other error'), dict(policy_failure=True),
    dict(error='ValueError: DDS namespace active/unreadable: not json'),
    dict(error='ValueError: DDS namespace active/unreadable: []'),
    dict(error='ValueError: DDS namespace active/unreadable: {"schema":"mimicx.dds-audit.launcher06","conflicts":[]}'),
    dict(classification='runtime_protocol_failure')])
def test_unproven_refusal_is_blocked(attempt, change):
    queue, row, audit, terminal, path = attempt
    path.write_text(json.dumps(dict(terminal, **change)))
    assert api()(queue, row, audit)['kind'] == 'blocked_invalid'


def test_physical_policy_failure_remains_physical(attempt):
    queue, row, audit, terminal, path = attempt
    path.write_text(json.dumps(dict(terminal, execution_started=True, protocol_valid=True,
                                    policy_failure=True, classification='valid_policy_failure')))
    (path.parent/'result.json').write_text('{}')
    audit.update(protocol_valid=True, complete_motion=False, strict_qualified=False)
    assert api()(queue, row, audit)['kind'] == 'physical'


def test_runtime_failure_is_not_a_prenative_retry(attempt):
    queue, row, audit, terminal, path = attempt
    path.write_text(json.dumps(dict(terminal, execution_started=True, classification='runtime_protocol_failure')))
    assert api()(queue, row, audit)['kind'] == 'physical'


def test_unstarted_and_unfinished_are_distinct(attempt):
    queue, row, audit, _, path = attempt
    path.unlink()
    audit['classification'] = 'unstarted'
    assert api()(queue, row, audit)['kind'] == 'unstarted'
    (queue.parent/'executions'/row['row_id']).mkdir(parents=True)
    with pytest.raises(ValueError, match='Unresolved'):
        api()(queue, row, audit)


def test_terminal_identity_mismatch_rejected(attempt):
    queue, row, audit, terminal, path = attempt
    path.write_text(json.dumps(dict(terminal, seed=22001)))
    with pytest.raises(ValueError, match='identity'):
        api()(queue, row, audit)


@pytest.mark.parametrize('field,value', [('finished_unix', float('nan')), ('wall_seconds', -1),
    ('execution_started', 0), ('wall_seconds', True)])
def test_malformed_terminal_is_not_retry_evidence(attempt, field, value):
    queue, row, audit, terminal, path = attempt
    path.write_text(json.dumps(dict(terminal, **{field: value})))
    with pytest.raises(ValueError):
        api()(queue, row, audit)


def test_dangling_execution_marker_is_not_absence(attempt):
    queue, row, audit, _, path = attempt
    (path.parent/'attempt.json').symlink_to('missing.json')
    with pytest.raises(ValueError, match='contradict'):
        api()(queue, row, audit)
