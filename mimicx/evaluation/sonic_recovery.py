"""Read-only classification of archived formal SONIC attempt evidence.

This ports the delivery chain's pre-native/physical distinction, not its queue
ownership, retry authorization, process checks or machine-specific launchers.
Callers must supply an independently audited row; this does not audit a result.
"""
import json
import math
from pathlib import Path


def classify_attempt(queue_path, declaration, audited):
    """Return physical, prelaunch_dds, blocked_invalid or unstarted; never retry.

    Supports the formal queue's row/terminal/executions layout. Any observed
    execution marker prevents a prelaunch classification, including a failed
    physical outcome or incomplete journal. Contradictory evidence raises.
    """
    def require(ok, message):
        if not ok:
            raise ValueError(message)

    def identity(row):
        require(type(row.get('seed')) is int and all(isinstance(row.get(k), str) and row[k]
                for k in ('task_id', 'condition', 'row_id')), 'Invalid row identity')
        key = f"{row['task_id']}__{row['condition']}__{row['seed']}"
        require(row['row_id'] == key and '/' not in key and '\\' not in key, 'Row identity mismatch')
        return key

    def present(path):
        # A dangling link is contradictory evidence, not proof of non-execution.
        return path.exists() or path.is_symlink()

    key = identity(declaration)
    require(identity(audited) == key and type(audited.get('protocol_valid')) is bool,
            'Audited identity/validity mismatch')
    root = Path(queue_path).resolve().parent
    base = Path(declaration['row_dir']).resolve()
    require(base.is_relative_to(root) and base != root, 'Row artifacts escape generation')
    execution = root / 'executions' / key
    require(execution.resolve().is_relative_to(root), 'Execution artifacts escape generation')
    terminal_path = base / 'terminal.json'
    terminal = None
    if present(terminal_path):
        require(terminal_path.is_file(), 'Unreadable terminal evidence')
        terminal = json.loads(terminal_path.read_text(encoding='utf-8'))
        require(isinstance(terminal, dict) and identity(terminal) == key
                and type(terminal.get('execution_started')) is bool, 'Terminal identity/execution flag mismatch')
    markers = [str(p) for p in [base / n for n in ('attempt.json', 'result.json', 'receipt.json', 'sim_state.npz',
                                                 'sim.log', 'deploy.log', 'sim_ready.json')]
               + [execution / n for n in ('launch.json', 'pid.json', 'runner.log', 'UNRESOLVED_LIVE_CHILD.json')]
               if present(p)]
    span = None
    if terminal is not None:
        finish, duration = terminal.get('finished_unix'), terminal.get('wall_seconds')
        require(type(finish) in (int, float) and math.isfinite(finish)
                and type(duration) in (int, float) and math.isfinite(duration) and 0 <= duration <= finish,
                'Invalid terminal chronology')
        span = (finish - duration, finish)
    result = dict(physical_markers=markers, span=span, terminal=terminal)
    if markers or (terminal is not None and terminal['execution_started']):
        require(terminal is not None, 'Unfinished physical execution')
        require(terminal['execution_started'] is True, 'Physical artifacts contradict prelaunch declaration')
        if audited['protocol_valid']:
            require((base / 'result.json').is_file() and terminal.get('protocol_valid') is True,
                    'Valid audit lacks completed physical result')
            require(type(audited.get('complete_motion')) is bool and type(audited.get('strict_qualified')) is bool
                    and (not audited['strict_qualified'] or audited['complete_motion']), 'Invalid physical outcome flags')
            require(terminal.get('policy_failure') is (not audited['complete_motion']), 'Policy failure mismatch')
        return dict(result, kind='physical')
    require(audited['protocol_valid'] is False, 'Valid audit without physical execution evidence')
    if terminal is None:
        require(not present(execution), 'Unresolved execution directory without terminal')
        require(audited.get('classification') == 'unstarted', 'Missing terminal for invalid row')
        return dict(result, kind='unstarted')
    prefix = 'ValueError: DDS namespace active/unreadable: '
    error = terminal.get('error')
    if (terminal.get('classification') == audited.get('classification') == 'prelaunch_invalid'
            and terminal.get('protocol_valid') is False and terminal.get('policy_failure') is None
            and isinstance(error, str) and error.startswith(prefix)):
        try:
            conflict = json.loads(error[len(prefix):])
        except json.JSONDecodeError:
            conflict = None
        if (isinstance(conflict, dict) and conflict.get('schema') == 'mimicx.dds-audit.launcher06'
                and isinstance(conflict.get('conflicts'), list) and conflict['conflicts']):
            return dict(result, kind='prelaunch_dds')
    return dict(result, kind='blocked_invalid')
