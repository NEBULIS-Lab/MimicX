"""CPU-only, fail-closed confirmation result collection. Inputs are never written."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys

METHODS = ('beyond_mimic', 'static_search', 'mimicx')
TRAIN_SEEDS = (101, 202)
SELECTION = [1001, 1002, 1003]
HOLDOUT = {'nominal': [21001, 21002, 21003], 'perturbed': [22001, 22002, 22003]}
METRICS = {f'{space}_{stat}_cm': (field, stat)
           for space, field in (('world', 'world_body_error_m'),
                                ('local', 'root_local_fk_body_error_m'))
           for stat in ('mean', 'p95')}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    def reject(value):
        raise ValueError(f'Nonfinite JSON constant: {value}')
    return json.loads(Path(path).read_text(), parse_constant=reject)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value):
    return type(value) in (float, int) and math.isfinite(value) and value >= 0


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def native_modules(snapshot):
    snapshot = Path(snapshot).resolve()
    before, bytecode = list(sys.path), sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(snapshot))
        modules = [importlib.import_module('mimicx.' + name) for name in (
            'experiments.motion_study', 'policy_recovery.metrics', 'experiments.motion_study_loop')]
    finally:
        sys.path[:] = before
        sys.dont_write_bytecode = bytecode
    for module in modules:
        require(Path(module.__file__).resolve().is_relative_to(snapshot), 'Wrong native snapshot imported')
    return modules


def issue(scope, key, state, reason, path=None):
    return dict(scope=scope, key=key, state=state, reason=str(reason), path=str(path) if path else None)


def validate_campaign(campaign, expected_strategy):
    tasks, jobs, runtime = campaign['tasks'], campaign['jobs'], campaign['runtime']
    ids = [t['task_id'] for t in tasks]
    require(len(ids) == len(set(ids)) == 12, 'Expected exactly 12 unique confirmation tasks')
    require(all(t['split'] == 'confirmation' and t['training_eligible'] is True for t in tasks),
            'All tasks must be eligible confirmation originals')
    require(expected_strategy not in ('legacy_v3', 'joint_neutral_v4', 'joint_precision_v5',
                                      'joint_adaptive_precision_v6'), 'Development strategy is not confirmation')
    require(runtime['refinement_strategy'] == expected_strategy, 'Frozen strategy mismatch')
    require(campaign.get('reference_contract'), 'Canonical reference contract required')
    reference = campaign['reference_contract']
    require(len(set(reference['body_names'])) == len(reference['body_names']) == 30 and
            len(set(reference['joint_names'])) == len(reference['joint_names']) == 29,
            'Require full-body 30-body/29-joint reference')
    protocol = runtime['validation_protocol']
    require(protocol['selection_seeds'] == SELECTION and protocol['holdout']['confirmation'] == HOLDOUT,
            'Confirmation/selection seeds differ from fixed contract')
    require(all(protocol[k] == v for k, v in dict(selection_noise=.005, nominal_noise=.005,
                                                  perturbed_noise=.01).items()), 'Noise contract changed')
    digest = hashlib.sha256(json.dumps(runtime['common_protocol'], sort_keys=True,
                                      separators=(',', ':')).encode()).hexdigest()
    require(digest == campaign['common_protocol_sha256'], 'Common protocol hash mismatch')
    require(type(runtime['num_envs']) is int and runtime['num_envs'] > 0 and
            runtime['steps_per_env'] == 24, 'Invalid training interaction budget')
    expected = set(itertools.product(ids, TRAIN_SEEDS, ('warmstart',) + METHODS))
    actual = [(j['task_id'], j['seed'], j['method']) for j in jobs]
    require(len(actual) == len(set(actual)) == 96 and set(actual) == expected,
            'Expected exact 72-arm + 24-donor matrix; no extra/missing/duplicate jobs')
    require(len({j['id'] for j in jobs}) == 96, 'Duplicate job IDs')
    lookup = {j['id']: j for j in jobs}
    for job in jobs:
        path = Path(job['id'])
        require(not path.is_absolute() and '..' not in path.parts, 'Unsafe job path')
        require(job['backend'] == 'mjlab' and job['updates'] == 3000 and job['executable'] is True,
                'Job backend/budget/eligibility mismatch')
        if job['method'] == 'warmstart':
            require(job['dependencies'] == [], 'Donor cannot depend on another job')
        else:
            require(job['selection_slots'] == 6 and len(job['dependencies']) == 1, 'Arm requires six slots and one donor')
            donor = lookup[job['dependencies'][0]]
            require((donor['task_id'], donor['seed'], donor['method']) ==
                    (job['task_id'], job['seed'], 'warmstart'), 'Mismatched donor dependency')
            expected_updates = [3000] if job['method'] == 'beyond_mimic' else [1500, 1500]
            require([s['updates'] for s in job['stages']] == expected_updates, 'Stage budgets changed')
    return {t['task_id']: t for t in tasks}


class Collector:
    def __init__(self, campaign_path, study, metrics, loop):
        self.path = Path(campaign_path).resolve()
        self.campaign_hash = sha(self.path)
        self.campaign = read(self.path)
        self.root = self.path.parent / 'jobs'
        self.study, self.metrics, self.loop = study, metrics, loop
        self.group_cache = {}
        self.tpu = self.campaign['runtime']['num_envs'] * self.campaign['runtime']['steps_per_env']

    def receipt(self, folder, required):
        before = read(folder / 'done.json')
        names = before['artifacts']
        require(set(required).issubset(names), f'Receipt missing required artifacts: {folder}')
        require(all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names),
                f'Unsafe receipt artifact path: {folder}')
        receipt = self.study.verified_receipt(folder)
        require(receipt == before, 'Receipt changed during verification')
        return receipt

    def common(self, rollout, task):
        import numpy as np
        common = read(rollout / 'common_metrics.json')
        frames = task['frames']
        require(common['frames'] == frames and type(common['complete_motion']) is bool,
                'Common frame/physical completion mismatch')
        require(common['scoring_body_names'] == self.campaign['reference_contract']['body_names'] and
                common['scoring_body_count'] == 30, 'Incomplete/mismatched scoring bodies')
        valid = common['valid_physical_frames']
        require(type(valid) is int and 0 <= valid <= frames, 'Invalid physical frame count')
        with np.load(rollout / 'native_actual.npz', allow_pickle=False) as actual:
            widths = dict(joint_pos=(29,), joint_vel=(29,), body_pos=(30, 3), body_quat=(30, 4),
                          body_lin_vel=(30, 3), body_ang_vel=(30, 3))
            for key, width in widths.items():
                require(actual[key].shape == (frames, *width) and np.isfinite(actual[key]).all(),
                        f'Invalid full-body numerical array: {key}')
            require(np.allclose(np.linalg.norm(actual['body_quat'], axis=-1), 1., atol=1e-3, rtol=0),
                    'Invalid full-body rotation quaternions')
            mask = actual['valid_mask']
            require(mask.shape == (frames,) and np.isin(mask, [0, 1]).all() and
                    not (np.diff(mask.astype(int)) > 0).any() and int(mask.sum()) == valid,
                    'Invalid/censored mask disagreement')
            require(np.allclose(actual['timestamps'], np.arange(frames) / 50., atol=1e-7, rtol=0),
                    'Common capture clock mismatch')
        full = common['full_horizon']
        require((full is None) == (valid < frames), 'Full-horizon censoring mismatch')
        require(not common['complete_motion'] or valid == frames, 'Censored motion cannot complete')
        result = dict(complete_motion=common['complete_motion'],
                      metric_unavailable_reason='postreset_full_horizon_censored' if full is None else None,
                      full_horizon=full, common_prefix=common.get('common_prefix'), pre_failure=common.get('pre_failure'))
        with np.load(rollout / 'common_steps.npz', allow_pickle=False) as steps:
            for key in steps.files:
                require(steps[key].shape == (frames,) and np.isfinite(steps[key]).all(),
                        f'Invalid common numerical trace: {key}')
            bad = steps['failure_predicate']
            require(np.isin(bad, [0, 1]).all(), 'Invalid physical failure predicate')
            persistence = self.campaign['runtime']['common_protocol']['persistence_frames']
            failures = np.flatnonzero(np.convolve(bad[:valid].astype(int), np.ones(persistence, dtype=int),
                                                  mode='valid') == persistence) if persistence <= valid else []
            native_failure = common['native_failure_frame']
            require(native_failure is None or (type(native_failure) is int and 0 <= native_failure < frames),
                    'Invalid native failure frame')
            first = min(int(failures[0]) if len(failures) else frames, valid,
                        frames if native_failure is None else native_failure)
            require(type(common['startup_failure']) is bool, 'Invalid startup failure flag')
            first = 0 if common['startup_failure'] else first
            require(common['complete_motion'] == (first == frames) and
                    common['first_failure_frame'] == (None if first == frames else first),
                    'Physical completion disagrees with full trace')
            for key, (field, stat) in METRICS.items():
                values = steps[field]
                require((values >= 0).all(), 'Negative common error trace')
                if full is None:
                    result[key] = None
                else:
                    value = full[field][stat]
                    actual = float(values.mean() if stat == 'mean' else np.quantile(values, .95))
                    require(finite(value) and math.isclose(value, actual, rel_tol=1e-9, abs_tol=1e-12),
                            f'Common trace/summary mismatch: {key}')
                    result[key] = 100 * value
        return common, result

    def group(self, folder, task, checkpoint, seeds, noise):
        cache_key = (str(folder), str(checkpoint), tuple(seeds), noise)
        if cache_key in self.group_cache:
            return self.group_cache[cache_key]
        summary = read(folder / 'summary.json')
        checkpoint_hash = sha(checkpoint)
        n = task['frames'] - 1
        require(summary['seeds'] == seeds and summary['horizon'] == n and summary['noise'] == noise and
                Path(summary['checkpoint']).resolve() == checkpoint.resolve() and summary['sha256'] == checkpoint_hash,
                f'Group identity mismatch: {folder}')
        natives, commons, rows = [], [], []
        for seed in seeds:
            run = folder / f'seed{seed}'
            rollout = run / 'rollout'
            receipt = self.receipt(run, ('rollout/steps.csv', 'rollout/common_metrics.json',
                                        'rollout/native_actual.npz', 'rollout/common_steps.npz',
                                        'rollout/replay/manifest.json', 'rollout/metrics.json'))
            signature = dict(campaign_sha256=self.campaign_hash, checkpoint_sha256=checkpoint_hash,
                             motion_sha256=task['sha256'], seed=seed, noise=noise, horizon=n)
            require(receipt['signature'] == signature and receipt['status'] == 'evaluated', 'Evaluation receipt identity mismatch')
            with (rollout / 'steps.csv').open(newline='') as stream:
                native = self.metrics.summarize(list(csv.DictReader(stream)), n)
            require(read(rollout / 'metrics.json')['steps'] == n, 'Native metrics horizon mismatch')
            self.loop.verify_replay(rollout / 'replay', n)
            common, row = self.common(rollout, task)
            require(common['native_failure_frame'] == native['first_failure_step'] and
                    common['valid_physical_frames'] == (task['frames'] if native['complete'] else native['valid_steps'] + 1),
                    'Native reset and common capture censoring disagree')
            natives.append(native)
            commons.append(common)
            rows.append(dict(seed=seed, native_complete=native['complete'], **row))
        require(summary['results'] == natives and summary['common_metrics'] == commons,
                f'Group/rollout summary mismatch: {folder}')
        require(type(summary['qualified']) is bool and summary['qualified'] == self.metrics.qualifies(natives),
                'Group native qualification mismatch')
        result = dict(native=natives, common=commons, rows=rows, qualified=summary['qualified'])
        self.group_cache[cache_key] = result
        return result

    def job(self, job, task):
        folder = self.root / job['id']
        required = ['training.json', 'effective_training.json', 'final.pt']
        if job['method'] != 'warmstart':
            required += ['report.json', 'selection.json']
        receipt = self.receipt(folder, required)
        require(receipt['signature'] == dict(campaign_sha256=self.campaign_hash, job=job), 'Job receipt identity mismatch')
        allowed_status = ('completed', 'warmstart_adopted', 'training_complete_selection_pending') if job['method'] == 'warmstart' else ('completed',)
        require(receipt['status'] in allowed_status, 'Nonterminal receipt status')
        training = read(folder / 'training.json')
        require(receipt['training'] == training, 'Receipt/training mismatch')
        require(training['completed_updates'] == 3000 and training['transitions'] == 3000 * self.tpu,
                'Incomplete/incorrect training budget')
        row = dict(job=job['id'], task_id=job['task_id'], train_seed=job['seed'], method=job['method'],
                   state='validated', completed_updates=3000, completed_transitions=3000 * self.tpu,
                   attempted_transitions_exact=training.get('attempted_transitions_exact'),
                   reused=receipt['status'] == 'warmstart_adopted')
        if row['attempted_transitions_exact'] is not None:
            require(type(row['attempted_transitions_exact']) is int and
                    row['attempted_transitions_exact'] >= row['completed_transitions'], 'Invalid attempted transitions')
        if job['method'] == 'warmstart':
            row.update(candidate_checkpoints=0, candidate_rollouts=0, holdout_rollouts=0)
            return row
        require(receipt['status'] == 'completed', 'Arm cannot be adopted donor')
        report, selection = read(folder / 'report.json'), read(folder / 'selection.json')
        require(selection['uses_holdout'] is False and selection['seeds'] == SELECTION and selection['slots'] == 6,
                'Selection used holdout or wrong opportunity count')
        require(selection['criterion'] == self.campaign['runtime']['refinement_strategy'], 'Selection strategy mismatch')
        require(report['common_protocol'] == self.campaign['runtime']['common_protocol'] and
                report['common_protocol_sha256'] == self.campaign['common_protocol_sha256'], 'Report protocol mismatch')
        slots = training['selection_checkpoints']
        require(len(slots) == len(set(slots)) == 6, 'Expected six distinct candidate checkpoints')
        require(all(not Path(s).is_absolute() and '..' not in Path(s).parts for s in slots), 'Unsafe candidate paths')
        require(set(slots).issubset(receipt['artifacts']), 'Unsealed candidate checkpoints')
        for record in (training, report):
            require(record['completed_updates'] == 3000 and record['transitions'] == 3000 * self.tpu and
                    [s['updates'] for s in record['stages']] == [s['updates'] for s in job['stages']] and
                    all(s['transitions'] == s['updates'] * self.tpu for s in record['stages']), 'Stage training budget mismatch')
            expected_budget = dict(shared_warmstart_updates=3000, continuation_updates=3000,
                                   arm_updates_including_reused_warmstart=6000,
                                   reused_warmstart_training_cost_not_zero=True,
                                   candidate_selection_rollouts=18, shared_warmstart_selection_rollouts=3,
                                   independent_holdout_rollouts=6)
            require(all(record['budget_accounting'].get(k) == v for k, v in expected_budget.items()), 'Budget accounting mismatch')
        donor = self.root / job['dependencies'][0] / 'final.pt'
        warm = self.group(folder.parent / 'shared_validation/warmstart', task, donor, SELECTION, .005)
        groups = [self.group(folder / 'validation' / f'slot{i:02d}', task, folder / s, SELECTION, .005)
                  for i, s in enumerate(slots)]
        selected = Path(selection['checkpoint']).resolve()
        candidates = [donor.resolve()] + [(folder / s).resolve() for s in slots]
        require(selected in candidates, 'Selected checkpoint is not donor or declared candidate')
        selected_group = ([warm] + groups)[candidates.index(selected)]
        require(selection['sha256'] == sha(selected) == sha(folder / 'final.pt'), 'Selected/final checkpoint mismatch')
        for record in (training, report):
            require(Path(record['selected_checkpoint']).resolve() == selected and
                    record['selected_sha256'] == selection['sha256'], 'Selected checkpoint report mismatch')
        retained = selected == donor.resolve()
        require(type(report['selected_warmstart']) is bool and report['selected_warmstart'] == retained,
                'Donor retention mismatch')
        require(report['selection'] == selected_group['native'] and
                report['common_selection_metrics'] == selected_group['common'], 'Selected metrics mismatch')
        # Replay the saved selection path, never rank candidates using confirmation outcomes.
        history = selection['history']
        require(len(history) == 6, 'Incomplete selection history')
        require({f'gates/slot{i:02d}.json' for i in range(6)}.issubset(receipt['artifacts']), 'Unsealed selection history')
        incumbent = donor.resolve()
        for i, gate in enumerate(history):
            require(gate == read(folder / 'gates' / f'slot{i:02d}.json'), 'Gate/history mismatch')
            require(gate['slot'] == i and Path(gate['candidate']).resolve() == candidates[i + 1] and
                    gate['candidate_sha256'] == sha(candidates[i + 1]) and
                    Path(gate['incumbent']).resolve() == incumbent and gate['incumbent_sha256'] == sha(incumbent) and
                    type(gate['accepted']) is bool, 'Selection history identity mismatch')
            if gate['accepted']:
                incumbent = candidates[i + 1]
        require(incumbent == selected, 'Saved gates do not produce selected checkpoint')
        conditions = {}
        all_groups = [selected_group]
        for kind, seeds in HOLDOUT.items():
            group = self.group(folder / 'holdout' / kind, task, selected, seeds,
                               self.campaign['runtime']['validation_protocol'][kind + '_noise'])
            require(report['holdout'][kind] == group['native'] and
                    report['common_physical_metrics'][kind] == group['common'], 'Holdout report mismatch')
            conditions[kind] = summarize_rows(group['rows'])
            all_groups.append(group)
        native = all(self.metrics.qualifies(g['native']) for g in all_groups)
        physical = all(r['complete_motion'] for g in all_groups for r in g['rows'])
        for key, value in dict(native_qualified=native, common_physical_qualified=physical,
                               qualified=native and physical).items():
            require(type(report[key]) is bool and report[key] == value, f'Incorrect report {key}')
        require(type(training['qualified']) is bool and training['qualified'] == (native and physical), 'Training qualification mismatch')
        row.update(qualified=native and physical, native_qualified=native, common_physical_qualified=physical,
                   retained_donor=retained, selected_checkpoint=str(selected), selected_sha256=selection['sha256'],
                   candidate_checkpoints=6, candidate_rollouts=18, holdout_rollouts=6, conditions=conditions)
        return row


def strict_mean(values):
    values = list(values)
    return mean(values) if values and all(v is not None for v in values) else None


def summarize_rows(rows):
    return dict(rollouts=rows, count=len(rows), complete=sum(r['complete_motion'] for r in rows),
                metric_defined_counts={k: sum(r[k] is not None for r in rows) for k in METRICS},
                **{k: strict_mean(r[k] for r in rows) for k in METRICS})


def gain(value, control):
    if value is None or control is None:
        return dict(reduction_cm=None, percent_reduction=None, undefined_reason='censored_metric')
    return dict(reduction_cm=control - value,
                percent_reduction=100 * (1 - value / control) if control else None,
                undefined_reason=None if control else 'zero_control_denominator')


def paired_rows(tasks, arms):
    lookup = {(r['task_id'], r['train_seed'], r['method']): r for r in arms}
    rows = []
    for task, seed, (control, method), kind in itertools.product(
            sorted(tasks), TRAIN_SEEDS, itertools.combinations(METHODS, 2), HOLDOUT):
        a, b = lookup.get((task, seed, control)), lookup.get((task, seed, method))
        ready = a is not None and b is not None
        row = dict(task_id=task, train_seed=seed, control=control, method=method, condition=kind,
                   state='validated' if ready else 'unavailable', qualified_change=None, physical_complete_change=None)
        if ready:
            row.update(qualified_change=int(b['qualified']) - int(a['qualified']),
                       physical_complete_change=b['conditions'][kind]['complete'] - a['conditions'][kind]['complete'])
        for key in METRICS:
            values = gain(b['conditions'][kind][key], a['conditions'][kind][key]) if ready else dict(
                reduction_cm=None, percent_reduction=None, undefined_reason='missing_or_invalid_arm')
            row.update({f'{key}_{name}': value for name, value in values.items()})
        rows.append(row)
    return rows


def timing(jobs_root, jobs):
    """Use only top-level worker attempts; never add nested stage/evaluation timers."""
    intervals, gpu_intervals, errors, coverage = [], {}, [], set()
    records = []
    for job in jobs:
        for path in sorted((jobs_root / job['id'] / 'attempts').glob('*.json')):
            try:
                record = read(path)
                start, end = record['started'], record.get('finished')
                if end is None:
                    continue
                require(finite(start) and finite(end) and end >= start, 'Invalid attempt timestamps')
                require(record.get('job_id', job['id']) == job['id'], 'Attempt job identity mismatch')
                gpu = record.get('gpu_uuid')
                assigned = record.get('assigned_gpu_wall_seconds')
                if assigned is not None:
                    require(finite(assigned), 'Invalid assigned GPU wall time')
                if assigned and gpu:
                    require(str(gpu).startswith('GPU-') and ',' not in str(gpu), 'Timing requires one explicit GPU UUID')
                    gpu_intervals.setdefault(gpu, []).append((start, end))
                intervals.append((start, end))
                coverage.add(job['id'])
                records.append(dict(path=str(path), job=job['id'], started=start, finished=end,
                                    status=record.get('status'), gpu_uuid=gpu, assigned_gpu_wall_seconds=assigned))
            except (ValueError, KeyError, TypeError, OSError) as exc:
                errors.append(issue('timing', job['id'], 'invalid', exc, path))

    def union(spans):
        merged = []
        for start, end in sorted(spans):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return sum(b - a for a, b in merged)

    return dict(source='jobs/<id>/attempts/*.json only; nested timers excluded',
                coverage_jobs=len(coverage), expected_jobs=len(jobs), measured_attempts=len(records),
                observed_elapsed_span_seconds=max(b for a, b in intervals) - min(a for a, b in intervals) if intervals else None,
                observed_busy_wall_union_seconds=union(intervals) if intervals else None,
                observed_assigned_gpu_union_seconds=sum(union(v) for v in gpu_intervals.values()) if gpu_intervals else None,
                measured_gpu_compute_seconds=None, records=records, invalid_records=errors,
                note='Observed attempts only, including retries; elapsed span includes gaps. Not a sum of nested timers. '
                     'Adopted donor historical time and missing timers remain unknown.')


def sonic_rows(input_path, adapter_path, campaign):
    expected = {(t['task_id'], kind, seed) for t in campaign['tasks'] for kind, seeds in HOLDOUT.items() for seed in seeds}
    problems, valid, raw = [], {}, {}
    provenance = None
    if input_path is not None and adapter_path is not None:
        try:
            adapter_hash = sha(adapter_path)
            package = module_at('confirmation_sonic_adapter', adapter_path).collect_rows(Path(input_path), campaign)
            require(sha(adapter_path) == adapter_hash, 'SONIC adapter changed during collection')
            provenance = dict(path=str(adapter_path), sha256=adapter_hash, input=str(input_path))
            raw = package['raw_counts']
            adapter_invalid = set()
            for i, item in enumerate(package.get('invalid_rows', [])):
                if isinstance(item, dict) and all(k in item for k in ('task_id', 'condition', 'seed')):
                    key = (item['task_id'], item['condition'], item['seed'])
                    adapter_invalid.add(key)
                    classification = item.get('classification', 'adapter_rejected')
                    state = 'missing' if classification in ('unstarted', 'missing') else 'invalid'
                    problems.append(issue('sonic', '/'.join(map(str, key)), state,
                                          classification + ': ' + str(item.get('error', 'No valid formal receipt'))))
                else:
                    problems.append(issue('sonic_adapter', str(i), 'invalid', item))
            groups = {}
            for i, row in enumerate(package['rows']):
                try:
                    key = (row['task_id'], row['condition'], row['seed'])
                    require(type(row['seed']) is int and key in expected, 'Unexpected SONIC formal identity')
                    groups.setdefault(key, []).append(row)
                except (ValueError, KeyError, TypeError) as exc:
                    problems.append(issue('sonic', str(i), 'invalid', exc))
            for key, rows in groups.items():
                try:
                    require(len(rows) == 1, 'Duplicate SONIC identity; no winner chosen')
                    require(key not in adapter_invalid, 'SONIC identity reported both valid and invalid')
                    row = rows[0]
                    require(row.get('protocol_valid', True) is True, 'SONIC protocol-invalid row')
                    require(type(row['complete_motion']) is bool and type(row['strict_qualified']) is bool,
                            'SONIC physical/qualification flags must be boolean')
                    require(not row['strict_qualified'] or row['complete_motion'], 'SONIC qualification without completion')
                    require(row.get('train_seed') is None, 'SONIC has no training seed dimension')
                    for metric in METRICS:
                        require(finite(row[metric]) or (row[metric] is None and not row['complete_motion'] and
                                row.get('metric_unavailable_reason')), f'Invalid SONIC {metric}')
                    require(row['evidence'], 'SONIC evidence required')
                    for evidence in row['evidence']:
                        require(Path(evidence['path']).is_absolute() and sha(evidence['path']) == evidence['sha256'],
                                'SONIC evidence missing/hash mismatch')
                    for field in ('training_updates', 'training_transitions', 'candidate_rollouts'):
                        require(row.get(field, 0) == 0, 'Released SONIC cannot have per-task PPO/search cost')
                    valid[key] = dict(row, method='sonic', training_updates=0, training_transitions=0,
                                      candidate_rollouts=0, upstream_pretraining_cost=None)
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    problems.append(issue('sonic', '/'.join(map(str, key)), 'invalid', exc))
        except (ValueError, KeyError, TypeError, OSError, ImportError, RuntimeError) as exc:
            problems.append(issue('sonic_adapter', 'input', 'invalid', exc, input_path))
    elif input_path is not None or adapter_path is not None:
        problems.append(issue('sonic_adapter', 'input', 'invalid', 'Both --sonic-input and --sonic-adapter are required'))
    invalid_keys = {p['key'] for p in problems if p['scope'] == 'sonic'}
    for key in sorted(expected - valid.keys()):
        name = '/'.join(map(str, key))
        if name not in invalid_keys:
            problems.append(issue('sonic', name, 'missing', 'No validated formal rollout'))
    return list(valid.values()), raw, problems, provenance


def collect(campaign_path, snapshot, expected_strategy, *, sonic_input=None, sonic_adapter=None,
            confirmation_helper=None, helper_function='verify_confirmation_contract'):
    """Collect with the campaign's frozen code, never a cached different checkout.

    Adapter and helper paths are trusted executable Python, not sandboxed data.
    An isolated CPU worker prevents a loaded public package from shadowing the
    requested snapshot. No GPU worker, evaluator or training command is launched.
    """
    snapshot = Path(snapshot).resolve()
    options = dict(sonic_input=str(sonic_input) if sonic_input is not None else None,
                   sonic_adapter=str(sonic_adapter) if sonic_adapter is not None else None,
                   confirmation_helper=str(confirmation_helper) if confirmation_helper is not None else None,
                   helper_function=helper_function)
    loaded = [module for name, module in sys.modules.items()
              if name == 'mimicx' or name.startswith('mimicx.')]
    foreign = any(getattr(module, '__file__', None) and
                  not Path(module.__file__).resolve().is_relative_to(snapshot) for module in loaded)
    if foreign:
        worker = '''import contextlib, importlib.util, json, sys
spec = importlib.util.spec_from_file_location("confirmation_collector_worker", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.path.insert(0, sys.argv[3])
with contextlib.redirect_stdout(sys.stderr):
    result = module._collect(sys.argv[2], sys.argv[3], sys.argv[4], **json.loads(sys.argv[5]))
print(json.dumps(result, allow_nan=False))
'''
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
        command = [sys.executable, '-I', '-B', '-c', worker, str(Path(__file__).resolve()),
                   str(Path(campaign_path).resolve()), str(snapshot), expected_strategy,
                   json.dumps(options, allow_nan=False)]
        run = subprocess.run(command, env=env, capture_output=True, text=True)
        require(run.returncode == 0, 'Frozen snapshot collector failed: ' + run.stderr.strip())
        return json.loads(run.stdout)
    return _collect(campaign_path, snapshot, expected_strategy, **options)


def _collect(campaign_path, snapshot, expected_strategy, *, sonic_input=None, sonic_adapter=None,
             confirmation_helper=None, helper_function='verify_confirmation_contract'):
    study, metrics, loop = native_modules(snapshot)
    collector = Collector(campaign_path, study, metrics, loop)
    campaign = collector.campaign
    problems, arms, donors, statuses = [], [], [], []
    global_ok = True
    helper_provenance = None
    try:
        tasks = validate_campaign(campaign, expected_strategy)
        study.verify_campaign(campaign)
        confirmation_helper = confirmation_helper or Path(snapshot) / 'mimicx/experiments/motion_study_confirmation.py'
        if confirmation_helper:
            helper_hash = sha(confirmation_helper)
            hook = getattr(module_at('mimicx.experiments.confirmation_results_helper', confirmation_helper), helper_function)
            result = hook(campaign)
            require(result is not False, 'Confirmation helper rejected campaign')
            require(sha(confirmation_helper) == helper_hash, 'Confirmation helper changed during validation')
            helper_provenance = dict(path=str(confirmation_helper), sha256=helper_hash, function=helper_function)
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, AttributeError, ImportError) as exc:
        global_ok = False
        problems.append(issue('campaign', str(campaign_path), 'invalid', exc))
        tasks = {t['task_id']: t for t in campaign.get('tasks', []) if 'task_id' in t}
    jobs = campaign.get('jobs', [])
    good_donors = set()
    if global_ok:
        for job in sorted(jobs, key=lambda j: (j['method'] != 'warmstart', j['id'])):
            folder = collector.root / job['id']
            state, reason = 'missing', 'No done.json; run is not verified complete'
            if (folder / 'done.json').exists():
                try:
                    if job['method'] != 'warmstart':
                        require(set(job['dependencies']).issubset(good_donors), 'Missing/invalid donor dependency')
                    row = collector.job(job, tasks[job['task_id']])
                    if job['method'] == 'warmstart':
                        donors.append(row)
                        good_donors.add(job['id'])
                    else:
                        arms.append(row)
                    state, reason = 'validated', None
                except (ValueError, RuntimeError, OSError, KeyError, TypeError, IndexError) as exc:
                    state, reason = 'invalid', str(exc)
            statuses.append(dict(job=job['id'], task_id=job['task_id'], train_seed=job['seed'],
                                 method=job['method'], state=state, reason=reason))
            if state != 'validated':
                problems.append(issue('native', job['id'], state, reason, folder))
    else:
        statuses = [dict(job=j.get('id'), task_id=j.get('task_id'), train_seed=j.get('seed'),
                         method=j.get('method'), state='unvalidated', reason='Campaign validation failed') for j in jobs]
        problems.extend(issue('native', row['job'], row['state'], row['reason']) for row in statuses)
    sonic, raw_counts, sonic_problems, sonic_provenance = sonic_rows(sonic_input, sonic_adapter, campaign)
    problems.extend(sonic_problems)
    if sha(collector.path) != collector.campaign_hash:
        global_ok = False
        problems.append(issue('campaign', str(campaign_path), 'invalid', 'Campaign changed during collection'))
    if global_ok:
        try:
            study.verify_campaign(campaign)
        except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
            global_ok = False
            problems.append(issue('campaign', str(campaign_path), 'invalid', exc))
    native_complete = global_ok and len(arms) == 72 and len(donors) == 24
    complete = native_complete and len(sonic) == 72 and not problems
    budget = dict(expected=dict(arms=72, donors=24, ppo_updates=288000, transitions=288000 * collector.tpu,
                               candidate_checkpoints=432, candidate_rollouts=1296,
                               shared_donor_selection_rollouts=72, trained_holdout_rollouts=432,
                               sonic_formal_rollouts=72, total_specified_rollouts=1872),
                  validated=dict(arms=len(arms), donors=len(donors),
                                 ppo_updates=sum(r['completed_updates'] for r in arms + donors),
                                 transitions=sum(r['completed_transitions'] for r in arms + donors),
                                 candidate_checkpoints=6 * len(arms), candidate_rollouts=18 * len(arms),
                                 shared_donor_selection_rollouts=3 * len({(r['task_id'], r['train_seed']) for r in arms}),
                                 trained_holdout_rollouts=6 * len(arms), sonic_formal_rollouts=len(sonic)),
                  sonic=dict(per_task_ppo_updates=0, per_task_training_transitions=0, candidate_search_rollouts=0,
                             upstream_pretraining_cost=None),
                  note='Donor training counted once across all three arms, including adopted donors. '
                       'Validated counts are not total resource consumption for unfinished/invalid runs. '
                       'Candidate counts are frozen slots, not selected winners; retries are not invented.')
    attempts = [r['attempted_transitions_exact'] for r in arms + donors]
    budget['validated']['attempted_transitions_exact'] = sum(attempts) if native_complete and all(
        v is not None for v in attempts) else None
    methods = {}
    if native_complete:
        for method in METHODS:
            rows = [r for r in arms if r['method'] == method]
            methods[method] = dict(arms=len(rows), qualified=sum(r['qualified'] for r in rows),
                                  native_qualified=sum(r['native_qualified'] for r in rows),
                                  common_physical_qualified=sum(r['common_physical_qualified'] for r in rows),
                                  retained_donors=sum(r['retained_donor'] for r in rows),
                                  conditions={kind: summarize_rows([trial for r in rows for trial in
                                              r['conditions'][kind]['rollouts']]) for kind in HOLDOUT})
    if complete:
        methods['sonic'] = dict(formal_rows=72, strict_qualified_rollouts=sum(r['strict_qualified'] for r in sonic),
                               qualification_unit='formal rollout, not trained arm',
                               conditions={kind: summarize_rows([r for r in sonic if r['condition'] == kind]) for kind in HOLDOUT})
    return dict(schema='mimicx.confirmation-results.v1', status='complete' if complete else 'partial',
                full_tables_allowed=complete, native_matrix_complete=native_complete,
                campaign=str(collector.path), campaign_sha256=collector.campaign_hash, snapshot=str(snapshot),
                expected_strategy=expected_strategy, confirmation_helper=helper_provenance,
                sonic_adapter=sonic_provenance, sonic_raw_counts=raw_counts,
                aggregation='All declared tasks/seeds retained. Equal task/train-seed/rollout weights. '
                            'p95 = mean of rollout time-p95, not pooled p95. Any censored constituent makes '
                            'the aggregate undefined; no complete-case result filtering.',
                qualification='Native qualifies(selection AND both holdouts), AND nine common physical completion flags. '
                              'SONIC uses its adapter formal qualification, reported separately by rollout.',
                budget=budget, jobs=statuses, arms=arms, donors=donors, sonic_rows=sonic,
                paired_changes=paired_rows(tasks, arms), methods=methods if complete else {},
                missing_invalid=problems, timing=timing(collector.root, jobs) if global_ok else None)


def csv_text(rows, empty_fields):
    import io
    stream = io.StringIO(newline='')
    fields = list(dict.fromkeys(k for row in rows for k in row)) if rows else empty_fields
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, sort_keys=True, allow_nan=False) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()})
    return stream.getvalue()


def status_markdown(report):
    counts = report['budget']['validated']
    lines = ['# Confirmation Evidence Status', '', f"Status: **{report['status']}**", '',
             f"Validated arms: {counts['arms']}/72; donors: {counts['donors']}/24; SONIC formal rows: {counts['sonic_formal_rollouts']}/72.",
             '', 'Missing, invalid and unvalidated jobs are not finished results.', '', report['aggregation'], '',
             report['qualification'], '', report['budget']['note'], '',
             'Full tables: ' + ('released after strict completeness checks.' if report['full_tables_allowed'] else 'withheld.'), '',
             'See report.json and missing_invalid.csv for all identities and reasons.', '']
    return '\n'.join(lines)


def full_markdown(report):
    require(report['full_tables_allowed'], 'Refusing full tables for partial results')
    lines = ['# Complete Confirmation Tables', '', report['aggregation'], '',
             '| Method | Condition | Physical complete | World mean cm | World p95 cm | Local mean cm | Local p95 cm |',
             '|---|---|---:|---:|---:|---:|---:|']
    for method, row in report['methods'].items():
        for kind, values in row['conditions'].items():
            metric = ['undefined (censored)' if values[k] is None else f'{values[k]:.6f}' for k in METRICS]
            lines.append(f"| {method} | {kind} | {values['complete']}/{values['count']} | " + ' | '.join(metric) + ' |')
    lines += ['', '| Method | Strict qualified arms | Native qualified arms | Physical qualified arms | Retained donors |',
              '|---|---:|---:|---:|---:|']
    for method in METHODS:
        row = report['methods'][method]
        lines.append(f"| {method} | {row['qualified']}/24 | {row['native_qualified']}/24 | "
                     f"{row['common_physical_qualified']}/24 | {row['retained_donors']}/24 |")
    lines += ['', f"SONIC strict qualification: {report['methods']['sonic']['strict_qualified_rollouts']}/72 formal rollouts (not PPO arms).",
              '', 'All 144 paired task/training-seed/condition comparisons are in paired_changes.csv; positive reduction favors the method over its control.',
              '', '```json', json.dumps(report['budget'], indent=2), '```', '']
    return '\n'.join(lines)


def write_outputs(report, output):
    output = Path(output)
    require(not output.exists(), 'Output directory must be new; refusing stale/overwritten results')
    rollout_rows = [dict(task_id=arm['task_id'], train_seed=arm['train_seed'], method=arm['method'],
                         condition=kind, **row) for arm in report['arms'] for kind in HOLDOUT
                    for row in arm['conditions'][kind]['rollouts']]
    arms = [{k: v for k, v in arm.items() if k != 'conditions'} for arm in report['arms']]
    tables = dict(jobs=report['jobs'], arms=arms, donors=report['donors'], native_rollouts=rollout_rows,
                  sonic_rows=report['sonic_rows'], paired_changes=report['paired_changes'],
                  missing_invalid=report['missing_invalid'])
    cost_rows = []
    for method in ('warmstart',) + METHODS:
        rows = [r for r in report['arms'] + report['donors'] if r['method'] == method]
        cost_rows.append(dict(scope='validated_only', method=method, expected_training_phases=24,
                              validated_training_phases=len(rows), updates=sum(r['completed_updates'] for r in rows),
                              transitions=sum(r['completed_transitions'] for r in rows),
                              candidate_checkpoints=sum(r['candidate_checkpoints'] for r in rows),
                              candidate_rollouts=sum(r['candidate_rollouts'] for r in rows),
                              note='Shared donor counted only here' if method == 'warmstart' else 'Continuation only; donor not repeated'))
    cost_rows.append(dict(scope='released_policy', method='sonic', expected_training_phases=0,
                          validated_training_phases=0, updates=0, transitions=0, candidate_checkpoints=0,
                          candidate_rollouts=0, note='Upstream pretraining unknown, not zero'))
    tables['costs'] = cost_rows
    tables['timing'] = report['timing']['records'] if report['timing'] else []
    tables['timing_invalid'] = report['timing']['invalid_records'] if report['timing'] else []
    texts = {'report.json': json.dumps(report, indent=2, allow_nan=False) + '\n',
             'missing_invalid.json': json.dumps(report['missing_invalid'], indent=2, allow_nan=False) + '\n',
             'status.md': status_markdown(report)}
    texts.update({name + '.csv': csv_text(rows, ['state', 'reason']) for name, rows in tables.items()})
    if report['full_tables_allowed']:
        texts['full_tables.md'] = full_markdown(report)
    output.mkdir(parents=True, exist_ok=False)
    for name, content in texts.items():
        (output / name).write_text(content)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--expected-strategy', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--confirmation-helper', type=Path)
    parser.add_argument('--helper-function', default='verify_confirmation_contract', help='CPU hook accepting campaign dict')
    parser.add_argument('--sonic-input', type=Path)
    parser.add_argument('--sonic-adapter', type=Path)
    parser.add_argument('--strict', action='store_true', help='Write status, then exit 2 unless all native and SONIC evidence validates')
    args = parser.parse_args(argv)
    require(not args.output_dir.exists(), 'Output directory must be new; refusing stale/overwritten results')
    report = collect(args.campaign, args.snapshot, args.expected_strategy, sonic_input=args.sonic_input,
                     sonic_adapter=args.sonic_adapter, confirmation_helper=args.confirmation_helper,
                     helper_function=args.helper_function)
    write_outputs(report, args.output_dir)
    print(json.dumps(dict(status=report['status'], counts=report['budget']['validated'], output=str(args.output_dir))))
    return 2 if args.strict and not report['full_tables_allowed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
