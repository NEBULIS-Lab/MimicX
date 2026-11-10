"""CPU-only four-method tables from a completed confirmation collector report.

Uses the delivered fixed-window checks and all-constituent aggregation rules.
This is a report consumer, not a raw-run auditor, scorer, selector or launcher.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from mimicx.experiments.motion_study_confirmation import (
    CONFIRMATION_STRATEGY, HOLDOUT_SEEDS, TRAINING_SEEDS, confirmation_validation_protocol,
)
from mimicx.experiments.motion_study_scoring import common_protocol

METHODS = ('beyond_mimic', 'static_search', 'mimicx', 'sonic')
LABELS = ('BeyondMimic', 'Static Search', 'MimicX', 'Released SONIC')
METRICS = ('world_mean_cm', 'world_p95_cm', 'local_mean_cm', 'local_p95_cm')
CHANNELS = {'world': 'world_body_error_m', 'local': 'root_local_fk_body_error_m'}
REPRESENTATION = 'canonical_asset_fk_of_root_pose_and_29_named_joints_for_both_controllers'
P95 = 'time quantile of per-frame mean body/EE error; no frame censoring'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    def reject(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def pin(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


class Sources:
    """Resolve canonical sources or hash-bound copies from transfer audit indexes."""
    def __init__(self, indexes=()):
        self.entries, self.pins = {}, {}
        for index in indexes:
            self.track(index)
            rows = read(index)
            require(isinstance(rows, list), 'Source index must be a list')
            for row in rows:
                require(isinstance(row, dict) and all(isinstance(row.get(k), str) and row[k]
                        for k in ('source', 'copy', 'sha256')), 'Malformed source index record')
                require(len(row['sha256']) == 64 and all(c in '0123456789abcdef' for c in row['sha256']),
                        'Malformed source index hash')
                path = (Path(index).parent / row['copy']).resolve()
                require(path.is_relative_to(Path(index).parent.resolve()), 'Source copy escapes index')
                require(row.get('expected_sha256') in (None, row['sha256']), 'Source expected hash mismatch')
                value = (path, row['sha256'])
                require(row['source'] not in self.entries or self.entries[row['source']] == value,
                        'Conflicting source index entry')
                self.entries[row['source']] = value

    def track(self, path, expected=None):
        item = pin(path)
        require(expected is None or item['sha256'] == expected, 'Source hash mismatch: ' + str(path))
        require(item['path'] not in self.pins or self.pins[item['path']] == item, 'Source changed during read')
        self.pins[item['path']] = item
        return Path(item['path'])

    def path(self, source, expected=None):
        source = str(source)
        if source in self.entries:
            path, digest = self.entries[source]
            require(expected is None or expected == digest, 'Source/index pin mismatch')
            return self.track(path, digest)
        return self.track(source, expected)

    def read(self, source, expected=None):
        return read(self.path(source, expected))

    def verify(self):
        for item in self.pins.values():
            require(sha(item['path']) == item['sha256'], 'Source changed before output')


def validate_matrix(report, manifest):
    require(report['schema'] == 'mimicx.confirmation-results.v1' and report['status'] == 'complete'
            and report['full_tables_allowed'] is True and report['native_matrix_complete'] is True
            and report['missing_invalid'] == [], 'Complete strict collector report required')
    tasks = {t['task_id']: t for t in manifest['tasks']}
    require(len(tasks) == len(manifest['tasks']) == 12, 'Expected 12 unique confirmation tasks')
    expected = {(t, seed, m) for t in tasks for seed in TRAINING_SEEDS
                for m in ('warmstart',) + METHODS[:3]}
    phases = report['arms'] + report['donors']
    for rows in (phases, report['jobs']):
        keys = [(r['task_id'], r['train_seed'], r['method']) for r in rows]
        require(len(keys) == len(set(keys)) == 96 and set(keys) == expected
                and all(type(r['train_seed']) is int and r['state'] == 'validated' for r in rows),
                'Native matrix identities/state differ')
    require(len(report['arms']) == 72 and len(report['donors']) == 24
            and all(r['method'] == 'warmstart' for r in report['donors'])
            and all(r['method'] in METHODS[:3] for r in report['arms']), 'Native phase roles differ')
    require(len({r['job'] for r in phases}) == 96 and
            {(r['job'], r['task_id'], r['train_seed'], r['method']) for r in phases} ==
            {(r['job'], r['task_id'], r['train_seed'], r['method']) for r in report['jobs']},
            'Native job identities differ')
    for arm in report['arms']:
        require(set(arm['conditions']) == set(HOLDOUT_SEEDS), 'Native conditions differ')
        for condition, seeds in HOLDOUT_SEEDS.items():
            trials = arm['conditions'][condition]['rollouts']
            require(len(trials) == 3 and all(type(r['seed']) is int for r in trials)
                    and sorted(r['seed'] for r in trials) == seeds, 'Native rollout seed identities differ')
    expected_sonic = {(t, c, seed) for t in tasks for c, seeds in HOLDOUT_SEEDS.items() for seed in seeds}
    keys = []
    for row in report['sonic_rows']:
        key = (row['task_id'], row['condition'], row['seed'])
        require(type(row['seed']) is int and row['row_id'] == f'{key[0]}__{key[1]}__{key[2]}'
                and row.get('method', 'sonic') == 'sonic' and row.get('train_seed') is None,
                'SONIC identity differs')
        require(row['protocol_valid'] is True and type(row['strict_qualified']) is bool
                and type(row['complete_motion']) is bool
                and (not row['strict_qualified'] or row['complete_motion']), 'Invalid SONIC outcome flags')
        keys.append(key)
    require(len(keys) == len(set(keys)) == 72 and set(keys) == expected_sonic
            and type(report['sonic_raw_counts']['physical_executions']) is int
            and report['sonic_raw_counts']['physical_executions'] == 72,
            'Expected 72 unique physical SONIC outcomes')
    return tasks


def check_window(common, row, task, campaign):
    protocol = campaign['runtime']['common_protocol']
    # The public scorer fixes 50 Hz even when the minimal protocol omits fps.
    n, fps = task['frames'], protocol.get('fps', 50)
    require(type(n) is int and n > 0 and fps == task['fps'] == 50, 'Common frame/clock differs')
    prefix = min(n, math.floor(protocol['common_prefix_seconds'] * fps) + 1)
    require(common['schema'] == 'mimicx.fixed-clock-common-metrics.v1'
            and type(common['frames']) is int and common['frames'] == n
            and type(common['common_prefix_frames']) is int and common['common_prefix_frames'] == prefix
            and finite(common['duration_seconds'])
            and math.isclose(common['duration_seconds'], (n - 1) / fps, rel_tol=0, abs_tol=1e-9),
            'Common prefix/full window mismatch')
    require(common['scoring_body_names'] == campaign['reference_contract']['body_names']
            and common['scoring_body_count'] == 30 and common['body_representation'] == REPRESENTATION
            and common['p95_definition'] == P95, 'Common body/metric definition mismatch')
    valid = common['valid_physical_frames']
    require(type(valid) is int and 0 <= valid <= n
            and (common['full_horizon'] is None) == (valid < n)
            and (common['common_prefix'] is None) == (valid < prefix), 'Common window censoring mismatch')
    require(type(common['complete_motion']) is bool and type(row['complete_motion']) is bool
            and common['complete_motion'] is row['complete_motion']
            and (not row['complete_motion'] or valid == n), 'Common completion mismatch')
    for window in ('full_horizon', 'common_prefix'):
        block = common[window]
        if window in row:
            require(row[window] == block, 'Collector/raw window metric mismatch')
        if block is not None:
            for channel in CHANNELS.values():
                require(all(finite(block[channel][stat]) for stat in ('mean', 'p95')), 'Invalid common metric')
    for metric in METRICS:
        space, stat, _ = metric.split('_')
        block = common['full_horizon']
        value = None if block is None else 100 * block[CHANNELS[space]][stat]
        require(row[metric] is None if value is None else finite(row[metric]) and
                math.isclose(row[metric], value, rel_tol=1e-9, abs_tol=1e-9),
                'Collector/common scalar metric mismatch')
    return dict(frames=n, fps=fps, prefix_start_frame=0, prefix_end_frame_inclusive=prefix - 1,
                prefix_frames=prefix, full_start_frame=0, full_end_frame_inclusive=n - 1)


def collect_trials(report, manifest, reader, manifest_path):
    tasks = validate_matrix(report, manifest)
    campaign = reader.read(report['campaign'], report['campaign_sha256'])
    require(campaign['manifest_sha256'] == sha(reader.path(manifest_path)), 'Supplied manifest differs')
    require(len(campaign['tasks']) == len(tasks) and
            {t['task_id']: (t['sha256'], t['frames'], t['fps']) for t in campaign['tasks']} ==
            {t: (v['sha256'], v['frames'], v['fps']) for t, v in tasks.items()}, 'Campaign task identity mismatch')
    runtime = campaign['runtime']
    require(runtime['refinement_strategy'] == CONFIRMATION_STRATEGY, 'Confirmation strategy differs')
    confirmation_validation_protocol(runtime['validation_protocol'])
    protocol = runtime['common_protocol']
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    require(digest == campaign['common_protocol_sha256'] and
            all(protocol.get(k) == v and (type(protocol[k]) in (int, float) if type(v) in (int, float)
                                         else type(protocol[k]) is type(v))
                for k, v in common_protocol().items()), 'Frozen common protocol differs')
    bodies = campaign['reference_contract']['body_names']
    require(isinstance(bodies, list) and all(isinstance(b, str) and b for b in bodies)
            and len(bodies) == len(set(bodies)) == 30, 'Expected 30 unique canonical bodies')
    jobs = Path(report['campaign']).parent / 'jobs'
    trials, windows = [], []
    for arm in report['arms']:
        task = tasks[arm['task_id']]
        job = Path(arm['job'])
        require(not job.is_absolute() and '..' not in job.parts, 'Native job escapes campaign')
        for condition, seeds in HOLDOUT_SEEDS.items():
            group = reader.read(jobs / job / 'holdout' / condition / 'summary.json')
            require(group['seeds'] == seeds and group['sha256'] == arm['selected_sha256']
                    and group['checkpoint'] == arm['selected_checkpoint'] and group['horizon'] == task['frames'] - 1
                    and group['noise'] == runtime['validation_protocol'][condition + '_noise']
                    and len(group['common_metrics']) == 3, 'Native holdout evidence identity mismatch')
            lookup = {r['seed']: r for r in arm['conditions'][condition]['rollouts']}
            for seed, block in zip(seeds, group['common_metrics']):
                row = lookup[seed]
                key = dict(task_id=task['task_id'], method=arm['method'], condition=condition,
                           seed=seed, train_seed=arm['train_seed'])
                windows.append(dict(key, **check_window(block, row, task, campaign)))
                trials.append(dict(key, complete_motion=row['complete_motion'], common=block))
    for row in report['sonic_rows']:
        key = dict(task_id=row['task_id'], method='sonic', condition=row['condition'], seed=row['seed'], train_seed=None)
        windows.append(dict(key, **check_window(row['common_metrics'], row, tasks[row['task_id']], campaign)))
        trials.append(dict(key, complete_motion=row['complete_motion'], common=row['common_metrics']))
    return trials, windows


def aggregate(trials):
    result = dict(count=len(trials), complete=sum(r['complete_motion'] for r in trials), defined={})
    for window in ('full_horizon', 'common_prefix'):
        for metric in METRICS:
            space, stat, _ = metric.split('_')
            values = []
            for row in trials:
                block = row['common'][window]
                value = None if block is None or (window == 'full_horizon' and not row['complete_motion']) else \
                    100 * block[CHANNELS[space]][stat]
                values.append(value)
            key = window + '_' + metric
            result['defined'][key] = sum(v is not None for v in values)
            result[key] = mean(values) if values and all(v is not None for v in values) else None
    return result


def render(cohort, tasks):
    def table(rows):
        by_method = {r['method']: r for r in rows}
        lines = ['| Metric | ' + ' | '.join(LABELS) + ' |', '| --- | --- | --- | --- | --- |',
                 '| Full-motion completion | ' + ' | '.join(
                     f"{by_method[m]['complete']}/{by_method[m]['count']}" for m in METHODS) + ' |']
        for window in ('full_horizon', 'common_prefix'):
            for metric in METRICS:
                values = [by_method[m][window + '_' + metric] for m in METHODS]
                lines.append('| ' + window + ' ' + metric + ' | ' + ' | '.join(
                    'undefined' if v is None else f'{v:.6g}' for v in values) + ' |')
        return lines
    lines = ['# Four-Method Confirmation Comparison', '',
             'All 12 tasks retained. Native: two training seeds x three rollouts per condition; '
             'SONIC: three released-policy rollouts per condition. No pooled native/SONIC mean.', '',
             'Errors are cm. Time-p95 is the mean of rollout time-p95, not a pooled quantile. '
             'Any failed or censored constituent makes full-horizon aggregates undefined. '
             'The inclusive phase-zero common prefix is separate, never a substitute for completion. '
             'Native endpoint qualification and SONIC rollout qualification are not compared. '
             'Costs and temporal tails are outside this report.', '']
    for condition in HOLDOUT_SEEDS:
        lines += ['## ' + condition.title(), ''] + table([r for r in cohort if r['condition'] == condition]) + ['']
        for task_id in dict.fromkeys(r['task_id'] for r in tasks):
            lines += ['### ' + task_id.replace('|', '\\|'), ''] + table(
                [r for r in tasks if r['condition'] == condition and r['task_id'] == task_id]) + ['']
    return '\n'.join(lines)


def collect(report_path, manifest_path, output, *, source_indexes=()):
    output = Path(output)
    require(not output.exists(), 'Output must be a new directory')
    reader = Sources(source_indexes)
    report, manifest = reader.read(report_path), reader.read(manifest_path)
    for source in (__file__, common_protocol.__code__.co_filename,
                   confirmation_validation_protocol.__code__.co_filename):
        reader.track(source)
    try:
        trials, windows = collect_trials(report, manifest, reader, manifest_path)
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ValueError(f'Malformed confirmation evidence: {exc}') from exc
    cohort = [dict(method=m, condition=c, **aggregate([r for r in trials if r['method'] == m and r['condition'] == c]))
              for c in HOLDOUT_SEEDS for m in METHODS]
    tasks = [dict(task_id=t['task_id'], method=m, condition=c, **aggregate(
        [r for r in trials if r['task_id'] == t['task_id'] and r['method'] == m and r['condition'] == c]))
        for t in manifest['tasks'] for c in HOLDOUT_SEEDS for m in METHODS]
    text = render(cohort, tasks)
    reader.verify()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'four_methods.md').write_text(text, encoding='utf-8')
    for name, rows in (('cohort', cohort), ('all_tasks', tasks), ('window_audit', windows)):
        (output / (name + '.json')).write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        with (output / (name + '.csv')).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, dict) else v for k, v in row.items()}
                             for row in rows)
    reader.verify()
    receipt = dict(schema='mimicx.confirmation-four-method-tables.v1', status='complete_four_method_tables',
                   inputs=list(reader.pins.values()), outputs=[pin(f) for f in sorted(output.iterdir())],
                   evidence_scope='CPU summary/window verification of a strict collector report; no new physical run')
    (output / 'FOUR_METHODS.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('report', 'manifest', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source-index', type=Path, action='append', default=[],
                        help='Repeatable transferred sha-audit.json index')
    args = parser.parse_args(argv)
    collect(args.report, args.manifest, args.output, source_indexes=args.source_index)
    print(json.dumps(dict(output=str(args.output), tasks=12, methods=4, window_checks=504)))


if __name__ == '__main__':
    main()
