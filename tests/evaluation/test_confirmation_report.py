"""Synthetic evidence only; no external datasets or physical executions."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


def api():
    assert importlib.util.find_spec('mimicx.evaluation.confirmation_report') is not None
    from mimicx.evaluation import confirmation_report
    return confirmation_report


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def evidence(tmp_path):
    from mimicx.experiments.motion_study_confirmation import (
        CONFIRMATION_STRATEGY, HOLDOUT_SEEDS, PROTOCOL_VERSION,
    )
    from mimicx.experiments.motion_study_scoring import common_protocol
    tasks = [dict(task_id=f'task_{i}', sha256=f'{i:064x}', frames=101, fps=50) for i in range(12)]
    manifest = tmp_path / 'manifest.json'
    campaign_path = tmp_path / 'campaign.json'
    protocol = dict(common_protocol(), fps=50)
    campaign = dict(tasks=tasks, manifest_sha256=write(manifest, dict(tasks=tasks)),
                    runtime=dict(refinement_strategy=CONFIRMATION_STRATEGY,
                        common_protocol=protocol, validation_protocol=dict(protocol_version=PROTOCOL_VERSION,
                            selection_seeds=[1001, 1002, 1003], selection_noise=.005,
                            nominal_noise=.005, perturbed_noise=.01,
                            holdout=dict(confirmation=HOLDOUT_SEEDS))),
                    reference_contract=dict(body_names=[f'body_{i}' for i in range(30)]))
    campaign['common_protocol_sha256'] = hashlib.sha256(json.dumps(
        protocol, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    block = {c: dict(mean=.01, p95=.02) for c in ('world_body_error_m', 'root_local_fk_body_error_m')}
    common = dict(schema='mimicx.fixed-clock-common-metrics.v1', frames=101,
                  common_prefix_frames=51, duration_seconds=2., scoring_body_count=30,
                  scoring_body_names=campaign['reference_contract']['body_names'],
                  body_representation='canonical_asset_fk_of_root_pose_and_29_named_joints_for_both_controllers',
                  p95_definition='time quantile of per-frame mean body/EE error; no frame censoring',
                  valid_physical_frames=101, complete_motion=True,
                  full_horizon=copy.deepcopy(block), common_prefix=copy.deepcopy(block))
    trial = dict(complete_motion=True, world_mean_cm=1., world_p95_cm=2.,
                 local_mean_cm=1., local_p95_cm=2., common_prefix=copy.deepcopy(block))
    report = dict(schema='mimicx.confirmation-results.v1', status='complete', full_tables_allowed=True,
                  native_matrix_complete=True, missing_invalid=[], arms=[], donors=[], jobs=[],
                  sonic_rows=[], sonic_raw_counts=dict(physical_executions=72),
                  campaign=str(campaign_path), campaign_sha256=write(campaign_path, campaign))
    for task in tasks:
        for train_seed in (101, 202):
            for method in ('warmstart', 'beyond_mimic', 'static_search', 'mimicx'):
                job = f'{task["task_id"]}/mjlab/seed{train_seed}/{method}'
                arm = dict(task_id=task['task_id'], train_seed=train_seed, method=method, job=job,
                           state='validated', selected_sha256='f'*64, selected_checkpoint=job+'/selected.pt',
                           qualified=True, native_qualified=True, common_physical_qualified=True,
                           retained_donor=False, conditions={})
                report['jobs'].append({k: arm[k] for k in ('task_id', 'train_seed', 'method', 'job', 'state')})
                report['donors' if method == 'warmstart' else 'arms'].append(arm)
                if method == 'warmstart':
                    continue
                for condition, seeds in HOLDOUT_SEEDS.items():
                    arm['conditions'][condition] = dict(rollouts=[dict(copy.deepcopy(trial), seed=s) for s in seeds])
                    write(tmp_path/'jobs'/job/'holdout'/condition/'summary.json',
                          dict(seeds=seeds, sha256=arm['selected_sha256'], checkpoint=arm['selected_checkpoint'],
                               horizon=100, noise=.005 if condition == 'nominal' else .01,
                               common_metrics=[copy.deepcopy(common) for _ in seeds]))
        for condition, seeds in HOLDOUT_SEEDS.items():
            for seed in seeds:
                report['sonic_rows'].append(dict(copy.deepcopy(trial), task_id=task['task_id'],
                    condition=condition, seed=seed, row_id=f'{task["task_id"]}__{condition}__{seed}',
                    method='sonic', train_seed=None, protocol_valid=True, strict_qualified=True,
                    common_metrics=copy.deepcopy(common)))
    report_path = tmp_path/'report.json'
    write(report_path, report)
    return report_path, manifest, report, campaign


def test_complete_four_method_report(evidence, tmp_path):
    f = api()
    report, manifest, _, _ = evidence
    output = tmp_path/'tables'
    receipt = f.collect(report, manifest, output)
    assert len(json.loads((output/'window_audit.json').read_text())) == 504
    rows = json.loads((output/'cohort.json').read_text())
    assert len(rows) == 8
    assert {r['count'] for r in rows if r['method'] == 'sonic'} == {36}
    assert {r['count'] for r in rows if r['method'] != 'sonic'} == {72}
    assert '| Metric | BeyondMimic | Static Search | MimicX | Released SONIC |' in (output/'four_methods.md').read_text()
    assert all(f.sha(p['path']) == p['sha256'] for p in receipt['outputs'])
    with pytest.raises(ValueError, match='new directory'):
        f.collect(report, manifest, output)


@pytest.mark.parametrize('mutation', ['duplicate_native', 'duplicate_sonic', 'seed', 'prefix', 'bodies',
                                      'value', 'boolean', 'strict_failure', 'pending', 'missing'])
def test_rejects_invalid_evidence(evidence, tmp_path, mutation):
    f = api()
    path, manifest, report, _ = evidence
    row = report['sonic_rows'][0]
    if mutation == 'duplicate_native': report['arms'][-1] = copy.deepcopy(report['arms'][0])
    if mutation == 'duplicate_sonic': report['sonic_rows'][-1] = copy.deepcopy(row)
    if mutation == 'seed': row['seed'] = True
    if mutation == 'prefix': row['common_metrics']['common_prefix_frames'] = 50
    if mutation == 'bodies': row['common_metrics']['scoring_body_names'].reverse()
    if mutation == 'value': row['common_metrics']['common_prefix']['world_body_error_m']['mean'] = .03
    if mutation == 'boolean': row['complete_motion'] = 1
    if mutation == 'strict_failure': row['complete_motion'] = False
    if mutation == 'pending': report['status'] = 'pending'
    if mutation == 'missing': report['missing_invalid'] = [dict(scope='sonic')]
    write(path, report)
    with pytest.raises(ValueError):
        f.collect(path, manifest, tmp_path/'rejected')
    assert not (tmp_path/'rejected').exists()


def test_failure_is_retained_not_averaged(evidence, tmp_path):
    f = api()
    path, manifest, report, _ = evidence
    row = report['sonic_rows'][0]
    row.update(complete_motion=False, strict_qualified=False)
    row['common_metrics']['complete_motion'] = False
    write(path, report)
    f.collect(path, manifest, tmp_path/'failed')
    rows = json.loads((tmp_path/'failed/cohort.json').read_text())
    sonic = next(r for r in rows if r['method'] == 'sonic' and r['condition'] == 'nominal')
    assert (sonic['complete'], sonic['count']) == (35, 36)
    assert sonic['full_horizon_world_mean_cm'] is None
    assert sonic['defined']['full_horizon_world_mean_cm'] == 35
    assert sonic['common_prefix_world_mean_cm'] == 1.
    assert report['sonic_rows'][0]['world_mean_cm'] == 1.


def test_protocol_drift_rejected_even_with_rehashed_campaign(evidence, tmp_path):
    f = api()
    path, manifest, report, campaign = evidence
    campaign['runtime']['common_protocol']['root_height_m'] = 999
    report['campaign_sha256'] = write(Path(report['campaign']), campaign)
    write(path, report)
    with pytest.raises(ValueError, match='protocol'):
        f.collect(path, manifest, tmp_path/'rejected')


@pytest.mark.parametrize('payload', [{}, [], dict(schema='wrong'), None])
def test_malformed_report_fails_before_output(evidence, tmp_path, payload):
    f = api()
    path, manifest, _, _ = evidence
    write(path, payload)
    with pytest.raises(ValueError):
        f.collect(path, manifest, tmp_path/'rejected')
    assert not (tmp_path/'rejected').exists()


@pytest.mark.parametrize('field,value', [('duration_seconds', 2.01), ('frames', 100),
    ('valid_physical_frames', 50), ('valid_physical_frames', True), ('common_prefix_frames', 50),
    ('p95_definition', 'pooled'), ('world_mean_cm', float('nan'))])
def test_malformed_clock_and_metrics_rejected(evidence, tmp_path, field, value):
    f = api()
    path, manifest, report, _ = evidence
    row = report['sonic_rows'][0]
    if field == 'world_mean_cm':
        row[field] = value
    else:
        row['common_metrics'][field] = value
    write(path, report)
    with pytest.raises(ValueError):
        f.collect(path, manifest, tmp_path/'rejected')
    assert not (tmp_path/'rejected').exists()


def test_missing_clock_prefix_is_not_imputed(evidence, tmp_path):
    f = api()
    path, manifest, report, _ = evidence
    row = report['sonic_rows'][0]
    row.update(complete_motion=False, strict_qualified=False, common_prefix=None)
    row.update({k: None for k in f.METRICS})
    row['common_metrics'].update(complete_motion=False, valid_physical_frames=50,
                                 full_horizon=None, common_prefix=None)
    write(path, report)
    f.collect(path, manifest, tmp_path/'censored')
    rows = json.loads((tmp_path/'censored/cohort.json').read_text())
    sonic = next(r for r in rows if r['method'] == 'sonic' and r['condition'] == 'nominal')
    assert sonic['count'] == 36 and sonic['complete'] == 35
    assert sonic['full_horizon_world_mean_cm'] is None
    assert sonic['common_prefix_world_mean_cm'] is None


def test_time_p95_is_mean_of_rollout_quantiles():
    f = api()
    rows = [dict(complete_motion=True, common={window: {channel: dict(mean=.1, p95=p95)
            for channel in f.CHANNELS.values()} for window in ('full_horizon', 'common_prefix')})
            for p95 in (.1, .3)]
    assert f.aggregate(rows)['full_horizon_world_p95_cm'] == 20


def test_minimal_protocol_still_enforces_fixed_50hz(evidence):
    f = api()
    _, _, report, campaign = evidence
    row = report['sonic_rows'][0]
    task = campaign['tasks'][0]
    campaign['runtime']['common_protocol'].pop('fps')
    assert f.check_window(row['common_metrics'], row, task, campaign)['fps'] == 50
    campaign['runtime']['common_protocol']['fps'] = 100
    with pytest.raises(ValueError, match='clock'):
        f.check_window(row['common_metrics'], row, task, campaign)


def test_source_index_pins_copies_and_rejects_escape(tmp_path):
    f = api()
    digest = write(tmp_path/'copy.json', dict(ok=True))
    index = tmp_path/'index.json'
    entry = dict(source='/unavailable/example.json', copy='copy.json', sha256=digest)
    write(index, [entry])
    reader = f.Sources([index])
    assert reader.read(entry['source']) == dict(ok=True)
    write(tmp_path/'copy.json', {})
    with pytest.raises(ValueError, match='changed'):
        reader.verify()
    entry['copy'] = '../outside.json'
    write(index, [entry])
    with pytest.raises(ValueError, match='escapes'):
        f.Sources([index])


@pytest.mark.parametrize('rows', [{}, [None], [{}], [dict(source='file', copy='copy', sha256='invalid')]])
def test_malformed_source_indexes(tmp_path, rows):
    f = api()
    path = tmp_path/'index.json'
    write(path, rows)
    with pytest.raises(ValueError, match='[Ss]ource index'):
        f.Sources([path])


def test_cli_help_has_no_simulator_imports():
    api()
    command = 'from mimicx.evaluation.confirmation_report import main; import sys; '
    command += 'assert not {"torch", "mujoco", "isaaclab", "mjlab"}.intersection(sys.modules); main(["--help"])'
    run = subprocess.run([sys.executable, '-c', command], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert '--source-index' in run.stdout
