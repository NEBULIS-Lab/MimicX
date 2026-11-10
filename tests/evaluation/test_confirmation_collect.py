"""Synthetic CPU receipts, not research results. No Torch, GPU, SSH or training."""
import copy
from contextlib import contextmanager
import csv
import importlib
import json
from pathlib import Path
import tempfile
import unittest
import sys
import subprocess
from unittest.mock import patch

import numpy as np
from mimicx.evaluation import confirmation_collect as cr

HERE = Path(__file__).resolve().parent
SNAPSHOT = Path(__file__).resolve().parents[2]
STUDY, METRICS, LOOP = cr.native_modules(SNAPSHOT)
HELPER = importlib.import_module('mimicx.experiments.motion_study_confirmation')
SCORING = importlib.import_module('mimicx.experiments.motion_study_scoring')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def seal(folder, signature, status, training=None):
    record = dict(signature=signature, status=status, artifacts={str(p.relative_to(folder)): cr.sha(p)
                  for p in folder.rglob('*') if p.is_file() and p != folder / 'done.json'})
    if training is not None:
        record['training'] = training
    save(folder / 'done.json', record)


def campaign_fixture(root):
    backend = root / 'backend'
    (backend / 'src').mkdir(parents=True)
    (backend / 'scripts').mkdir()
    (backend / 'scripts/train.py').write_text('# fixture only\n')
    xml = root / 'canonical.xml'
    xml.write_text('<mujoco/>')
    protocol = SCORING.common_protocol()
    runtime = dict(backend_root=str(backend), num_envs=4096, steps_per_env=24,
                   common_model_xml=str(xml), refinement_strategy=HELPER.CONFIRMATION_STRATEGY,
                   seeds={'confirmation': [101, 202]}, backends=['mjlab'], common_protocol=protocol,
                   validation_protocol=HELPER.confirmation_validation_protocol({'holdout': {'confirmation': cr.HOLDOUT}}))
    tasks = []
    for i in range(12):
        motion = root / f'motion{i}.npz'
        np.savez(motion, fixture=np.zeros(3))
        tasks.append(dict(task_id=f'task{i:02d}', split='confirmation', training_eligible=True,
                          frames=3, fps=50., duration=.06, motion_path=str(motion), sha256=cr.sha(motion)))
    save(root / 'manifest.json', dict(tasks=tasks))
    campaign = dict(tasks=tasks, runtime=runtime, jobs=STUDY.make_jobs(tasks, runtime),
                    budget=HELPER.BUDGET, reference_contract=dict(joint_names=[f'j{i}' for i in range(29)],
                    body_names=[f'b{i}' for i in range(30)], xml_sha256=cr.sha(xml)),
                    common_protocol_sha256=HELPER._sha(protocol), sources=STUDY.source_identity(runtime),
                    manifest_sha256=cr.sha(root / 'manifest.json'))
    contract = root / 'freeze.json'
    save(contract, HELPER._record(campaign))
    runtime.update(confirmation_contract=str(contract), confirmation_contract_sha256=cr.sha(contract))
    path = root / 'campaign.json'
    save(path, campaign)
    HELPER.verify_confirmation_contract(campaign)
    return path, campaign


def group_fixture(root, path, campaign, task, checkpoint, seeds, noise, value=.02, censored=False):
    rows = [dict(step=i, robot_anchor_x=value, robot_anchor_y=0., robot_anchor_z=0.,
                 ref_anchor_x=0., ref_anchor_y=0., ref_anchor_z=0.,
                 body_pos_error_max=value, error_anchor_rot=value,
                 done_any=censored and i == 2) for i in range(1, 3)]
    native = METRICS.summarize(rows, 2)
    common = dict(frames=3, complete_motion=not censored, scoring_body_count=30,
                  scoring_body_names=campaign['reference_contract']['body_names'],
                  valid_physical_frames=2 if censored else 3,
                  native_failure_frame=2 if censored else None, first_failure_frame=2 if censored else None,
                  startup_failure=False,
                  full_horizon=None if censored else {field: dict(mean=value, p95=value)
                                                    for field in {v[0] for v in cr.METRICS.values()}})
    common.update(schema='mimicx.fixed-clock-common-metrics.v1', duration_seconds=.04,
                  common_prefix_frames=3, common_prefix=copy.deepcopy(common['full_horizon']),
                  body_representation='canonical_asset_fk_of_root_pose_and_29_named_joints_for_both_controllers',
                  p95_definition='time quantile of per-frame mean body/EE error; no frame censoring')
    for seed in seeds:
        run = root / f'seed{seed}'
        rollout = run / 'rollout'
        rollout.mkdir(parents=True)
        with (rollout / 'steps.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        save(rollout / 'metrics.json', dict(steps=2))
        save(rollout / 'common_metrics.json', common)
        actual = {k: np.zeros((3, *v)) for k, v in dict(joint_pos=(29,), joint_vel=(29,), body_pos=(30, 3),
                  body_quat=(30, 4), body_lin_vel=(30, 3), body_ang_vel=(30, 3)).items()}
        actual['body_quat'][:, :, 0] = 1.
        np.savez(rollout / 'native_actual.npz', **actual, valid_mask=[True, True, not censored], timestamps=np.arange(3) / 50.)
        np.savez(rollout / 'common_steps.npz', failure_predicate=np.zeros(3, dtype=bool), **{field: np.full(3, value) for field in
                  {v[0] for v in cr.METRICS.values()}})
        replay = rollout / 'replay'
        replay.mkdir()
        arrays = {}
        for key in ('timestamps', 'robot_root_pose', 'robot_joint_positions', 'robot_body_pose',
                    'reference_root_pose', 'reference_joint_positions', 'reference_body_pose'):
            arr = np.arange(1, 3) / 50. if key == 'timestamps' else np.zeros((2, 3))
            np.save(replay / f'{key}.npy', arr)
            arrays[key] = dict(filename=f'{key}.npy', sha256=cr.sha(replay / f'{key}.npy'))
        save(replay / 'manifest.json', dict(frames=2, fps=50., joint_count=29, arrays=arrays))
        seal(run, dict(campaign_sha256=cr.sha(path), checkpoint_sha256=cr.sha(checkpoint),
                       motion_sha256=task['sha256'], seed=seed, noise=noise, horizon=2), 'evaluated')
    summary = dict(checkpoint=str(checkpoint), sha256=cr.sha(checkpoint), seeds=seeds, noise=noise,
                   horizon=2, results=[native] * 3, common_metrics=[common] * 3, qualified=METRICS.qualifies([native] * 3))
    save(root / 'summary.json', summary)
    return summary


def native_fixture(path, campaign):
    jobs = path.parent / 'jobs'
    for job in campaign['jobs']:
        folder = jobs / job['id']
        folder.mkdir(parents=True)
        task = next(t for t in campaign['tasks'] if t['task_id'] == job['task_id'])
        tpu = 4096 * 24
        training = dict(completed_updates=3000, transitions=3000 * tpu,
                        status='training_complete_selection_pending', selection_checkpoints=[],
                        attempted_transitions_exact=None)
        if job['method'] == 'warmstart':
            (folder / 'final.pt').write_bytes(b'fixture donor, not real weights')
            save(folder / 'training.json', training)
            save(folder / 'effective_training.json', dict(fixture=True))
            seal(folder, dict(campaign_sha256=cr.sha(path), job=job), training['status'], training)
            group_fixture(folder.parent / 'shared_validation/warmstart', path, campaign, task,
                          folder / 'final.pt', cr.SELECTION, .005)
            continue
        slots = [f'model_{i}.pt' for i in range(6)]
        donor = folder.parent / 'warmstart/final.pt'
        history = []
        for i, slot in enumerate(slots):
            (folder / slot).write_bytes(f'candidate {i}'.encode())
            group_fixture(folder / 'validation' / f'slot{i:02d}', path, campaign, task, folder / slot, cr.SELECTION, .005)
            gate = dict(slot=i, accepted=False, candidate=str(folder / slot), candidate_sha256=cr.sha(folder / slot),
                        incumbent=str(donor), incumbent_sha256=cr.sha(donor))
            history.append(gate)
            save(folder / 'gates' / f'slot{i:02d}.json', gate)
        selection = dict(checkpoint=str(donor), sha256=cr.sha(donor), history=history, uses_holdout=False,
                         seeds=cr.SELECTION, slots=6, criterion=HELPER.CONFIRMATION_STRATEGY)
        save(folder / 'selection.json', selection)
        stages = [dict(updates=s['updates'], transitions=s['updates'] * tpu) for s in job['stages']]
        budget = dict(shared_warmstart_updates=3000, continuation_updates=3000,
                      arm_updates_including_reused_warmstart=6000, reused_warmstart_training_cost_not_zero=True,
                      candidate_selection_rollouts=18, shared_warmstart_selection_rollouts=3, independent_holdout_rollouts=6)
        common, native = {}, {}
        for kind, seeds in cr.HOLDOUT.items():
            value = {'beyond_mimic': .02, 'static_search': .03, 'mimicx': .04}[job['method']]
            # A real negative/censored outcome must remain valid and in the matrix.
            censored = job['task_id'] == 'task00' and job['method'] == 'mimicx' and kind == 'perturbed'
            group = group_fixture(folder / 'holdout' / kind, path, campaign, task, donor, seeds,
                                  .005 if kind == 'nominal' else .01, value=value, censored=censored)
            common[kind], native[kind] = group['common_metrics'], group['results']
        warm = cr.read(folder.parent / 'shared_validation/warmstart/summary.json')
        native_ok = all(METRICS.qualifies(v) for v in native.values())
        physical = all(m['complete_motion'] for v in common.values() for m in v)
        report = dict(status='completed', selected_checkpoint=str(donor), selected_sha256=cr.sha(donor),
                      selected_warmstart=True, selection=warm['results'], common_selection_metrics=warm['common_metrics'],
                      holdout=native, common_physical_metrics=common, qualified=native_ok and physical,
                      native_qualified=native_ok, common_physical_qualified=physical,
                      common_protocol=campaign['runtime']['common_protocol'],
                      common_protocol_sha256=campaign['common_protocol_sha256'], completed_updates=3000,
                      transitions=3000 * tpu, stages=stages, budget_accounting=budget)
        training.update(status='completed', stages=stages, budget_accounting=budget,
                        selection_checkpoints=slots, selected_checkpoint=str(donor), selected_sha256=cr.sha(donor),
                        qualified=report['qualified'])
        save(folder / 'report.json', report)
        save(folder / 'training.json', training)
        save(folder / 'effective_training.json', dict(fixture=True))
        (folder / 'final.pt').write_bytes(donor.read_bytes())
        seal(folder, dict(campaign_sha256=cr.sha(path), job=job), 'completed', training)


def sonic_fixture(root, campaign):
    evidence = root / 'sonic-evidence.txt'
    evidence.write_text('synthetic normalized evidence only')
    rows = [dict(task_id=task['task_id'], condition=kind, seed=seed, complete_motion=True, strict_qualified=True,
                 evidence=[dict(path=str(evidence), sha256=cr.sha(evidence))], **{k: 4. for k in cr.METRICS})
            for task in campaign['tasks'] for kind, seeds in cr.HOLDOUT.items() for seed in seeds]
    for row in rows:
        common = copy.deepcopy(cr.read(root / 'jobs' / row['task_id'] / 'mjlab/seed101/beyond_mimic/holdout'
                                       / row['condition'] / 'summary.json')['common_metrics'][0])
        for window in ('full_horizon', 'common_prefix'):
            for values in common[window].values():
                values.update(mean=.04, p95=.04)
        row.update(row_id=f"{row['task_id']}__{row['condition']}__{row['seed']}",
                   protocol_valid=True, common_metrics=common, common_prefix=copy.deepcopy(common['common_prefix']))
    path = root / 'normalized-sonic.json'
    save(path, dict(rows=rows, raw_counts={'received': 79, 'formal': 72, 'diagnostic': 7,
                                        'physical_executions': 72}, invalid_rows=[]))
    adapter = root / 'fixture_adapter.py'
    adapter.write_text('import json\ndef collect_rows(input_path, campaign):\n    return json.loads(input_path.read_text())\n')
    return path, adapter


@contextmanager
def changed(path, content=None):
    old = path.read_bytes()
    try:
        if content is None:
            path.unlink()
        else:
            path.write_bytes(content)
        yield
    finally:
        path.write_bytes(old)


class CollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='cpu-collector-fixture-')
        cls.root = Path(cls.temp.name)
        cls.path, cls.campaign = campaign_fixture(cls.root)
        native_fixture(cls.path, cls.campaign)
        cls.sonic, cls.adapter = sonic_fixture(cls.root, cls.campaign)
        cls.collector = cr.Collector(cls.path, STUDY, METRICS, LOOP)
        cls.job = next(j for j in cls.campaign['jobs'] if j['method'] == 'mimicx')
        cls.folder = cls.root / 'jobs' / cls.job['id']

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def collect(self, **kwargs):
        return cr.collect(self.path, SNAPSHOT, HELPER.CONFIRMATION_STRATEGY, **kwargs)

    def test_full_matrix_and_outputs(self):
        report = self.collect(sonic_input=self.sonic, sonic_adapter=self.adapter)
        self.assertEqual(report['missing_invalid'], [])
        self.assertEqual(report['status'], 'complete')
        budget = report['budget']['validated']
        self.assertEqual((budget['arms'], budget['donors'], budget['ppo_updates']), (72, 24, 288000))
        self.assertEqual(budget['candidate_rollouts'], 1296)
        self.assertEqual(budget['shared_donor_selection_rollouts'], 72)
        self.assertEqual(budget['trained_holdout_rollouts'], 432)
        self.assertEqual(report['sonic_raw_counts']['received'], 79)
        self.assertEqual(len(report['paired_changes']), 144)
        self.assertEqual(report['methods']['mimicx']['qualified'], 22)
        self.assertIsNone(report['methods']['mimicx']['conditions']['perturbed']['world_mean_cm'])
        self.assertTrue(any(r['world_mean_cm_reduction_cm'] is not None and r['world_mean_cm_reduction_cm'] < 0
                            for r in report['paired_changes']))
        output = self.root / 'full-output'
        cr.write_outputs(report, output)
        self.assertTrue((output / 'full_tables.md').exists())
        with (output / 'native_rollouts.csv').open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 432)
        with (output / 'costs.csv').open() as stream:
            self.assertEqual(sum(int(r['updates']) for r in csv.DictReader(stream)), 288000)
        with self.assertRaises(ValueError):
            cr.write_outputs(report, output)

    def test_public_cli_producer_to_four_method_consumer(self):
        from mimicx.evaluation import confirmation_report
        output = self.root / 'cli-collected'
        command = [sys.executable, '-B', str(SNAPSHOT / 'scripts/experiments/collect_results.py'),
                   '--campaign', str(self.path), '--snapshot', str(SNAPSHOT),
                   '--expected-strategy', HELPER.CONFIRMATION_STRATEGY, '--output-dir', str(output),
                   '--sonic-input', str(self.sonic), '--sonic-adapter', str(self.adapter), '--strict']
        run = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        report = cr.read(output / 'report.json')
        self.assertTrue(report['full_tables_allowed'])
        receipt = confirmation_report.collect(output / 'report.json', self.root / 'manifest.json',
                                               self.root / 'cli-four-method-tables')
        self.assertTrue(all(cr.sha(item['path']) == item['sha256'] for item in receipt['outputs']))
        self.assertEqual(len(cr.read(self.root / 'cli-four-method-tables/window_audit.json')), 504)

    def test_partial_missing_sonic_does_not_release_tables(self):
        report = self.collect()
        self.assertEqual(report['status'], 'partial')
        self.assertTrue(report['native_matrix_complete'])
        self.assertEqual(len([p for p in report['missing_invalid'] if p['state'] == 'missing']), 72)
        self.assertEqual(report['methods'], {})
        output = self.root / 'partial-output'
        cr.write_outputs(report, output)
        self.assertFalse((output / 'full_tables.md').exists())
        with patch.object(cr, 'collect', return_value=report):
            result = cr.main(['--campaign', str(self.path), '--snapshot', str(SNAPSHOT),
                              '--expected-strategy', HELPER.CONFIRMATION_STRATEGY, '--strict',
                              '--output-dir', str(self.root / 'strict-output')])
        self.assertEqual(result, 2)

    def test_other_snapshot_isolated_from_loaded_public_package(self):
        frozen = self.root / 'isolated-snapshot'
        sources = {
            'mimicx/__init__.py': '', 'mimicx/experiments/__init__.py': '',
            'mimicx/policy_recovery/__init__.py': '',
            'mimicx/experiments/motion_study.py':
                'def verify_campaign(campaign):\n    raise RuntimeError("frozen snapshot selected")\n',
            'mimicx/experiments/motion_study_loop.py': '', 'mimicx/policy_recovery/metrics.py': '',
        }
        for name, content in sources.items():
            target = frozen / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        before = list(sys.path)
        report = cr.collect(self.path, frozen, HELPER.CONFIRMATION_STRATEGY)
        self.assertFalse(report['native_matrix_complete'])
        self.assertTrue(any('frozen snapshot selected' in p['reason'] for p in report['missing_invalid']))
        self.assertEqual(sys.path, before)
        self.assertEqual(Path(STUDY.__file__).resolve().parents[2], SNAPSHOT)
        self.assertFalse(list(frozen.rglob('*.pyc')))

    def test_public_cli_native_only_needs_no_sonic_adapter(self):
        output = self.root / 'cli-native-only'
        command = [sys.executable, '-B', str(SNAPSHOT / 'scripts/experiments/collect_results.py'),
                   '--campaign', str(self.path), '--snapshot', str(SNAPSHOT),
                   '--expected-strategy', HELPER.CONFIRMATION_STRATEGY,
                   '--output-dir', str(output), '--strict']
        run = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(run.returncode, 2, run.stderr)
        report = cr.read(output / 'report.json')
        self.assertTrue(report['native_matrix_complete'])
        self.assertFalse(report['full_tables_allowed'])
        self.assertEqual(report['budget']['validated']['trained_holdout_rollouts'], 432)
        self.assertTrue(all(p['scope'] == 'sonic' and p['state'] == 'missing'
                            for p in report['missing_invalid']))
        self.assertFalse((output / 'full_tables.md').exists())

    def test_missing_done_never_completed(self):
        with changed(self.folder / 'done.json'):
            report = self.collect()
        row = next(r for r in report['jobs'] if r['job'] == self.job['id'])
        self.assertEqual(row['state'], 'missing')
        self.assertEqual(report['budget']['validated']['arms'], 71)
        self.assertEqual(len(report['paired_changes']), 144)
        self.assertTrue(any(r['state'] == 'unavailable' for r in report['paired_changes']))

    def test_bad_receipt_hash(self):
        with changed(self.folder / 'report.json', b'{}'):
            with self.assertRaisesRegex(RuntimeError, 'Corrupt'):
                self.collector.job(self.job, self.campaign['tasks'][0])

    def test_empty_receipt_artifacts_rejected(self):
        receipt = cr.read(self.folder / 'done.json')
        receipt['artifacts'] = {}
        with changed(self.folder / 'done.json', json.dumps(receipt).encode()):
            with self.assertRaisesRegex(ValueError, 'missing required artifacts'):
                self.collector.job(self.job, self.campaign['tasks'][0])

    def test_missing_donor_invalidates_all_dependents(self):
        donor = self.folder.parent / 'warmstart/done.json'
        with changed(donor):
            report = self.collect()
        self.assertEqual(report['budget']['validated']['donors'], 23)
        self.assertEqual(report['budget']['validated']['arms'], 69)
        self.assertEqual(sum(r['state'] == 'invalid' for r in report['jobs']), 3)
        self.assertFalse(report['native_matrix_complete'])

    def test_frozen_input_changed_blocks_matrix(self):
        motion = Path(self.campaign['tasks'][0]['motion_path'])
        with changed(motion, b'changed reference'):
            report = self.collect()
        self.assertEqual(report['budget']['validated']['arms'], 0)
        self.assertTrue(all(r['state'] == 'unvalidated' for r in report['jobs']))
        self.assertTrue(any(p['scope'] == 'campaign' for p in report['missing_invalid']))

    def test_group_seed_mismatch_rejected(self):
        folder = self.folder / 'validation/slot00'
        summary = cr.read(folder / 'summary.json')
        summary['seeds'] = list(reversed(summary['seeds']))
        collector = cr.Collector(self.path, STUDY, METRICS, LOOP)
        with changed(folder / 'summary.json', json.dumps(summary).encode()):
            with self.assertRaisesRegex(ValueError, 'Group identity'):
                collector.group(folder, self.campaign['tasks'][0], self.folder / 'model_0.pt', cr.SELECTION, .005)

    def test_common_numerical_and_physical_checks(self):
        rollout = self.folder / 'holdout/nominal/seed21001/rollout'
        path = rollout / 'native_actual.npz'
        with np.load(path, allow_pickle=False) as stream:
            arrays = {key: stream[key] for key in stream.files}
        arrays['body_pos'][0, 0, 0] = np.nan
        with changed(path, path.read_bytes()):
            np.savez(path, **arrays)
            with self.assertRaisesRegex(ValueError, 'full-body numerical'):
                self.collector.common(rollout, self.campaign['tasks'][0])

    def test_clock_and_reset_mask_are_not_repaired(self):
        rollout = self.folder / 'holdout/nominal/seed21001/rollout'
        path = rollout / 'native_actual.npz'
        with np.load(path, allow_pickle=False) as stream:
            original = {key: stream[key] for key in stream.files}
        for key, value in [('timestamps', np.array([0., .02, .02])),
                           ('timestamps', np.array([0., .02, .05])),
                           ('valid_mask', np.array([True, False, True]))]:
            with self.subTest(key=key, value=value):
                with changed(path, path.read_bytes()):
                    np.savez(path, **dict(original, **{key: value}))
                    with self.assertRaises(ValueError):
                        self.collector.common(rollout, self.campaign['tasks'][0])
        path = rollout / 'common_metrics.json'
        common = cr.read(path)
        common['complete_motion'] = False
        common['first_failure_frame'] = 1
        with changed(path, json.dumps(common).encode()):
            with self.assertRaisesRegex(ValueError, 'Physical completion disagrees'):
                self.collector.common(rollout, self.campaign['tasks'][0])
        common = cr.read(path)
        common['full_horizon']['world_body_error_m']['mean'] = .99
        with changed(path, json.dumps(common).encode()):
            with self.assertRaisesRegex(ValueError, 'trace/summary mismatch'):
                self.collector.common(rollout, self.campaign['tasks'][0])

    def test_forged_qualification_recomputed(self):
        report = cr.read(self.folder / 'report.json')
        report['qualified'] = True
        done = cr.read(self.folder / 'done.json')
        with changed(self.folder / 'report.json', json.dumps(report).encode()):
            done['artifacts']['report.json'] = cr.sha(self.folder / 'report.json')
            with changed(self.folder / 'done.json', json.dumps(done).encode()):
                with self.assertRaisesRegex(ValueError, 'Incorrect report qualified'):
                    self.collector.job(self.job, self.campaign['tasks'][0])

    def test_holdout_selection_forbidden(self):
        selection = cr.read(self.folder / 'selection.json')
        selection['uses_holdout'] = True
        done = cr.read(self.folder / 'done.json')
        with changed(self.folder / 'selection.json', json.dumps(selection).encode()):
            done['artifacts']['selection.json'] = cr.sha(self.folder / 'selection.json')
            with changed(self.folder / 'done.json', json.dumps(done).encode()):
                with self.assertRaisesRegex(ValueError, 'Selection used holdout'):
                    self.collector.job(self.job, self.campaign['tasks'][0])

    def test_matrix_and_seed_validation(self):
        for mutate in (lambda c: c['jobs'].pop(), lambda c: c['jobs'].append(c['jobs'][0]),
                       lambda c: c['runtime']['validation_protocol']['holdout']['confirmation'].update(nominal=cr.SELECTION)):
            campaign = copy.deepcopy(self.campaign)
            mutate(campaign)
            with self.assertRaises(ValueError):
                cr.validate_campaign(campaign, HELPER.CONFIRMATION_STRATEGY)

    def test_sonic_duplicate_nonfinite_cost_and_missing(self):
        original = cr.read(self.sonic)
        for field, value in [('world_mean_cm', -1), ('world_p95_cm', float('nan')),
                             ('training_updates', 3000), ('complete_motion', 1), ('train_seed', 101)]:
            package = copy.deepcopy(original)
            package['rows'][0][field] = value
            with changed(self.sonic, json.dumps(package).encode()):
                rows, raw, problems, _ = cr.sonic_rows(self.sonic, self.adapter, self.campaign)
            self.assertEqual(len(rows), 71)
            self.assertTrue(any(p['state'] == 'invalid' for p in problems))
            self.assertEqual(raw['received'], 79)
        package = copy.deepcopy(original)
        package['rows'].append(package['rows'][0])
        with changed(self.sonic, json.dumps(package).encode()):
            rows, _, problems, _ = cr.sonic_rows(self.sonic, self.adapter, self.campaign)
        self.assertEqual(len(rows), 71)
        self.assertTrue(any('Duplicate' in p['reason'] for p in problems))

    def test_sonic_evidence_and_missing_input(self):
        evidence = self.root / 'sonic-evidence.txt'
        with changed(evidence, b'tampered'):
            rows, _, problems, _ = cr.sonic_rows(self.sonic, self.adapter, self.campaign)
        self.assertEqual(rows, [])
        self.assertEqual(sum(p['state'] == 'invalid' for p in problems), 72)
        rows, _, problems, _ = cr.sonic_rows(self.root / 'missing.json', self.adapter, self.campaign)
        self.assertEqual(rows, [])
        self.assertTrue(any(p['scope'] == 'sonic_adapter' for p in problems))

    def test_timing_ignores_nested_and_unions_overlaps(self):
        root = self.root / 'timing-fixture'
        jobs = [dict(id='a'), dict(id='b')]
        for jid, start, end in [('a', 10., 20.), ('b', 15., 30.)]:
            save(root / jid / 'attempts/1.json', dict(started=start, finished=end, job_id=jid,
                 gpu_uuid='GPU-fixture', assigned_gpu_wall_seconds=end-start))
            save(root / jid / 'stage0/attempts/1.json', dict(started=0, finished=10000))
        timing = cr.timing(root, jobs)
        self.assertEqual(timing['observed_elapsed_span_seconds'], 20.)
        self.assertEqual(timing['observed_busy_wall_union_seconds'], 20.)
        self.assertEqual(timing['observed_assigned_gpu_union_seconds'], 20.)
        self.assertEqual(timing['measured_attempts'], 2)

    def test_no_complete_case_filter_and_zero_denominator(self):
        self.assertIsNone(cr.strict_mean([1., None, 2.]))
        self.assertEqual(cr.gain(1., 0.)['undefined_reason'], 'zero_control_denominator')
        self.assertEqual(cr.gain(None, 1.)['undefined_reason'], 'censored_metric')


if __name__ == '__main__':
    unittest.main(verbosity=2)
