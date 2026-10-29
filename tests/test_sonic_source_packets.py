"""Adversarial regressions for causal SONIC startup packet evidence."""
import copy
import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mimicx.experiments import sonic_study as study
from mimicx.experiments.sonic_helpers import StartupPublications


def fixture():
    q = np.linspace(-1.3, 1.1, 29).astype(np.float32).astype(float)
    dq = np.linspace(-12., 12., 29).astype(np.float32).astype(float)
    base = np.array([1., 0., 0., 0.])
    torso = np.array([np.cos(.1), 0., np.sin(.1), 0.], dtype=np.float32).astype(float)
    gyro = np.array([.125, -.25, .5])
    accel = np.array([.25, .5, 9.75])
    evidence = {'joint_names': np.asarray(study.JOINT_NAMES), 'motor_indices': np.arange(29)}
    for name in ('lowstate', 'torso'):
        for key, value in dict(sequence=0, source_time=2., source_monotonic=10.,
                               write_start=10.002, write_end=10.003,
                               write_success=True).items():
            evidence[name + '_' + key] = np.asarray([value])
    for key, value in dict(tick=2000, mode_machine=0, wire_motor_count=35, q=q, dq=dq, quat=base,
                           physical_q=q, physical_dq=dq,
                           physical_base_quat=base).items():
        evidence['lowstate_' + key] = np.asarray([value])
    for key, value in dict(quat=torso, gyro=gyro, accel=accel,
                           physical_torso_quat=torso).items():
        evidence['torso_' + key] = np.asarray([value])
    controller = dict(q=np.round(q, 9), dq=np.round(dq, 9), base_quat=base,
                      torso_quat=np.round(torso, 9), torso_gyro=gyro,
                      torso_accel=accel, control_read_monotonic=10.010,
                      log_monotonic=10.012, lowstate_tick=2000,
                      lowstate_received_monotonic=10.006,
                      torso_received_monotonic=10.007, mode_machine=0)
    bounds = dict(reset_monotonic=9.95, reset_simulation_time=1.95,
                  original_header=0, max_age_s=.02)
    append_packet(evidence, 'lowstate', sequence=1, source_time=2.005,
                  source_monotonic=10.015, write_start=10.016, write_end=10.017,
                  tick=2005, q=q + .1)
    append_packet(evidence, 'torso', sequence=1, source_time=2.005,
                  source_monotonic=10.015, write_start=10.016, write_end=10.017,
                  gyro=gyro + .1)
    return evidence, controller, bounds


def append_packet(evidence, prefix, **changes):
    for key in tuple(evidence):
        if key.startswith(prefix + '_'):
            value = changes.get(key[len(prefix) + 1:], evidence[key][-1])
            evidence[key] = np.concatenate((evidence[key], np.asarray([value])), axis=0)


class SourcePacketAdversarialTests(unittest.TestCase):
    def evaluate(self, evidence, controller, bounds):
        return study.causal_publication_audit(evidence, controller, **bounds)

    def test_accept_fresh_fast_motion_without_current_time_comparison(self):
        evidence, controller, bounds = fixture()
        # At 12 rad/s, 10ms of real evolution exceeds the old 0.02 rad guard.
        drift = np.sqrt(np.mean((controller['dq'] * .010) ** 2))
        self.assertGreater(drift, .02)
        before = copy.deepcopy((evidence, controller, bounds))
        result = self.evaluate(evidence, controller, bounds)
        self.assertIsInstance(result, dict)
        for actual, expected in zip((evidence, controller, bounds), before):
            for key in actual:
                np.testing.assert_array_equal(actual[key], expected[key])

    def test_accept_distinct_independent_torso_source(self):
        evidence, controller, bounds = fixture()
        evidence['torso_source_time'][0] = 2.002
        evidence['torso_source_monotonic'][0] = 10.005
        evidence['torso_write_start'][0] = 10.006
        evidence['torso_write_end'][0] = 10.0065
        self.evaluate(evidence, controller, bounds)

    def test_accept_unique_prior_source_despite_identical_future_payload(self):
        evidence, controller, bounds = fixture()
        # A packet published after the read cannot be its source, even if equal.
        evidence['torso_gyro'][1] = evidence['torso_gyro'][0]
        self.evaluate(evidence, controller, bounds)

    def test_accept_write_return_after_used_receipt(self):
        evidence, controller, bounds = fixture()
        for prefix in ('lowstate', 'torso'):
            evidence[prefix + '_write_end'][0] = 10.014
        self.evaluate(evidence, controller, bounds)

    def repeated_fresh_torso(self):
        evidence, controller, bounds = fixture()
        evidence['torso_gyro'][1] = evidence['torso_gyro'][0]
        evidence['torso_source_time'][1] = 2.004
        evidence['torso_source_monotonic'][1] = 10.004
        evidence['torso_write_start'][1] = 10.005
        evidence['torso_write_end'][1] = 10.006
        return evidence, controller, bounds

    def test_accept_repeated_static_torso_only_with_all_candidates_fresh(self):
        evidence, controller, bounds = self.repeated_fresh_torso()
        result = self.evaluate(evidence, controller, bounds)
        torso = result['publications']['torso']
        self.assertEqual(torso['candidate_count'], 2)
        self.assertEqual(torso['candidate_sequences'], [0, 1])
        self.assertIsNone(torso['sequence'])
        self.assertTrue(torso['all_candidates_fresh'])
        self.assertAlmostEqual(torso['source_age_s'], .010)
        self.assertFalse(result['rollout_clock_fitted'])

    def test_reject_one_aged_identical_torso_despite_fresh_candidate(self):
        evidence, controller, bounds = self.repeated_fresh_torso()
        evidence['torso_source_time'][0] = 1.98
        evidence['torso_source_monotonic'][0] = 9.98
        evidence['torso_write_start'][0] = 9.981
        evidence['torso_write_end'][0] = 9.982
        with self.assertRaisesRegex(ValueError, 'overage'):
            self.evaluate(evidence, controller, bounds)

    def test_reject_any_candidate_torso_physical_mapping_failure(self):
        for candidate in (0, 1):
            with self.subTest(candidate=candidate):
                evidence, controller, bounds = self.repeated_fresh_torso()
                evidence['torso_physical_torso_quat'][candidate] = [0., 0., 0., 1.]
                with self.assertRaisesRegex(ValueError, 'mapping guard'):
                    self.evaluate(evidence, controller, bounds)

    def test_capture_retains_prereset_payloads_for_replay_detection(self):
        evidence, _, _ = fixture()
        low = SimpleNamespace(tick=0, mode_machine=0,
                              motor_state=[SimpleNamespace(q=q, dq=dq) for q, dq in
                                           zip(evidence['lowstate_q'][0], evidence['lowstate_dq'][0])]
                              + [SimpleNamespace(q=0., dq=0.) for _ in range(6)],
                              imu_state=SimpleNamespace(quaternion=evidence['lowstate_quat'][0]))
        torso = SimpleNamespace(quaternion=evidence['torso_quat'][0],
                                gyroscope=evidence['torso_gyro'][0],
                                accelerometer=evidence['torso_accel'][0])
        bridge = SimpleNamespace(low_state_puber=SimpleNamespace(Write=lambda sample: True),
                                 torso_imu_puber=SimpleNamespace(Write=lambda sample: True))

        def publish(obs):
            low.tick = int(obs['time'] * 1000)
            bridge.low_state_puber.Write(low)
            bridge.torso_imu_puber.Write(torso)

        bridge.PublishLowState = publish
        physical = {key.removeprefix('lowstate_'): values[0].copy() for key, values in evidence.items()
                    if key.startswith('lowstate_physical_')}
        physical['physical_torso_quat'] = evidence['torso_physical_torso_quat'][0].copy()
        recorder = StartupPublications(bridge, lambda: physical, joint_names=study.JOINT_NAMES)
        with patch('mimicx.experiments.sonic_helpers.time.monotonic', return_value=9.90):
            bridge.PublishLowState({'time': 1.90})
        recorder.start(dict(monotonic_time=9.95, simulation_time=1.95))
        with patch('mimicx.experiments.sonic_helpers.time.monotonic', return_value=10.):
            bridge.PublishLowState({'time': 2.})
        self.assertEqual(len(recorder.rows['torso']), 2,
                         'pre-reset torso must remain available for duplicate identity detection')

    def test_reject_prereset_source_republished_after_reset(self):
        for prefix in ('lowstate', 'torso'):
            with self.subTest(prefix=prefix):
                evidence, controller, bounds = fixture()
                evidence[prefix + '_source_monotonic'][0] = 9.94
                evidence[prefix + '_source_time'][0] = 1.94
                with self.assertRaises(ValueError):
                    self.evaluate(evidence, controller, bounds)

    def test_reject_old_source_with_fresh_receipt(self):
        for prefix in ('lowstate', 'torso'):
            with self.subTest(prefix=prefix):
                evidence, controller, bounds = fixture()
                evidence[prefix + '_source_monotonic'][0] = 9.98
                with self.assertRaises(ValueError):
                    self.evaluate(evidence, controller, bounds)

    def test_reject_future_source_and_publication(self):
        for prefix in ('lowstate', 'torso'):
            for field in ('source_monotonic', 'write_start'):
                with self.subTest(prefix=prefix, field=field):
                    evidence, controller, bounds = fixture()
                    evidence[prefix + '_' + field][0] = 10.02
                    evidence[prefix + '_write_end'][0] = 10.021
                    with self.assertRaises(ValueError):
                        self.evaluate(evidence, controller, bounds)

    def test_reject_invalid_receipt_read_order(self):
        for key, value in (('lowstate_received_monotonic', 10.001),
                           ('torso_received_monotonic', 10.001),
                           ('lowstate_received_monotonic', 10.011),
                           ('torso_received_monotonic', 10.011),
                           ('control_read_monotonic', 10.013)):
            with self.subTest(key=key, value=value):
                evidence, controller, bounds = fixture()
                controller[key] = value
                with self.assertRaises(ValueError):
                    self.evaluate(evidence, controller, bounds)

    def test_reject_unsuccessful_publication(self):
        for prefix in ('lowstate', 'torso'):
            with self.subTest(prefix=prefix):
                evidence, controller, bounds = fixture()
                evidence[prefix + '_write_success'][0] = False
                with self.assertRaises(ValueError):
                    self.evaluate(evidence, controller, bounds)

    def test_reject_duplicate_source_in_full_prior_history(self):
        for prefix in ('lowstate', 'torso'):
            with self.subTest(prefix=prefix):
                evidence, controller, bounds = fixture()
                for key in tuple(evidence):
                    if key.startswith(prefix + '_') and key != prefix + '_sequence':
                        evidence[key][1] = evidence[key][0]
                for key, value in (('source_monotonic', 9.90), ('source_time', 1.9),
                                   ('write_start', 9.901), ('write_end', 9.902)):
                    evidence[prefix + '_' + key][0] = value
                with self.assertRaises(ValueError):
                    self.evaluate(evidence, controller, bounds)

    def test_reject_payload_mismatch_in_each_independent_field(self):
        for key in ('q', 'dq', 'base_quat', 'torso_quat', 'torso_gyro', 'torso_accel'):
            with self.subTest(key=key):
                evidence, controller, bounds = fixture()
                controller[key] = controller[key].copy()
                controller[key][-1] += .01
                with self.assertRaises(ValueError):
                    self.evaluate(evidence, controller, bounds)

    def test_reject_wrong_tick_or_unrestored_header(self):
        for mutation in ('tick', 'header'):
            with self.subTest(mutation=mutation):
                evidence, controller, bounds = fixture()
                if mutation == 'tick':
                    controller['lowstate_tick'] = 1995
                else:
                    evidence['lowstate_mode_machine'][0] = 1
                with self.assertRaises(ValueError):
                    self.evaluate(evidence, controller, bounds)

    def test_reject_nonfinite_clocks_and_payload(self):
        for key in ('control_read_monotonic', 'log_monotonic',
                    'lowstate_received_monotonic', 'torso_received_monotonic'):
            with self.subTest(key=key):
                evidence, controller, bounds = fixture()
                controller[key] = float('nan')
                with self.assertRaises(ValueError):
                    self.evaluate(evidence, controller, bounds)
        for key in ('lowstate_source_time', 'lowstate_source_monotonic',
                    'torso_write_end', 'lowstate_q', 'torso_quat'):
            with self.subTest(key=key):
                evidence, controller, bounds = fixture()
                evidence[key].flat[0] = float('nan')
                with self.assertRaises(ValueError):
                    self.evaluate(evidence, controller, bounds)

    def test_reject_physical_mapping_error_at_source(self):
        for prefix, field in (('lowstate', 'physical_q'),
                              ('lowstate', 'physical_base_quat'),
                              ('torso', 'physical_torso_quat')):
            with self.subTest(prefix=prefix, field=field):
                evidence, controller, bounds = fixture()
                if field == 'physical_q':
                    evidence[prefix + '_' + field][0] += .03
                else:
                    evidence[prefix + '_' + field][0] = [0., 0., 0., 1.]
                with self.assertRaises(ValueError):
                    self.evaluate(evidence, controller, bounds)

    def test_reject_physical_dq_mismatch_even_when_wire_and_controller_agree(self):
        for mutation in ('all_joints', 'one_joint', 'permuted_joints'):
            with self.subTest(mutation=mutation):
                evidence, controller, bounds = fixture()
                if mutation == 'all_joints':
                    evidence['lowstate_physical_dq'][0] += 100.
                elif mutation == 'one_joint':
                    evidence['lowstate_physical_dq'][0, 7] += .001
                else:
                    evidence['lowstate_physical_dq'][0] = evidence['lowstate_physical_dq'][0, ::-1]
                np.testing.assert_allclose(evidence['lowstate_dq'][0], controller['dq'], atol=2e-8, rtol=0)
                with self.assertRaises(ValueError):
                    self.evaluate(evidence, controller, bounds)

    def test_accept_physical_dq_difference_removed_by_float32_wire_cast(self):
        evidence, controller, bounds = fixture()
        evidence['lowstate_physical_dq'][0, 0] = -12. + 1e-7
        self.assertGreater(abs(evidence['lowstate_physical_dq'][0, 0] - controller['dq'][0]), 2e-8)
        self.assertEqual(float(np.float32(evidence['lowstate_physical_dq'][0, 0])), controller['dq'][0])
        self.evaluate(evidence, controller, bounds)

    def test_physical_dq_guard_uses_absolute_not_relative_or_rmse_tolerance(self):
        for joint, delta in ((14, 1e-8), (14, 3e-8), (0, 1e-5)):
            with self.subTest(joint=joint, delta=delta):
                evidence, controller, bounds = fixture()
                evidence['lowstate_physical_dq'][0, joint] += delta
                error = abs(float(np.float32(evidence['lowstate_physical_dq'][0, joint]))
                            - evidence['lowstate_dq'][0, joint])
                if error <= 2e-8:
                    self.evaluate(evidence, controller, bounds)
                else:
                    with self.assertRaises(ValueError):
                        self.evaluate(evidence, controller, bounds)


@unittest.skipUnless(importlib.util.find_spec('mujoco'), 'CPU MuJoCo environment required')
class StartupContractEvidenceTests(unittest.TestCase):
    def setUp(self):
        from tests import test_sonic_study as support

        self.support = support.PhysicsTests()
        self.addCleanup(self.support.doCleanups)
        self.support.setUp()
        self.root = self.support.root
        self.sources = [self.root / name for name in (
            'gear_sonic/utils/mujoco_sim/simulator_factory.py',
            'gear_sonic/utils/network/unitree_bridge.py',
            'gear_sonic/__init__.py',
            'gear_sonic/utils/rotation.py',
            'gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp',
            'gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/file_sink.cpp',
            'gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/state_logger.hpp',
        )]
        for path in self.sources:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('// source identity fixture\n' if path.suffix != '.py' else '# source identity fixture\n')
        output = self.root / 'prepared'
        with patch.object(study, 'preflight', return_value={'status': 'blocked'}), \
                patch.object(study.subprocess, 'Popen') as launch:
            runs = study.prepare(self.support.manifest(), self.support.config(), output,
                                 seed=1001, perturb={'mode': 'nominal'})
        launch.assert_not_called()
        self.path = output / runs[0]
        self.run = study.verify_run(self.path)

    def test_v4_contract_binds_causal_evidence_and_deploy_binary(self):
        contract = self.run['startup_contract']
        expected = dict(
            schema='mimicx.sonic-startup.v4',
            deploy_binary_sha256=study.sha(self.run['runtime']['deploy_binary']),
            causal_source_schema='mimicx.sonic-causal-startup.v1',
            torso_identity='all causally prior exact-payload candidates must be fresh and post-reset',
            velocity_mapping='physical joint velocity rounded to float32; 2e-8 absolute tolerance',
            telemetry_storage='run_directory; external staging requires separately frozen provenance',
        )
        for key, value in expected.items():
            with self.subTest(field=key):
                self.assertEqual(contract.get(key), value)
        self.assertEqual(self.run['startup_contract_sha256'], study.protocol_sha256(contract))

    def test_removing_v4_evidence_fields_invalidates_prepared_run(self):
        original = self.path.read_bytes()
        for field in ('deploy_binary_sha256', 'causal_source_schema', 'torso_identity',
                      'velocity_mapping', 'telemetry_storage'):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.run)
                self.assertIn(field, changed['startup_contract'])
                del changed['startup_contract'][field]
                study.dump(self.path, changed)
                try:
                    with self.assertRaisesRegex(ValueError, 'prepared run contract changed'):
                        study.verify_run(self.path)
                finally:
                    self.path.write_bytes(original)
        study.verify_run(self.path)

    def test_full_source_guard_rejects_drift_outside_selected_python_subtrees(self):
        for source in self.sources:
            with self.subTest(source=str(source.relative_to(self.root))):
                original = source.read_bytes()
                source.write_bytes(original + b'\n// changed after preparation\n')
                try:
                    with self.assertRaises(ValueError):
                        study.verify_run(self.path)
                finally:
                    source.write_bytes(original)
                study.verify_run(self.path)


if __name__ == '__main__':
    unittest.main()
