"""CPU state-capture/common-scorer integration; no controller or GPU launch."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from mimicx.experiments.motion_study_scoring import align_rotation_csv, capture_class, common_protocol, score_capture


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)
    def __getitem__(self, index):
        return Tensor(self.value[index])
    def __or__(self, other):
        return Tensor(self.value | other.value)
    def any(self):
        return Tensor(self.value.any())
    def detach(self):
        return self
    def cpu(self):
        return self
    def numpy(self):
        return self.value
    def item(self):
        return self.value.item()


class CaptureTests(unittest.TestCase):
    def test_phase_zero_link_velocities_named_order_and_native_reset_flag(self):
        class Env:
            step_dt = .02
            def __init__(self):
                self.count = 0
                self.data = SimpleNamespace(joint_pos=Tensor([[1., 2.]]), joint_vel=Tensor([[3., 4.]]),
                    body_link_pos_w=Tensor([[[1., 0, 0], [2., 0, 0]]]),
                    body_link_quat_w=Tensor([[[1., 0, 0, 0], [1., 0, 0, 0]]]),
                    body_link_lin_vel_w=Tensor([[[5., 0, 0], [6., 0, 0]]]),
                    body_link_ang_vel_w=Tensor([[[7., 0, 0], [8., 0, 0]]]))
                self.scene = {'robot': SimpleNamespace(data=self.data, joint_names=['a', 'b'], body_names=['hand', 'pelvis'])}
            def step(self, action):
                self.count += 1
                self.data.joint_pos = Tensor([[self.count + 1., self.count + 2.]])
                return None, None, Tensor([self.count == 2]), Tensor([False]), {}

        samples = []
        cls = capture_class(Env, {'joint_names': ['b', 'a'], 'body_names': ['pelvis', 'hand']}, samples)
        env = cls()
        env.step(None)
        env.step(None)
        self.assertEqual(len(samples), 3)
        np.testing.assert_array_equal(samples[0][0]['joint_pos'], [2, 1])
        np.testing.assert_array_equal(samples[0][0]['body_lin_vel'][:, 0], [6, 5])
        np.testing.assert_array_equal(samples[0][0]['body_ang_vel'][:, 0], [8, 7])
        self.assertEqual([s[1] for s in samples], [False, False, True])
        np.testing.assert_array_equal(samples[2][0]['joint_pos'], [4, 3])
        env.step_dt = .01
        with self.assertRaisesRegex(ValueError, '50 Hz'):
            env.step(None)


class RotationClockTests(unittest.TestCase):
    def fixture(self, frames=2):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        output = Path(temporary.name)
        rows = [dict(step=i, done_any=False, robot_anchor_x=.04, robot_anchor_y=0., robot_anchor_z=1.,
                     ref_anchor_x=0., ref_anchor_y=0., ref_anchor_z=1., body_pos_error_max=.04,
                     error_anchor_rot=.23) for i in range(1, frames)]
        with (output / 'steps.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        actual = {'body_quat': np.tile([np.cos(.23/2), 0, 0, np.sin(.23/2)], (frames, 1, 1))}
        target = {'body_quat': np.tile([np.cos(.04/2), 0, 0, np.sin(.04/2)], (frames, 1, 1))}
        target['body_quat'][0, 0] = [1., 0, 0, 0]
        return output, rows, actual, target

    def test_same_frame_rotation_controls_gate_and_keeps_native_raw_values(self):
        from mimicx.experiments.motion_study_gate import accept
        output, rows, actual, target = self.fixture()
        original_bytes = (output / 'steps.csv').read_bytes()
        align_rotation_csv(output, actual, target, ['torso_link'], np.ones(2, dtype=bool), 'torso_link')
        with (output / 'steps.csv').open() as stream:
            aligned = list(csv.DictReader(stream))
        self.assertAlmostEqual(float(aligned[0]['error_anchor_rot']), .19)
        self.assertEqual(float(aligned[0]['error_anchor_rot_native_cached']), .23)
        self.assertEqual((output / 'steps.native_raw.csv').read_bytes(), original_bytes)
        incumbent = copy.deepcopy(rows)
        incumbent[0]['done_any'] = True
        self.assertFalse(accept([incumbent] * 3, [rows] * 3, 1)['accepted'])
        self.assertTrue(accept([incumbent] * 3, [aligned] * 3, 1)['accepted'])
        audit = json.loads((output / 'rotation_clock.json').read_text())
        self.assertEqual(audit['aligned_valid_rows'], 1)
        self.assertAlmostEqual(audit['maximum_cached_vs_aligned_delta_rad'], .04)

    def test_reset_tail_is_not_corrected_or_reused_as_evidence(self):
        output, rows, actual, target = self.fixture(4)
        align_rotation_csv(output, actual, target, ['torso_link'], np.array([True, True, False, False]), 'torso_link')
        with (output / 'steps.csv').open() as stream:
            aligned = list(csv.DictReader(stream))
        self.assertEqual([r['rotation_clock_aligned'] for r in aligned], ['True', 'False', 'False'])
        self.assertEqual(float(aligned[1]['error_anchor_rot']), .23)
        self.assertEqual(float(aligned[2]['error_anchor_rot']), .23)
        with self.assertRaises(FileExistsError):
            align_rotation_csv(output, actual, target, ['torso_link'], np.ones(4, dtype=bool), 'torso_link')


@unittest.skipUnless(importlib.util.find_spec('mujoco'), 'CPU MuJoCo environment required')
class PhysicalScoringTests(unittest.TestCase):
    def setUp(self):
        from tests.test_sonic_study import PhysicsTests
        self.fixture = PhysicsTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        motion = self.fixture.motion
        self.task = dict(frames=len(motion['joint_pos']), motion_path=str(self.fixture.motion_path))
        self.campaign = dict(reference_contract={key: list(motion[key]) for key in ('joint_names', 'body_names')},
            runtime=dict(common_model_xml=str(self.fixture.xml), common_protocol=common_protocol()))
        fields = dict(joint_pos='joint_pos', joint_vel='joint_vel', body_pos='body_pos_w', body_quat='body_quat_w',
                      body_lin_vel='body_lin_vel_w', body_ang_vel='body_ang_vel_w')
        self.samples = [({key: motion[source][i].copy() for key, source in fields.items()}, False)
                        for i in range(self.task['frames'])]

    def score(self):
        output = self.fixture.root / 'score'
        output.mkdir(exist_ok=True)
        return score_capture(self.campaign, self.task, self.samples, output)

    def test_actual_common_scorer_inclusive_full_motion_and_zero_error(self):
        result = self.score()
        self.assertTrue(result['complete_motion'])
        self.assertEqual(result['frames'], self.task['frames'])
        self.assertEqual(result['full_horizon']['world_body_error_m']['mean'], 0.)
        result_json = json.loads((self.fixture.root / 'score/common_metrics.json').read_text())
        self.assertNotIn('_steps', result_json)
        self.assertEqual(result_json['native_termination'], result_json['native_termination_source'])
        self.assertIn('MjLab', result_json['native_termination'])
        self.assertIn('automatic reset', result_json['native_termination'])
        self.assertNotIn('no automatic reset', result_json['native_termination'])
        with np.load(self.fixture.root / 'score/native_actual.npz') as capture:
            np.testing.assert_allclose(capture['timestamps'], np.arange(self.task['frames']) / 50.)
            self.assertTrue(capture['valid_mask'].all())

    def test_post_reset_tail_never_counts_as_physical_completion(self):
        state, _ = self.samples[4]
        self.samples[4] = (state, True)
        for state, _ in self.samples[4:]:
            state['body_pos'] += 9.
        result = self.score()
        self.assertFalse(result['complete_motion'])
        self.assertEqual(result['native_failure_frame'], 4)
        self.assertEqual(result['valid_physical_frames'], 4)
        self.assertIsNone(result['full_horizon'])
        self.assertIsNone(result['common_prefix'])
        self.assertEqual(result['pre_failure']['world_body_error_m']['mean'], 0.)

    def test_common_protocol_or_union_persistence_is_backdated(self):
        names = self.campaign['reference_contract']['body_names']
        root = names.index('pelvis')
        for i in (1, 3, 5):
            self.samples[i][0]['body_pos'][root, 2] += .3
        for i in (2, 4):
            self.samples[i][0]['body_quat'][root] = [np.cos(.6), np.sin(.6), 0, 0]
        result = self.score()
        self.assertEqual(result['first_failure_frame'], 1)
        self.assertFalse(result['complete_motion'])

    def test_short_or_nonfinite_capture_is_invalid_not_success(self):
        self.samples.pop()
        with self.assertRaisesRegex(ValueError, 'inclusive'):
            self.score()
        self.samples.append(copy.deepcopy(self.samples[-1]))
        self.samples[2][0]['joint_pos'][0] = np.nan
        with self.assertRaisesRegex(ValueError, 'invalid common-metric'):
            self.score()


if __name__ == '__main__':
    unittest.main()
