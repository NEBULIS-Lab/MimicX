import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mimicx.datasets.motion_study import (
    CanonicalG1, load_eligible_manifest, load_shortlist, prepare, read_motion,
    sha256, timing_check, window_motion, rebuild_csv, export_isaac_pilot,
    freeze_ground_contact_protocol, protocol_digest, GROUND_PROTOCOL_BLOCKER,
)

CONFIG = Path(__file__).resolve().parents[1] / "mimicx/configs/datasets/motion_study_v1.json"


def motion(n=11):
    q = np.zeros((n, 30, 4))
    q[..., 0] = 1
    return dict(fps=np.array([50.]), joint_pos=np.zeros((n, 29)),
                joint_vel=np.zeros((n, 29)), body_pos_w=np.zeros((n, 30, 3)),
                body_quat_w=q, body_lin_vel_w=np.zeros((n, 30, 3)),
                body_ang_vel_w=np.zeros((n, 30, 3)))


class MotionStudyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_all_is_complete(self):
        self.assertEqual(len(load_shortlist(CONFIG)), 18)
        self.assertEqual(len(load_shortlist(CONFIG, "development")), 6)
        self.assertEqual(len(load_shortlist(CONFIG, "confirmation")), 12)
        with self.assertRaises(ValueError):
            load_shortlist(CONFIG, "winners")

    def test_duplicate_and_missing_cohort_rejected(self):
        config = json.loads(CONFIG.read_text())
        config["tasks"][1]["task_id"] = config["tasks"][0]["task_id"]
        p = self.root / "config.json"
        p.write_text(json.dumps(config))
        with self.assertRaises(ValueError):
            load_shortlist(p)
        config["tasks"].pop()
        p.write_text(json.dumps(config))
        with self.assertRaises(ValueError):
            load_shortlist(p)

    def test_timing_discrepancy_not_silently_repaired(self):
        self.assertTrue(timing_check(278, 50, 667, 120)["passed"])
        self.assertFalse(timing_check(3702, 50, 2222, 120)["passed"])
        self.assertFalse(timing_check(164, 50, 399, 120)["passed"])
        for fps in (float("nan"), 0, -1):
            with self.assertRaises(ValueError):
                timing_check(3, fps, 3, 30)

    def test_window_preserves_clock_and_sources(self):
        a = motion()
        selected, window = window_motion(a, [.04, .16])
        self.assertEqual(len(selected["joint_pos"]), 7)
        self.assertEqual(window["start_frame"], 2)
        self.assertEqual(len(a["joint_pos"]), 11)
        for seconds in ([0, .3], [.01, .1], [.1, .1], [-.02, .1]):
            with self.assertRaises(ValueError):
                window_motion(a, seconds)

    def test_invalid_arrays_fail_closed(self):
        p = self.root / "motion.npz"
        a = motion()
        np.savez(p, **a)
        self.assertEqual(len(read_motion(p)["joint_pos"]), 11)
        for key, value in [("joint_pos", np.full((11, 29), np.nan)),
                           ("body_quat_w", np.zeros((11, 30, 4))),
                           ("joint_vel", np.zeros((10, 29))),
                           ("fps", np.array([0.])),
                           ("fps", np.array([50., 50.]))]:
            invalid = dict(a, **{key: value})
            np.savez(p, **invalid)
            with self.assertRaises(ValueError, msg=key):
                read_motion(p)

    def manifest(self):
        p = self.root / "motion.npz"
        np.savez(p, **motion())
        tasks = [dict(t, quality_status="passed", quality={"status": "passed"},
                      motion_path="motion.npz", sha256=sha256(p)) for t in load_shortlist(CONFIG)]
        m = self.root / "manifest.json"
        m.write_text(json.dumps({"schema": "mimicx.motion-study.v1", "tasks": tasks}))
        return m, tasks

    def test_portable_manifest_and_tamper(self):
        p, _ = self.manifest()
        self.assertEqual(len(load_eligible_manifest(p, root=self.root)), 18)
        (self.root / "motion.npz").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            load_eligible_manifest(p)

    def test_blocked_not_silently_filtered_and_no_escape(self):
        p, tasks = self.manifest()
        m = json.loads(p.read_text())
        m["tasks"][0]["quality_status"] = "blocked"
        p.write_text(json.dumps(m))
        with self.assertRaises(ValueError):
            load_eligible_manifest(p)
        self.assertEqual(len(load_eligible_manifest(p, "confirmation")), 12)
        m["tasks"] = tasks
        m["tasks"][0]["motion_path"] = "../escape.npz"
        p.write_text(json.dumps(m))
        with self.assertRaises(ValueError):
            load_eligible_manifest(p)

    def test_freeze_protocol_preserves_input_and_unrelated_blockers(self):
        p, _ = self.manifest()
        m = json.loads(p.read_text())
        for t in m['tasks']:
            if t.get('requires_ground_contact_protocol'):
                t['quality_status'] = 'blocked'
                t['quality'] = {'status': 'blocked', 'reasons': [GROUND_PROTOCOL_BLOCKER]}
            else:
                t['quality']['reasons'] = []
        m['tasks'][-1]['quality']['reasons'].append('independent FK failure')
        p.write_text(json.dumps(m))
        source_hash = sha256(p)
        protocol = json.loads(CONFIG.read_text())['ground_contact_protocol']
        approvals = {role: {'protocol_sha256': protocol_digest(protocol), 'reference': 'test-only agreement'}
                     for role in ('native_coordinator', 'sonic_scorer')}
        with tempfile.TemporaryDirectory() as out:
            dest = Path(out)/'new'
            frozen = freeze_ground_contact_protocol(p, dest, protocol, approvals)
            self.assertEqual(sha256(p), source_hash)
            self.assertEqual(frozen['counts'], {'passed': 17, 'blocked': 1})
            self.assertEqual(frozen['tasks'][-1]['quality']['reasons'], ['independent FK failure'])
            self.assertEqual(sha256(dest/'motion.npz'), sha256(self.root/'motion.npz'))
            with self.assertRaises(FileExistsError):
                freeze_ground_contact_protocol(p, dest, protocol, approvals)
            with self.assertRaises(ValueError):
                freeze_ground_contact_protocol(p, Path(out)/'absent', protocol, {})
            altered = dict(protocol, persistence_frames=1)
            with self.assertRaises(ValueError):
                freeze_ground_contact_protocol(p, Path(out)/'altered', altered, approvals)


class CpuFKTests(unittest.TestCase):
    def setUp(self):
        try:
            import mujoco
        except ImportError:
            self.skipTest("MuJoCo optional; run in existing physics environment for FK tests")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        bodies = []
        for i in range(29):
            name = {5: "left_ankle_roll_link", 11: "right_ankle_roll_link"}.get(i, f"link_{i}")
            bodies.append(f'<body name="{name}" pos="0 0 .05"><joint name="joint_{i}" range="-1 1"/><geom type="sphere" size=".01" mass=".1"/>')
        xml = '<mujoco><compiler angle="radian"/><worldbody><body name="pelvis"><freejoint/><geom type="sphere" size=".05" mass="1"/>' + ''.join(bodies) + '</body>' * 30 + '</worldbody></mujoco>'
        self.xml = self.root / "g1.xml"
        self.xml.write_text(xml)
        self.robot = CanonicalG1(self.xml)
        self.a = motion()
        d = mujoco.MjData(self.robot.model)
        mujoco.mj_forward(self.robot.model, d)
        self.a["body_pos_w"][:] = d.xpos[1:]
        self.a["body_quat_w"][:] = d.xquat[1:]

    def test_fk_checks_all_bodies_and_sign_equivalent_quats(self):
        self.assertTrue(self.robot.audit(self.a)["fk_passed"])
        self.a["body_quat_w"] *= -1
        self.assertTrue(self.robot.audit(self.a)["fk_passed"])
        self.a["body_pos_w"][0, 20, 0] += .01
        self.assertFalse(self.robot.audit(self.a)["fk_passed"])

    def test_bad_names_and_joint_limits(self):
        self.a["joint_names"] = np.array(["wrong"] * 29)
        with self.assertRaises(ValueError):
            self.robot.audit(self.a)
        del self.a["joint_names"]
        self.a["joint_pos"][5, 0] = 2
        self.assertFalse(self.robot.audit(self.a)["joint_limits_passed"])

    def test_missing_inputs_still_emit_all_entries(self):
        empty = self.root / "empty"
        empty.mkdir()
        roots = {name: empty for name in ("BONES", "AMASS", "LaFAN1")}
        m = prepare(CONFIG, self.root / "output", roots, self.xml)
        self.assertEqual(m["counts"], {"blocked": 18})
        self.assertEqual(len(m["tasks"]), 18)
        self.assertTrue((self.root / "output/AUDIT.md").is_file())
        with self.assertRaises(ValueError):
            prepare(CONFIG, self.root / "output", roots, self.xml)
        first = load_shortlist(CONFIG)[0]
        np.savez(empty / (Path(first['stem']).name + '.npz'), **self.a)
        found = prepare(CONFIG, self.root/'with_source', roots, self.xml)
        self.assertEqual(found['producer_evidence'], [])
        self.assertIsNotNone(found['tasks'][0]['motion_path'])

    def test_rebuild_elapsed_rotation_and_world_link_velocities(self):
        from scipy.spatial.transform import Rotation
        times = np.linspace(0, 1., 9)
        csv = np.zeros((len(times), 36))
        csv[:, 0] = times
        csv[:, 3:7] = Rotation.from_euler('z', (times * .5)[:, None]).as_quat()
        a = rebuild_csv(csv, times, self.robot)
        self.assertEqual(len(a['joint_pos']), 51)
        self.assertTrue(timing_check(51, 50, 121, 120)['passed'])
        np.testing.assert_allclose(a['body_pos_w'][:, 0, 0], np.arange(51)/50, atol=1e-7)
        np.testing.assert_allclose(a['body_lin_vel_w'][:, 0, 0], 1., atol=1e-6)
        np.testing.assert_allclose(a['body_ang_vel_w'][:, :, 2], .5, atol=1e-6)
        self.assertTrue(self.robot.audit(a)['fk_passed'])
        with self.assertRaises(ValueError):
            rebuild_csv(csv, times[::-1], self.robot)

    def test_eligible_pilot_named_permutation_and_no_overwrite(self):
        self.a['joint_names'] = np.array(self.robot.joint_names)
        self.a['body_names'] = np.array(self.robot.body_names)
        src = self.root / 'source.npz'
        np.savez(src, **self.a)
        m = self.root / 'manifest.json'
        m.write_text(json.dumps(dict(schema='mimicx.motion-study.v1', canonical_g1=self.robot.identity,
            tasks=[dict(task_id='pilot', quality_status='passed', quality={'status':'passed'},
                motion_path=src.name, sha256=sha256(src), frames=11, fps=50, duration=.2)])))
        live = self.root / 'live.json'
        live.write_text(json.dumps(dict(joint_names=self.robot.joint_names[::-1], body_names=self.robot.body_names[::-1])))
        report = export_isaac_pilot(m, 'pilot', live, self.root/'pilot')
        self.assertEqual(report['permutations']['joint_target_to_source'], list(range(28, -1, -1)))
        with self.assertRaises(FileExistsError):
            export_isaac_pilot(m, 'pilot', live, self.root/'pilot')


if __name__ == "__main__":
    unittest.main()
