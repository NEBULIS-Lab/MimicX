"""CPU-only SONIC protocol tests. No policy, network, DDS or GPU is started."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from mimicx.experiments import sonic_study as study

SCORING_BODIES = list(study.BODY_INDEX) + [f"extra_{i}" for i in range(16)]


def protocol():
    return dict(protocol_id="synthetic.fixed-clock.v1", root_height_m=.25, relative_ee_height_m=.25, root_orientation_rad=1.,
                persistence_frames=1, clock_tolerance_s=.03, state_gap_s=.045,
                common_prefix_seconds=.04,
                critical_bodies=["left_ankle_roll_link", "right_ankle_roll_link",
                                 "left_wrist_yaw_link", "right_wrist_yaw_link"])


def fixture(target=None):
    if target is None:
        n, b = 6, len(SCORING_BODIES)
        target = {k: np.zeros((n, 29) if k.startswith("joint") else (n, b, 4 if k == "body_quat" else 3))
                  for k in study.FIELDS}
        target["body_quat"][..., 0] = 1
        target["body_pos"][..., 2] = 1
        target.update(joint_names=np.asarray(study.JOINT_NAMES), body_names=np.asarray(SCORING_BODIES))
    n = len(target["joint_pos"])
    wall = 9.9 + np.arange(2 * n + 19) * .01
    states = {k: np.repeat(v[:1], len(wall), axis=0) for k, v in target.items() if k in study.FIELDS}
    states.update(time=wall - 9.9, monotonic_time=wall, joint_names=target["joint_names"],
                  body_names=target["body_names"], elastic_band_enabled=np.zeros(len(wall), bool),
                  external_wrench_max=np.zeros(len(wall)), qfrc_applied_max=np.zeros(len(wall)),
                  fall_detected=np.zeros(len(wall), bool))
    dtype = [("index", float), ("time_monotonic_ms", float)] + [(f"q_{i}", float) for i in range(29)]
    q = np.zeros(n + 2, dtype=dtype)
    q["index"] = np.arange(n + 2)
    q["time_monotonic_ms"] = (9.98 + np.arange(n + 2) * .02) * 1000
    playing = np.zeros(n + 2, dtype=[("index", float), ("time_monotonic_ms", float), ("playing_0", float)])
    playing["index"], playing["time_monotonic_ms"] = q["index"], q["time_monotonic_ms"]
    playing["playing_0"][1:-1] = 1
    logged = np.zeros((n + 2, 36))
    logged[:, :3] = target["body_pos"][0, 0]
    logged[:, 3:7] = target["body_quat"][0, 0]
    ack = dict(enabled=False, root_constraints=False, monotonic_time=9.95, simulation_time=.05,
               external_wrench_max=0., qfrc_applied_max=0.)
    request = dict(monotonic_time=9.97)
    return states, target, q, playing, logged, ack, request, protocol()


class ClockTests(unittest.TestCase):
    def test_event_clock_fixed_complete_horizon(self):
        a, audit = study.align_recording(*fixture())
        self.assertEqual(a["joint_pos"].shape, (6, 29))
        self.assertFalse(audit["fitted_onset"])
        self.assertEqual(audit["fixed_frames"], 6)
        self.assertAlmostEqual(audit["simulation_to_wall_ratio"], 1.)
        self.assertEqual(audit["endpoint_hold_seconds"], 0)

    def reject(self, change, message):
        args = list(fixture())
        change(args)
        with self.assertRaisesRegex(ValueError, message):
            study.align_recording(*args)

    def test_release_force_and_bracket_gates(self):
        self.reject(lambda a: a[5].update(enabled=True), "release")
        self.reject(lambda a: a[5].update(root_constraints=True), "release")
        self.reject(lambda a: a[5].update(monotonic_time=10.01), "release")
        self.reject(lambda a: a[5].update(external_wrench_max=float("nan")), "release")
        self.reject(lambda a: a[5].update(simulation_time=100), "physical clock")
        self.reject(lambda a: a[0]["external_wrench_max"].__setitem__(8, .001), "forces")
        self.reject(lambda a: a[0]["qfrc_applied_max"].__setitem__(8, .001), "forces")
        self.reject(lambda a: a[0]["elastic_band_enabled"].__setitem__(9, True), "forces")
        self.reject(lambda a: a[0]["external_wrench_max"].__setitem__(8, -1), "telemetry")

    def test_no_truncation_pause_wrap_or_stale_row_identity(self):
        self.reject(lambda a: a[3]["playing_0"].__setitem__(1, 0), "complete")
        self.reject(lambda a: a[3]["playing_0"].__setitem__(0, 1), "complete")
        self.reject(lambda a: a[3]["index"].__setitem__(3, 99), "identity")
        self.reject(lambda a: a[3]["playing_0"].__setitem__(1, .4), "flags")
        self.reject(lambda a: a[4].__setitem__((2, 10), 1), "played target")
        self.reject(lambda a: a[4].__setitem__((2, 0), .1), "played target")

    def test_slow_clock_reset_and_sampling_gap_rejected(self):
        self.reject(lambda a: a[0].update(time=a[0]["time"] * .8), "clock")
        self.reject(lambda a: a[0]["time"].__setitem__(8, 0), "increasing")
        self.reject(lambda a: a[0]["monotonic_time"].__setitem__(8, 9.), "increasing")
        self.reject(lambda a: a[7].update(state_gap_s=.005), "samples")

    def test_cold_start_gap_allowed_but_reset_scored_and_bracket_gaps_rejected(self):
        args = list(fixture())
        args[0]["time"][0] -= .2
        args[0]["monotonic_time"][0] -= .2
        _, audit = study.align_recording(*args)
        self.assertGreater(audit["sampling_gap_audit"]["cold_start_maximum_wall_gap_s"], .2)
        self.assertLess(audit["sampling_gap_audit"]["maximum_wall_gap_s"], .045)
        for start, end, message in ((9.99, 10.06, "simulator samples"),
                                     (10.10, 10.16, "simulator samples"),
                                     (9.93, 9.98, "simulator samples")):
            args = list(fixture())
            states = args[0]
            args[5]["initialization"] = dict(monotonic_time=9.92)
            keep = (states["monotonic_time"] < start) | (states["monotonic_time"] > end)
            n = len(keep)
            args[0] = {k: v[keep] if k not in ("body_names", "joint_names") and len(v) == n else v
                       for k, v in states.items()}
            with self.assertRaisesRegex(ValueError, message):
                study.align_recording(*args)
        args = list(fixture())
        states = args[0]
        keep = (states["monotonic_time"] < 9.99) | (states["monotonic_time"] > 10.06)
        n = len(keep)
        args[0] = {k: v[keep] if k not in ("body_names", "joint_names") and len(v) == n else v
                   for k, v in states.items()}
        args[0]["time"] = np.arange(keep.sum()) * .01
        with self.assertRaisesRegex(ValueError, "wall-clock samples"):
            study.align_recording(*args)

    def test_no_error_fitting_or_logger_re_normalization(self):
        args = list(fixture())
        args[0]["joint_pos"][:] = .05
        a, audit = study.align_recording(*args)
        np.testing.assert_allclose(a["joint_pos"], .05)
        self.assertAlmostEqual(audit["state_logger_vs_named_sim_joints_rmse_rad"], .05)
        self.reject(lambda a: a[0]["joint_pos"].fill(.11), "named simulator")

    def test_named_mapping_not_storage_order(self):
        args = list(fixture())
        values = np.arange(29) / 1000
        args[1]["joint_pos"][:] = values
        args[4][:, 7:] = values
        for i in range(29):
            args[2][f"q_{i}"][:] = values[i]
        args[0]["joint_pos"][:] = values[::-1]
        args[0]["joint_names"] = args[0]["joint_names"][::-1]
        a, _ = study.align_recording(*args)
        np.testing.assert_allclose(a["joint_pos"][0], values)

    def test_coverage_never_endpoint_pads(self):
        source = np.array([0., .02, .04])
        with self.assertRaisesRegex(ValueError, "coverage"):
            study.resample(np.zeros((3, 2)), source, np.array([0., .06]))
        args = list(fixture())
        for k, v in args[0].items():
            if k not in ("joint_names", "body_names"):
                args[0][k] = v[:19]
        with self.assertRaisesRegex(ValueError, "coverage"):
            study.align_recording(*args)

    def test_slerp_preserves_sign_equivalence(self):
        q = np.array([[1., 0, 0, 0], [-1., 0, 0, 0]])
        np.testing.assert_allclose(study.resample(q, np.array([0., 1]), np.array([0., .5, 1]), quaternion=True),
                                   np.tile([1., 0, 0, 0], (3, 1)))

    def test_startup_failure_retained(self):
        args = list(fixture())
        args[0]["fall_detected"][7] = True
        _, audit = study.align_recording(*args)
        self.assertTrue(audit["startup_failure"])

    def test_phase_zero_ground_pose_not_failed_by_absolute_native_height(self):
        args = list(fixture())
        args[0]["body_pos"][..., 2] = .1
        args[1]["body_pos"][..., 2] = .1
        args[4][:, 2] = .1
        args[0]["fall_detected"][:] = True
        args[5]["initialization"] = {"mode": "reference_phase_zero_once"}
        _, audit = study.align_recording(*args)
        self.assertFalse(audit["startup_failure"])


class PortableTests(unittest.TestCase):
    def test_sdk_target_csv_allows_only_one_empty_trailing_field(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "target.csv"
            p.write_text(",".join(["0"] * 36) + ",\n")
            self.assertEqual(study.target_log(p).shape, (1, 36))
            p.write_text(",".join(["0"] * 37) + "\n")
            with self.assertRaisesRegex(ValueError, "36 numeric"):
                study.target_log(p)
            p.write_text(",".join(["nan"] + ["0"] * 35) + ",\n")
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                study.target_log(p)

    def test_vendored_release_and_no_reset_need_no_private_sources(self):
        from mimicx.experiments import sonic_helpers as helper
        with tempfile.TemporaryDirectory() as d:
            request = Path(d) / "release"
            env = SimpleNamespace(elastic_band=SimpleNamespace(enable=True), band_attached_link=1,
                                  mj_data=SimpleNamespace(time=1., qpos=np.array([0., 0., .1]),
                                                          xfrc_applied=np.ones((3, 6))))
            request.touch()
            self.assertTrue(helper.release_if_requested(env, request))
            self.assertFalse(env.elastic_band.enable)
            np.testing.assert_array_equal(env.mj_data.xfrc_applied[1], 0)
            np.testing.assert_array_equal(env.mj_data.xfrc_applied[0], 1)
            helper.install_no_reset_fall_monitor(env)
            env.check_fall()
            self.assertTrue(env.fall)
            self.assertEqual(env.mj_data.time, 1.)
            with self.assertRaises(ValueError):
                helper.require_forward_time(1., 1.)

    def test_application_dds_reset_echo_restores_header_without_motor_changes(self):
        from mimicx.experiments.sonic_helpers import DdsResetEcho
        bridge = SimpleNamespace(low_state=SimpleNamespace(mode_machine=0),
            low_cmd=SimpleNamespace(mode_machine=0, motor_cmd=[SimpleNamespace(q=0.,dq=0.,kp=1.,kd=.1,tau=0.) for _ in range(29)]))
        echo = DdsResetEcho(bridge, monotonic_time=1.)
        self.assertEqual(bridge.low_state.mode_machine, 1)
        self.assertIsNone(echo.observe(monotonic_time=1.001))
        bridge.low_cmd.mode_machine = 1
        self.assertIsNone(echo.observe(monotonic_time=1.002))
        self.assertEqual(bridge.low_state.mode_machine, 0)
        bridge.low_cmd.mode_machine = 0
        proof = echo.observe(monotonic_time=1.004)
        self.assertTrue(proof["original_header_restored"])
        self.assertTrue(proof["motor_commands_unchanged"])
        self.assertEqual(bridge.low_cmd.motor_cmd[0].kp, 1.)

    def test_dds_reset_echo_rejects_stale_receipt_and_changed_motor_commands(self):
        from mimicx.experiments.sonic_helpers import DdsResetEcho
        bridge = SimpleNamespace(low_state=SimpleNamespace(mode_machine=0),
            low_cmd=SimpleNamespace(mode_machine=0, motor_cmd=[SimpleNamespace(q=0.,dq=0.,kp=1.,kd=.1,tau=0.) for _ in range(29)]))
        echo = DdsResetEcho(bridge, monotonic_time=1.)
        with self.assertRaisesRegex(ValueError, "timeout"):
            echo.observe(monotonic_time=1.251)
        self.assertEqual(bridge.low_state.mode_machine, 0)
        echo = DdsResetEcho(bridge, monotonic_time=2.)
        bridge.low_cmd.mode_machine = 1
        echo.observe(monotonic_time=2.005)
        bridge.low_cmd.mode_machine = 0
        bridge.low_cmd.motor_cmd[0].q = .1
        with self.assertRaisesRegex(ValueError, "motor command changed"):
            echo.observe(monotonic_time=2.01)

    def test_seed_is_explicit_reproducible_and_nominal_is_not_robustness(self):
        spec = dict(mode="release_state_uniform", joint_position_rad=.01, root_velocity_m_s=.02)
        one = study.perturbation(1001, spec)
        self.assertEqual(one, study.perturbation(1001, spec))
        self.assertNotEqual(one, study.perturbation(1002, spec))
        self.assertFalse(one["forces_applied"])
        self.assertFalse(one["reference_modified"])
        self.assertTrue(np.max(np.abs(one["joint_position_delta_rad"])) <= .01)
        self.assertEqual(study.perturbation(1, {"mode": "nominal"})["joint_position_delta_rad"], [0.] * 29)
        for seed, p in ((-1, spec), (True, spec), (1, {}), (1, {"mode": "release_state_uniform"})):
            with self.assertRaises(ValueError):
                study.perturbation(seed, p)

    def test_permutation_validation(self):
        np.testing.assert_array_equal(study.permutation(["a", "b"], ["b", "a"]), [1, 0])
        for source, target in ((["a", "a"], ["a", "b"]), (["a"], ["b"]), ([1], [1])):
            with self.assertRaises(ValueError):
                study.permutation(source, target)

    def test_parameter_parser_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "params.hpp"
            p.write_text("const auto mujoco_to_isaaclab = {" + ",".join(map(str, range(29))) + "};\n"
                         "const auto isaaclab_to_mujoco = {" + ",".join(map(str, range(29))) + "};")
            np.testing.assert_array_equal(study.parameter_mapping(p), np.arange(29))
            p.write_text("const auto mujoco_to_isaaclab = {0,0};")
            with self.assertRaises(ValueError):
                study.parameter_mapping(p)

    def test_per_task_manifest_does_not_hide_blocked_cohort(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            motion = root / "motion.npz"
            motion.write_bytes(b"test identity")
            base = dict(motion_path="motion.npz", sha256=study.sha(motion), quality_status="passed",
                        quality={"status": "passed"})
            tasks = [dict(base, task_id="new_task_a"), dict(base, task_id="new_task_b"),
                     dict(base, task_id="new_task_c", quality_status="blocked")]
            manifest = root / "manifest.json"
            study.dump(manifest, dict(schema="mimicx.motion-study.v1", tasks=tasks))
            self.assertEqual(len(study.selected_tasks(manifest, ["new_task_b"])), 1)
            with self.assertRaisesRegex(ValueError, "ineligible"):
                study.selected_tasks(manifest)
            with self.assertRaisesRegex(ValueError, "unknown"):
                study.selected_tasks(manifest, ["missing"])
            motion.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "integrity"):
                study.selected_tasks(manifest, ["new_task_a"])

    def test_cpu_loader_failure_records_glibc_without_running_binary(self):
        c = {k: "/placeholder" for k in study.PATH_KEYS}
        c.update(sonic_root=".", deploy_binary="/placeholder/deploy", sim_python=sys.executable, environment={})
        with patch.object(study.subprocess, "run", side_effect=[
                subprocess.CompletedProcess([], 1, "", "libc.so.6: version `GLIBC_2.38' not found"),
                subprocess.CompletedProcess([], 0, "{}", "")]) as mocked:
            result = study.preflight(c)
        self.assertEqual(result["status"], "blocked")
        self.assertTrue(result["glibc_blocked"])
        self.assertFalse(result["gpu_started"])
        self.assertEqual(mocked.call_args_list[0].args[0][0], "ldd")
        self.assertEqual(mocked.call_args_list[1].kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")

    def test_run_guard_is_per_run_nonblocking(self):
        with tempfile.TemporaryDirectory() as d:
            with study.duplicate_guard(Path(d)):
                with self.assertRaisesRegex(ValueError, "already active"):
                    with study.duplicate_guard(Path(d)):
                        self.fail("duplicate acquired")


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "CPU MuJoCo environment required")
class PhysicsTests(unittest.TestCase):
    def setUp(self):
        import mujoco
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.xml = self.root / "robot.xml"
        bodies = list(study.BODY_INDEX)[1:] + [f"extra_{i}" for i in range(16)]
        children = "".join(f'<body name="{b}" pos="{i*.01} 0 0"><joint name="{j}" type="hinge" '
                           'range="-5 5"/><geom type="sphere" size=".01" mass=".1"/></body>'
                           for i, (b, j) in enumerate(zip(bodies, study.JOINT_NAMES)))
        self.xml.write_text('<mujoco><compiler angle="radian"/><worldbody><body name="pelvis" pos="0 0 1">'
                            '<freejoint name="floating_base_joint"/><geom type="sphere" size=".05" mass="1"/>'
                            + children + '</body></worldbody></mujoco>')
        model = mujoco.MjModel.from_xml_path(str(self.xml))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        n = 11
        self.motion = {"fps": np.array([50.]), "joint_pos": np.zeros((n, 29)),
                       "joint_vel": np.zeros((n, 29)), "body_pos_w": np.tile(data.xpos[1:], (n, 1, 1)),
                       "body_quat_w": np.tile(data.xquat[1:], (n, 1, 1)),
                       "body_lin_vel_w": np.zeros((n, 30, 3)), "body_ang_vel_w": np.zeros((n, 30, 3)),
                       "joint_names": np.asarray(study.JOINT_NAMES),
                       "body_names": np.asarray([model.body(i).name for i in range(1, 31)])}
        self.motion_path = self.root / "motion.npz"
        np.savez_compressed(self.motion_path, **self.motion)

    def config(self):
        corrected = self.root / "corrected"
        corrected.mkdir(exist_ok=True)
        for name in study.HELPERS:
            (corrected / name).write_text("# synthetic copied provenance source\n")
        params = self.root / "params.hpp"
        params.write_text("const auto mujoco_to_isaaclab = {" + ",".join(map(str, range(29))) + "};\n"
                          "const auto isaaclab_to_mujoco = {" + ",".join(map(str, range(29))) + "};")
        for name in ("sim.yaml", "decoder.onnx", "encoder.onnx", "obs.yaml"):
            (self.root / name).touch()
        c = dict(sonic_root=".", corrected_source="corrected", policy_parameters="params.hpp",
                 canonical_model="robot.xml", simulator_model="robot.xml", simulator_config="sim.yaml",
                 deploy_binary=sys.executable, sim_python=sys.executable, decoder="decoder.onnx",
                 encoder="encoder.onnx", observation_config="obs.yaml", protocol=protocol(),
                 dds_isolation="parent_serialized_domain_zero")
        study.dump(self.root / "protocol.json", protocol())
        c.update(protocol_file="protocol.json", protocol_sha256=study.protocol_sha256(protocol()))
        path = self.root / "runtime.json"
        study.dump(path, c)
        return path

    def manifest(self):
        t = dict(task_id="a_brand_new_motion", motion_path="motion.npz", sha256=study.sha(self.motion_path),
                 frames=11, fps=50, duration=.2, quality_status="passed", quality={"status": "passed"},
                 protocol_id=protocol()["protocol_id"], protocol_sha256=study.protocol_sha256(protocol()))
        path = self.root / "manifest.json"
        study.dump(path, dict(schema="mimicx.motion-study.v1", tasks=[t], protocol=protocol(),
                             protocol_sha256=study.protocol_sha256(protocol())))
        return path

    def test_protocol_binding_rejects_runtime_and_task_drift(self):
        runtime = study.runtime_config(self.config())
        manifest = study.read(self.manifest())
        digest = study.bind_protocol(manifest, runtime)
        self.assertEqual(digest, study.protocol_sha256(protocol()))
        runtime["protocol"]["persistence_frames"] = 5
        with self.assertRaisesRegex(ValueError, "differs"):
            study.bind_protocol(manifest, runtime)
        runtime = study.runtime_config(self.config())
        manifest["tasks"][0]["protocol_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "task protocol"):
            study.bind_protocol(manifest, runtime)
        old_manifest = dict(schema="mimicx.motion-study.v1", tasks=[])
        self.assertEqual(study.bind_protocol(old_manifest, runtime), digest)

    def test_named_fk_rejects_even_a_just_over_2mm_difference(self):
        c = study.runtime_config(self.config())
        _, audit = study.validate_reference(self.motion_path, c)
        self.assertTrue(audit["fk_passed"])
        self.motion["body_pos_w"][:, 8, 0] += .00201
        np.savez_compressed(self.motion_path, **self.motion)
        with self.assertRaisesRegex(ValueError, "FK/limits"):
            study.validate_reference(self.motion_path, c)

    def test_unnamed_or_deployment_asset_mismatch_fails_closed(self):
        c = study.runtime_config(self.config())
        del self.motion["joint_names"]
        np.savez_compressed(self.motion_path, **self.motion)
        with self.assertRaisesRegex(ValueError, "joint_names"):
            study.validate_reference(self.motion_path, c)

    def test_asset_frame_bridge_accepts_10mm_origin_change_without_motion_repair(self):
        config = self.config()
        c = study.runtime_config(config)
        different = self.root / "different.xml"
        different.write_text(self.xml.read_text().replace('pos="0.01 0 0"', 'pos="0.02 0 0"'))
        c["simulator_model"] = str(different)
        arrays, audit = study.validate_reference(self.motion_path, c)
        self.assertLess(audit["deployment_fk_position_m"], 1e-8)
        self.assertGreater(max(audit["cross_asset_body_origin_difference_by_name_m"].values()), .009)
        self.assertEqual(audit["reference_mode"], "reference_to_deployment_fk")
        control = study.reference_to_deployment_fk(arrays, c)
        np.testing.assert_array_equal(control["joint_pos"], arrays["joint_pos"])
        np.testing.assert_allclose(control["body_pos"][:, 0], arrays["body_pos_w"][:, 0], atol=1e-12)
        target = {k: arrays[v] for k, v in study.FIELDS.items()}
        result = self.score(control, target)
        self.assertAlmostEqual(result["full_horizon"]["world_body_error_m"]["mean"], 0., places=12)
        self.assertAlmostEqual(result["full_horizon"]["root_local_fk_body_error_m"]["mean"], 0., places=12)
        study.dump(config, c)
        output = self.root / "bridged"
        with patch.object(study, "preflight", return_value={"status": "blocked"}):
            runs = study.prepare(self.manifest(), config, output, seed=1001, perturb={"mode": "nominal"})
        folder = (output / runs[0]).parent
        saved = study.load_npz(folder / "target.npz")
        np.testing.assert_array_equal(saved["joint_pos"], arrays["joint_pos"])
        np.testing.assert_array_equal(saved["body_pos"], arrays["body_pos_w"])
        self.assertEqual(study.sha(folder / "motion.npz"), study.sha(self.motion_path))

    def test_public_prepare_does_not_require_private_corrected_sources(self):
        config = self.config()
        c = study.read(config)
        del c["corrected_source"]
        study.dump(config, c)
        with patch.object(study, "preflight", return_value={"status": "blocked"}):
            runs = study.prepare(self.manifest(), config, self.root / "public", seed=1, perturb={"mode": "nominal"})
        folder = (self.root / "public" / runs[0]).parent
        self.assertTrue((folder / "code/sonic_helpers.py").is_file())
        self.assertFalse((folder / "code/direct_policy_comparison.py").exists())

    def test_prepare_is_executable_frozen_and_does_not_launch(self):
        config = self.config()
        output = self.root / "prepared"
        with patch.object(study, "preflight", return_value={"status": "blocked"}), \
                patch.object(study.subprocess, "Popen") as launch:
            runs = study.prepare(self.manifest(), config, output, seed=1001, perturb={"mode": "nominal"})
        launch.assert_not_called()
        self.assertEqual(runs, ["a_brand_new_motion/seed_1001/run.json"])
        path = output / runs[0]
        run = study.verify_run(path)
        command = study.commands(path, run)
        self.assertIn("_simulate", command["sim"])
        self.assertIn("--target-motion-logfile", command["deploy"])
        self.assertEqual(run["frames"], 11)
        self.assertEqual(len(run["scoring_body_names"]), 30)
        self.assertEqual(len(run["control_body_names"]), 14)
        target = study.load_npz(path.parent / "target.npz")
        self.assertEqual(target["body_pos"].shape[1], 30)
        csv = np.loadtxt(path.parent / "reference/a_brand_new_motion/body_pos.csv", delimiter=",", skiprows=1)
        self.assertEqual(csv.shape, (11, 14 * 3))
        self.assertAlmostEqual(run["duration"], .2)
        self.assertFalse(study.read(output / "prepared.json")["launched"])
        inspected = subprocess.run([sys.executable, "scripts/experiments/sonic_study.py", "commands",
                                    "--run", str(path)], text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(inspected.stdout)["deploy"], command["deploy"])
        self.assertIn("weights", command["deploy"][2])
        with self.assertRaisesRegex(ValueError, "empty destination"):
            study.prepare(self.manifest(), config, output, seed=1001, perturb={"mode": "nominal"})
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=True), \
                patch.object(study, "preflight", return_value={"status": "blocked"}):
            with self.assertRaisesRegex(ValueError, "preflight blocked"):
                study.evaluate(path)
        (path.parent / "target.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            study.verify_run(path)

    def test_source_environment_dds_constraints(self):
        p = self.config()
        c = study.read(p)
        c["dds_domain"] = 2
        study.dump(p, c)
        with self.assertRaisesRegex(ValueError, "DDS domain 0"):
            study.runtime_config(p)
        c["dds_domain"] = 0
        c["environment"] = {"CUDA_VISIBLE_DEVICES": "0"}
        study.dump(p, c)
        with self.assertRaisesRegex(ValueError, "caller"):
            study.runtime_config(p)
        c["environment"] = {}
        c["release_settle_s"] = 3
        study.dump(p, c)
        with self.assertRaisesRegex(ValueError, "hold"):
            study.runtime_config(p)

    def test_saved_recording_scores_end_to_end_without_launching(self):
        with patch.object(study, "preflight", return_value={"status": "blocked"}):
            runs = study.prepare(self.manifest(), self.config(), self.root / "prepared", seed=1001,
                                 perturb={"mode": "nominal"})
        path = self.root / "prepared" / runs[0]
        folder, run = path.parent, study.read(path)
        target = study.load_npz(folder / "target.npz")
        states, _, q, playing, logged, ack, request, _ = fixture(target)
        np.savez_compressed(folder / "sim_state.npz", **states)
        (folder / "state_logs").mkdir()
        for name, values in (("q.csv", q), ("motion_playing.csv", playing)):
            np.savetxt(folder / "state_logs" / name, np.column_stack([values[k] for k in values.dtype.names]),
                       delimiter=",", header=",".join(values.dtype.names), comments="")
        np.savetxt(folder / "target_motion.csv", logged, delimiter=",")
        ack["perturbation"] = run["perturbation"]
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(self.xml))
        init = study.initialize_phase_zero(model, mujoco.MjData(model), target, run["perturbation"])
        init["monotonic_time"] = 9.948
        ack.update(initialization=init, dds_reset_echo=dict(marker_received_monotonic=9.949,
                   restored_received_monotonic=9.95, original_header_restored=True, motor_commands_unchanged=True))
        study.dump(folder / "release-band.ack.json", ack)
        request["monotonic_time"] = 9.985
        study.dump(folder / "play_request.json", request)
        study.dump(folder / "control_request.json", dict(monotonic_time=9.97, keys="I]"))
        study.dump(folder / "startup_events.json", dict(sdk_init_done_observed_monotonic=9.94,
                    reference_reset_release_request_monotonic=9.945, first_control_heading_observed_monotonic=9.984))
        columns = ["index", "time_monotonic_ms", "quat_0", "quat_1", "quat_2", "quat_3"]
        imu = np.column_stack([q["index"], q["time_monotonic_ms"], np.ones(len(q)), np.zeros((len(q), 3))])
        for filename in ("base_quat.csv", "torso_quat.csv"):
            np.savetxt(folder / "state_logs" / filename, imu, delimiter=",", header=",".join(columns), comments="")
        (folder / "deploy.log").write_text("Reset heading state to 1,0,0,0\nPlaying motion\n")
        study.dump(folder / "sim_ready.json", dict(states_only=True, reset_on_fall=False, sim_dt=.005))
        study.dump(folder / "runtime_fk_audit.json", dict(deployment_fk_position_m=0., deployment_fk_rotation_rad=0.))
        valid_q = q.copy()
        q["q_0"][0] = 1.
        with self.assertRaisesRegex(ValueError, "stale pre-reset"):
            study.verify_startup_evidence(folder, run, states, target, q, playing, ack)
        q = valid_q
        ack["dds_reset_echo"]["original_header_restored"] = False
        with self.assertRaisesRegex(ValueError, "receipt proof"):
            study.verify_startup_evidence(folder, run, states, target, q, playing, ack)
        ack["dds_reset_echo"]["original_header_restored"] = True
        (folder / "deploy.log").write_text("Reset heading state to 0,0,0,1\nPlaying motion\n")
        with self.assertRaisesRegex(ValueError, "stale suspended"):
            study.verify_startup_evidence(folder, run, states, target, q, playing, ack)
        (folder / "deploy.log").write_text("Reset heading state to 1,0,0,0\nPlaying motion\n")
        stale_imu = imu.copy()
        stale_imu[0, 2:] = [0, 0, 0, 1]
        np.savetxt(folder / "state_logs/torso_quat.csv", stale_imu, delimiter=",",
                   header=",".join(columns), comments="")
        with self.assertRaisesRegex(ValueError, "stale pre-reset"):
            study.verify_startup_evidence(folder, run, states, target, q, playing, ack)
        np.savetxt(folder / "state_logs/torso_quat.csv", imu, delimiter=",", header=",".join(columns), comments="")
        with patch.object(study.subprocess, "Popen") as launch:
            result = study.score_recording(path)
        launch.assert_not_called()
        self.assertTrue(result["physical_success"])
        self.assertEqual(result["adaptation_updates"], 0)
        self.assertTrue(result["comparative_initialization_compatible"])
        startup = study.read(folder / "startup_audit.json")
        self.assertEqual(startup["reference_frame"], 0)
        self.assertEqual(startup["actual_CONTROL_monotonic"], 9.98)
        self.assertEqual(len(study.load_npz(folder / "paired_states.npz")["time"]), 11)
        self.assertEqual(result["evidence_sha256"]["sim_state.npz"], study.sha(folder / "sim_state.npz"))
        with self.assertRaisesRegex(ValueError, "existing result"):
            study.score_recording(path)
        original = {p: study.sha(p) for p in folder.rglob("*") if p.is_file()}
        with patch.object(study.subprocess, "Popen") as launch:
            rescored = study.score_recording(path, output=self.root / "rescore", reason="CPU sample-gap scope regression")
        launch.assert_not_called()
        self.assertTrue(rescored["physical_success"])
        provenance = study.read(self.root / "rescore/rescoring.json")
        self.assertFalse(provenance["new_physical_execution"])
        self.assertTrue(provenance["source_evidence_unchanged"])
        self.assertEqual(original, {p: study.sha(p) for p in folder.rglob("*") if p.is_file()})
        with self.assertRaises(FileExistsError):
            study.score_recording(path, output=self.root / "rescore", reason="never overwrite")
        with self.assertRaisesRegex(ValueError, "outside"):
            study.score_recording(path, output=folder / "rescore", reason="never edit old capture")

    def test_failed_launch_waits_for_only_its_children_and_records_attempt(self):
        with patch.object(study, "preflight", return_value={"status": "cpu_preflight_passed_runtime_unverified"}):
            runs = study.prepare(self.manifest(), self.config(), self.root / "prepared", seed=1001,
                                 perturb={"mode": "nominal"})
        path = self.root / "prepared" / runs[0]
        sim = SimpleNamespace(pid=100000001, returncode=None)
        deploy = SimpleNamespace(pid=100000002, returncode=None)

        def stopped(process):
            process.returncode = 0

        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=True), \
                patch.object(study, "preflight", return_value={"status": "cpu_preflight_passed_runtime_unverified"}), \
                patch.object(study.subprocess, "Popen", side_effect=[sim, deploy]) as popen, \
                patch.object(study, "wait_for", side_effect=[None, RuntimeError("synthetic startup failure")]), \
                patch.object(study, "stop", side_effect=stopped) as stop:
            with self.assertRaisesRegex(RuntimeError, "synthetic startup"):
                study.evaluate(path)
        self.assertEqual(popen.call_count, 2)
        self.assertEqual([c.args[0].pid for c in stop.call_args_list], [deploy.pid, sim.pid])
        attempt = study.read(path.parent / "attempt.json")
        self.assertEqual(attempt["status"], "failed")
        self.assertEqual(attempt["cuda_visible_devices"], "0")
        self.assertIsNone(attempt["tmux_session"])
        self.assertIsNone(attempt["tmux"])
        self.assertEqual(attempt["deploy_pid"], deploy.pid)
        self.assertGreater(attempt["wall_seconds"], 0)
        self.assertFalse((path.parent / "result.json").exists())

    def score(self, actual, target, **kwargs):
        return study.common_metrics(actual, target, model_xml=self.xml,
                                    joint_names=study.JOINT_NAMES, body_names=SCORING_BODIES,
                                    protocol=protocol(), **kwargs)

    def test_full_and_prospective_prefix_failure_metrics(self):
        _, target, *_ = fixture()
        actual = copy.deepcopy(target)
        result = self.score(actual, target)
        self.assertTrue(result["complete_motion"])
        self.assertEqual(result["common_prefix_frames"], 3)
        actual["body_pos"][3:, 0, 2] += .3
        result = self.score(actual, target)
        self.assertFalse(result["complete_motion"])
        self.assertEqual(result["first_failure_frame"], 3)
        self.assertAlmostEqual(result["normalized_failure_horizon"], .6)
        self.assertEqual(result["common_prefix"]["world_root_error_m"]["mean"], 0.)
        self.assertGreater(result["full_horizon"]["world_root_error_m"]["mean"], 0.)

    def test_startup_native_failure_and_postreset_mask(self):
        _, target, *_ = fixture()
        result = self.score(target, target, startup_failure=True)
        self.assertEqual(result["first_failure_frame"], 0)
        result = self.score(target, target, native_failure_frame=2, valid_mask=[1, 1, 0, 0, 0, 0])
        self.assertEqual(result["normalized_failure_horizon"], .4)
        self.assertIsNone(result["full_horizon"])
        self.assertIsNone(result["common_prefix"])
        with self.assertRaisesRegex(ValueError, "postreset"):
            self.score(target, target, valid_mask=[1, 1, 0, 1, 0, 0])
        with self.assertRaisesRegex(ValueError, "phase-zero"):
            self.score(target, target, reference_frames=np.arange(6) + 1)

    def test_world_error_does_not_change_canonical_local_fk(self):
        _, target, *_ = fixture()
        actual = copy.deepcopy(target)
        actual["body_pos"] += np.array([2., 0, 0])
        result = self.score(actual, target)
        self.assertEqual(result["full_horizon"]["world_root_error_m"]["mean"], 2.)
        self.assertEqual(result["full_horizon"]["root_local_fk_body_error_m"]["mean"], 0.)

    def test_five_frame_union_persistence_is_backdated_without_grace(self):
        _, target, *_ = fixture()
        actual = copy.deepcopy(target)
        p = protocol()
        p["persistence_frames"] = 5
        # Different channels alternate, but their OR-union persists for five frames.
        actual["body_pos"][[1, 3, 5], 0, 2] += .3
        actual["body_quat"][[2, 4], 0] = [np.cos(.55), np.sin(.55), 0, 0]
        result = study.common_metrics(actual, target, model_xml=self.xml, joint_names=study.JOINT_NAMES,
                                      body_names=SCORING_BODIES, protocol=p)
        self.assertEqual(result["first_failure_frame"], 1)
        self.assertFalse(result["complete_motion"])
        actual["body_quat"][4, 0] = target["body_quat"][4, 0]
        result = study.common_metrics(actual, target, model_xml=self.xml, joint_names=study.JOINT_NAMES,
                                      body_names=SCORING_BODIES, protocol=p)
        self.assertTrue(result["complete_motion"])

    def test_control_14_body_subset_cannot_be_compared_as_common_30(self):
        _, target, *_ = fixture()
        subset = {k: v[:, :14] if k.startswith("body_") and k != "body_names" else v
                  for k, v in target.items()}
        with self.assertRaisesRegex(ValueError, "all 30"):
            study.common_metrics(subset, subset, model_xml=self.xml, joint_names=study.JOINT_NAMES,
                                 body_names=list(study.BODY_INDEX), protocol=protocol())

    def test_invalid_tail_cannot_supply_failure_persistence(self):
        _, target, *_ = fixture()
        actual = copy.deepcopy(target)
        actual["body_pos"][2:, 0, 2] += .3
        p = protocol()
        p["persistence_frames"] = 5
        result = study.common_metrics(actual, target, model_xml=self.xml, joint_names=study.JOINT_NAMES,
                                      body_names=SCORING_BODIES, protocol=p, valid_mask=[1, 1, 1, 1, 0, 0],
                                      native_failure_frame=4)
        self.assertEqual(result["first_failure_frame"], 4)
        self.assertIsNone(result["full_horizon"])

    def test_existing_14_body_recording_reconstructs_all30_from_physical_state(self):
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(self.xml))
        d = mujoco.MjData(model)
        mujoco.mj_forward(model, d)
        states, target, *_ = fixture()
        n = len(states["time"])
        states["qpos"] = np.repeat(d.qpos[None], n, axis=0)
        states["qvel"] = np.repeat(d.qvel[None], n, axis=0)
        states["body_names"] = states["body_names"][:14]
        for key in ("body_pos", "body_quat", "body_lin_vel", "body_ang_vel"):
            states[key] = states[key][:, :14]
        rebuilt, audit = study.canonical_body_states(states, target, self.xml)
        self.assertTrue(audit["reconstructed"])
        self.assertEqual(rebuilt["body_pos"].shape, (n, 30, 3))
        np.testing.assert_allclose(rebuilt["body_pos"][0], d.xpos[1:])
        del states["qvel"]
        with self.assertRaisesRegex(ValueError, "full saved physical"):
            study.canonical_body_states(states, target, self.xml)

    def test_deferred_body_fk_does_not_step_or_modify_measured_states(self):
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(self.xml))
        d = mujoco.MjData(model)
        positions, velocities = np.repeat(d.qpos[None], 4, axis=0), np.repeat(d.qvel[None], 4, axis=0)
        before_p, before_v = positions.copy(), velocities.copy()
        with patch.object(mujoco, "mj_step", side_effect=AssertionError("must never advance simulation")):
            body = study.physical_body_trajectory(model, positions, velocities, SCORING_BODIES)
        np.testing.assert_array_equal(positions, before_p)
        np.testing.assert_array_equal(velocities, before_v)
        self.assertEqual(body["body_pos"].shape, (4, 30, 3))

    def test_phase_zero_initializer_preserves_clock_and_world_link_velocity(self):
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(self.xml))
        data = mujoco.MjData(model)
        data.time = 12.5
        _, target, *_ = fixture()
        target["body_pos"][:, 0] = [1., 2., .8]
        target["body_quat"][:, 0] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
        target["body_lin_vel"][:, 0] = [.1, .2, .3]
        target["body_ang_vel"][:, 0] = [.2, .3, .4]
        target["joint_vel"][:] = .5
        p = study.perturbation(1001, dict(mode="release_state_uniform", joint_position_rad=.005, root_velocity_m_s=0.))
        with patch.object(mujoco, "mj_step", side_effect=AssertionError("initialization cannot step physics")):
            init = study.initialize_phase_zero(model, data, target, p)
        self.assertEqual(data.time, 12.5)
        np.testing.assert_allclose(data.qpos[:3], [1., 2., .8])
        np.testing.assert_allclose(data.qpos[7:], p["joint_position_delta_rad"])
        physical = study.physical_body_trajectory(model, data.qpos[None], data.qvel[None], SCORING_BODIES)
        np.testing.assert_allclose(physical["body_lin_vel"][0, 0], [.1, .2, .3], atol=1e-12)
        np.testing.assert_allclose(physical["body_ang_vel"][0, 0], [.2, .3, .4], atol=1e-12)
        self.assertEqual(init["count"], 1)
        self.assertTrue(init["before_control"])
        reordered = dict(target, joint_names=target["joint_names"][::-1],
                         joint_pos=target["joint_pos"][:, ::-1], joint_vel=target["joint_vel"][:, ::-1])
        other = mujoco.MjData(model)
        study.initialize_phase_zero(model, other, reordered, p)
        np.testing.assert_array_equal(other.qpos, data.qpos)
        np.testing.assert_array_equal(other.qvel, data.qvel)


if __name__ == "__main__":
    unittest.main()
