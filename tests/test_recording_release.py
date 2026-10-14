"""CPU contracts for baseline recording and source-camera replay."""
import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest


def recording():
    return importlib.import_module("mimicx.visualization.baseline_recording")


def camera():
    return importlib.import_module("mimicx.visualization.source_camera")


def states():
    return dict(time=np.arange(5) * .02, monotonic_time=10 + np.arange(5) * .02,
                elastic_band_enabled=np.zeros(5, bool), external_wrench_max=np.zeros(5))


@pytest.mark.parametrize("previous,current", [(2, 2), (2, 0), (2, np.nan), (np.nan, 3)])
def test_forward_clock_rejects_stalls_resets_and_nonfinite(previous, current):
    with pytest.raises(ValueError):
        recording().require_forward_time(previous, current)


def test_release_clears_only_band_body_and_acknowledges(tmp_path):
    request = tmp_path / "release"
    env = SimpleNamespace(elastic_band=SimpleNamespace(enable=True), band_attached_link=1,
                          mj_data=SimpleNamespace(time=3., xfrc_applied=np.ones((3, 6))))
    assert not recording().release_if_requested(env, request)
    request.touch()
    assert recording().release_if_requested(env, request)
    assert not env.elastic_band.enable
    np.testing.assert_array_equal(env.mj_data.xfrc_applied[1], 0)
    np.testing.assert_array_equal(env.mj_data.xfrc_applied[[0, 2]], 1)
    assert json.loads((tmp_path / "release.ack.json").read_text())["enabled"] is False
    assert not recording().release_if_requested(env, request)


def test_stale_ack_cannot_leave_reenabled_band_hanging(tmp_path):
    request = tmp_path / "release"
    request.touch()
    env = SimpleNamespace(elastic_band=SimpleNamespace(enable=True), band_attached_link=0,
                          mj_data=SimpleNamespace(time=3., xfrc_applied=np.ones((1, 6))))
    recording().release_if_requested(env, request)
    env.elastic_band.enable = True
    with pytest.raises(ValueError, match="acknowledg"):
        recording().release_if_requested(env, request)


def test_fall_monitor_does_not_reset_or_modify_pose():
    env = SimpleNamespace(mj_data=SimpleNamespace(time=7., qpos=np.array([0., 0., .15])))
    recording().install_no_reset_fall_monitor(env)
    env.check_fall()
    assert env.fall and env.mj_data.time == 7.
    np.testing.assert_array_equal(env.mj_data.qpos, [0, 0, .15])


def test_playback_measures_real_clock_instead_of_assuming_logger_fps():
    out = recording().validate_playback(states(), [10.01, 10.03, 10.07])
    np.testing.assert_allclose(out["playback_time"], [0, .02, .06])
    assert out["simulation_to_wall_ratio"] == pytest.approx(1)


@pytest.mark.parametrize("mutation", ["support", "negative_wrench", "nan_wrench", "slow", "reset", "missing", "short"])
def test_playback_rejects_invalid_force_clock_and_coverage(mutation):
    sample = states()
    wall = [10.01, 10.07]
    if mutation == "support":
        sample["elastic_band_enabled"][0] = True  # interpolation bracket counts too
    elif mutation == "negative_wrench":
        sample["external_wrench_max"][1] = -1
    elif mutation == "nan_wrench":
        sample["external_wrench_max"][1] = np.nan
    elif mutation == "slow":
        sample["time"] *= .5
    elif mutation == "reset":
        sample["time"][2] = 0
    elif mutation == "missing":
        del sample["elastic_band_enabled"]
    else:
        wall[-1] = 10.1
    with pytest.raises(ValueError):
        recording().validate_playback(sample, wall)


def test_joint_order_is_named_and_requires_bijection():
    np.testing.assert_array_equal(recording().joint_permutation(["hip", "knee"], ["knee", "hip"]), [1, 0])
    for names in (["hip", "hip"], ["missing", "knee"], ["hip"]):
        with pytest.raises(ValueError):
            recording().joint_permutation(["hip", "knee"], names)


def test_physical_resampling_keeps_positions_and_tail_without_retiming():
    root = np.zeros((3, 7)); root[:, 3] = 1; root[:, 0] = [5, 6, 7]
    out = camera().resample([0, .5, 1], root, [[0], [1], [2]], [0, .25, 1, 1.01])
    np.testing.assert_allclose(out[:, 0], [5, 5.5, 7, 7])
    np.testing.assert_allclose(out[:, 7], [0, .5, 2, 2])
    np.testing.assert_allclose(np.linalg.norm(out[:, 3:7], axis=1), 1)
    with pytest.raises(ValueError, match="coverage"):
        camera().resample([0, .5, 1], root, [[0], [1], [2]], [0, 1.2])
    np.testing.assert_allclose(camera().frame_times(4, 25), [0, .04, .08, .12])


def test_offcenter_camera_matches_source_pinhole():
    cam = SimpleNamespace(pos=np.zeros(3), forward=np.zeros(3), up=np.zeros(3))
    k = np.array([[400, 0, 170], [0, 420, 130], [0, 0, 1.]])
    camera().set_camera(SimpleNamespace(camera=[cam]), [1, 2, 3], np.eye(3), k, (400, 300))
    pixels = camera().project_points([[1.6, 2.5, 6]], [1, 2, 3], np.eye(3), k)
    np.testing.assert_allclose(pixels, [[250, 200]])
    x, y = .2 * cam.frustum_near, -1 / 6 * cam.frustum_near
    np.testing.assert_allclose([(x - cam.frustum_center + cam.frustum_width) / (2 * cam.frustum_width) * 400,
                               (cam.frustum_top - y) / (cam.frustum_top - cam.frustum_bottom) * 300], [250, 200])


def test_ghost_translucency_never_tints_actual_robot():
    opaque = np.full((5, 5, 3), 240, np.uint8)
    ghost = np.full_like(opaque, 40)
    mask = np.zeros((5, 5), bool); mask[2, 2] = True
    out = camera().composite_ghost(opaque, ghost, np.ones((5, 5), bool), mask, alpha=.25, guard_pixels=1)
    np.testing.assert_array_equal(out[1:4, 1:4], 240)
    np.testing.assert_array_equal(out[0, 0], 190)
    np.testing.assert_array_equal(opaque, 240)
    with pytest.raises(ValueError):
        camera().composite_ghost(opaque, ghost, mask, mask, alpha=1)


def test_stabilization_uses_only_source_and_keeps_proper_rotations():
    t = np.arange(120)
    origin = np.column_stack([t * .01 + .08 * (-1.) ** t, np.zeros((120, 2))])
    calibration = dict(camera_origin_world_m=origin.tolist(),
                       world_from_camera_linear=np.tile(np.eye(3), (120, 1, 1)).tolist(),
                       intrinsics=np.tile(np.eye(3), (120, 1, 1)).tolist())
    background = dict(valid_fraction=1., step_p95_px=4., accumulated_max_px=100., source_width=1000)
    out = camera().stabilize(calibration, background)
    assert not out["stabilization_uses_policy_rollout"]
    assert out["display_stability"]["new_step_max_cm"] < out["display_stability"]["old_step_max_cm"]
    assert np.ptp(np.asarray(out["camera_origin_world_m"])[:, 0]) > .5
    np.testing.assert_allclose(np.linalg.det(out["world_from_camera_linear"]), 1)
    background.update(step_p95_px=.1, accumulated_max_px=1.)
    out = camera().stabilize(calibration, background)
    np.testing.assert_allclose(np.diff(out["camera_origin_world_m"], axis=0), 0)
    np.testing.assert_array_equal(calibration["camera_origin_world_m"], origin)


def test_recording_modules_do_not_import_simulator_or_graphics():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-c", "import sys; import mimicx.visualization.source_camera; "
                             "import mimicx.visualization.contact_replay; "
                             "assert 'mujoco' not in sys.modules; assert 'torch' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_state_recorder_uses_named_order_and_rejects_clock_reset(tmp_path):
    class Model:
        jnt_type = np.array([0, 3, 3])
        jnt_qposadr = np.array([0, 7, 8])
        jnt_dofadr = np.array([0, 6, 7])

        def joint(self, name):
            return SimpleNamespace(id={"hip": 1, "knee": 2}[name])

    recorder = recording().StateRecorder(Model(), ["knee", "hip"])
    data = SimpleNamespace(time=0., qpos=np.array([0, 0, 1, 1, 0, 0, 0, .1, .2]),
                           qvel=np.zeros(8), xfrc_applied=np.zeros((1, 6)))
    recorder.append(data, monotonic_time=10, elastic_band_enabled=False)
    data.qpos[7:] = [.3, .4]; data.time = .02
    recorder.append(data, monotonic_time=10.02, elastic_band_enabled=False)
    with pytest.raises(ValueError):
        recorder.append(data, monotonic_time=10.04, elastic_band_enabled=False)
    recorder.save(tmp_path / "state.npz")
    with np.load(tmp_path / "state.npz", allow_pickle=False) as out:
        np.testing.assert_allclose(out["joint_pos"], [[.2, .1], [.4, .3]])
        np.testing.assert_array_equal(out["joint_names"], ["knee", "hip"])
        np.testing.assert_allclose(out["time"], [0, .02])


def test_source_preparation_preserves_root_and_explicit_source_frame_clock():
    arrays = dict(timestamps=np.array([0., .1]),
                  robot_root_pose=np.array([[5, 0, 1, 1, 0, 0, 0], [6, 0, 1, 1, 0, 0, 0.]]),
                  robot_joint_positions=np.array([[0.], [1.]]),
                  reference_root_pose=np.array([[0, 0, 1, 1, 0, 0, 0], [1, 0, 1, 1, 0, 0, 0.]]),
                  reference_joint_positions=np.array([[0.], [2.]]))
    out = camera().prepare_source_replay(arrays, frames=4, fps=30)
    np.testing.assert_allclose(out["actual"][:, 0], [5, 5 + 1/3, 5 + 2/3, 6])
    np.testing.assert_allclose(out["reference"][:, 0], [0, 1/3, 2/3, 1])
    assert out["coordinate_transform"] == "identity"
    assert len(out["time"]) == 4
    with pytest.raises(ValueError):
        camera().prepare_source_replay(arrays, frames=20, fps=30)


def test_recording_cli_audits_npz_without_simulator(tmp_path):
    import subprocess
    import sys
    np.savez(tmp_path / "state.npz", **states())
    np.save(tmp_path / "wall.npy", [10.01, 10.03, 10.07])
    result = subprocess.run([sys.executable, "scripts/visualization/audit_recording.py", "--states",
                             str(tmp_path / "state.npz"), "--playback-wall", str(tmp_path / "wall.npy")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    np.testing.assert_allclose(json.loads(result.stdout)["playback_time"], [0, .02, .06])


def test_source_cli_prepares_all_source_frames_without_spatial_alignment(tmp_path):
    import subprocess
    import sys
    replay = tmp_path / "replay"; replay.mkdir()
    root = np.array([[5, 0, 1, 1, 0, 0, 0], [6, 0, 1, 1, 0, 0, 0.]])
    arrays = dict(timestamps=[0, .1], robot_root_pose=root, reference_root_pose=root,
                  robot_joint_positions=[[0], [1]], reference_joint_positions=[[0], [1]])
    for name, array in arrays.items():
        np.save(replay / f"{name}.npy", array)
    (replay / "joint_names.json").write_text('["hip"]')
    output = tmp_path / "prepared.npz"
    result = subprocess.run([sys.executable, "scripts/visualization/source_camera.py", "prepare", "--replay",
                             str(replay), "--frames", "4", "--fps", "30", "--output", str(output)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with np.load(output, allow_pickle=False) as data:
        assert data["actual"].shape == (4, 8)
        np.testing.assert_allclose(data["actual"][[0, -1], 0], [5, 6])


def _export_request(tmp_path):
    actual = np.tile([0., 0, 3, 1, 0, 0, 0, 0], (4, 1))
    return dict(prepared=dict(time=np.arange(4) / 25, actual=actual, reference=actual.copy()),
                joint_names=["hinge"], calibration=dict(camera_origin_world_m=np.zeros((4, 3)),
                    world_from_camera_linear=np.tile(np.eye(3), (4, 1, 1)),
                    intrinsics=np.tile([[280., 0, 150], [0, 280, 112], [0, 0, 1]], (4, 1, 1))),
                model_path=tmp_path / "unused.xml", output=tmp_path / "output.mp4", fps=25,
                source_size=(320, 240), output_size=(160, 120))


@pytest.mark.parametrize("size", [(160, 160), (320, 120)])
def test_export_rejects_aspect_change_before_graphics_import(tmp_path, size):
    request = _export_request(tmp_path)
    request["output_size"] = size
    with pytest.raises(ValueError, match="aspect"):
        camera().export_source_video(**request)
    assert not request["output"].exists()


@pytest.mark.parametrize("size", [(0, 240), (320, 0), (np.nan, 240), (320.5, 240)])
def test_export_rejects_invalid_source_dimensions_before_graphics_import(tmp_path, size):
    request = _export_request(tmp_path)
    request["source_size"] = size
    with pytest.raises(ValueError, match="dimensions"):
        camera().export_source_video(**request)


def test_export_never_overwrites_existing_video(tmp_path):
    request = _export_request(tmp_path)
    request["output"].write_bytes(b"existing video")
    with pytest.raises(FileExistsError):
        camera().export_source_video(**request)
    assert request["output"].read_bytes() == b"existing video"


@pytest.mark.parametrize("command", ["prepare", "stabilize", "render"])
def test_source_cli_refuses_existing_output_before_loading_inputs(tmp_path, command):
    import subprocess
    import sys
    output = tmp_path / ("replay.mp4" if command == "render" else "data.json")
    if command == "prepare":
        output = tmp_path / "data.npz"
    output.write_bytes(b"existing artifact")
    args = [sys.executable, "scripts/visualization/source_camera.py", command, "--output", str(output)]
    if command == "stabilize":
        args += ["--calibration", str(tmp_path / "absent.json"), "--background", str(tmp_path / "absent.json")]
    else:
        args += ["--replay", str(tmp_path / "absent"), "--frames", "4", "--fps", "25"]
        if command == "render":
            args += ["--calibration", "absent.json", "--model", "absent.xml", "--source-size", "320", "240", "--backend", "egl"]
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode != 0 and "FileExistsError" in result.stderr
    assert output.read_bytes() == b"existing artifact"


def test_source_cli_preserves_existing_render_receipt(tmp_path):
    import subprocess
    import sys
    (tmp_path / "replay.json").write_text('{"existing": true}')
    result = subprocess.run([sys.executable, "scripts/visualization/source_camera.py", "render",
                             "--output", str(tmp_path / "replay.mp4"), "--replay", "absent", "--frames", "4", "--fps", "25",
                             "--calibration", "absent.json", "--model", "absent.xml", "--source-size", "320", "240", "--backend", "egl"],
                            capture_output=True, text=True)
    assert result.returncode != 0 and "FileExistsError" in result.stderr
    assert json.loads((tmp_path / "replay.json").read_text()) == {"existing": True}
    assert not (tmp_path / "replay.mp4").exists()
