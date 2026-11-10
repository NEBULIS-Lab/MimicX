"""CPU-only checks for complete clocks and invariant comparison displays."""
import copy
import importlib
import os
import subprocess

import numpy as np
import pytest

from mimicx.visualization import source_camera


def calibration(frames=4):
    return dict(camera_origin_world_m=np.zeros((frames, 3)),
                world_from_camera_linear=np.tile(np.eye(3), (frames, 1, 1)),
                intrinsics=np.tile([[100., 0, 80], [0, 100, 60], [0, 0, 1]], (frames, 1, 1)))


def test_reference_endpoint_hold_is_reported_independently():
    root = np.tile([0., 0, 1, 1, 0, 0, 0], (2, 1))
    arrays = dict(timestamps=[0., .12], reference_timestamps=[.01, .1],
                  robot_root_pose=root, reference_root_pose=root,
                  robot_joint_positions=np.zeros((2, 0)), reference_joint_positions=np.zeros((2, 0)))
    prepared = source_camera.prepare_source_replay(arrays, frames=4, fps=25)
    assert prepared["endpoint_hold_seconds"] == [0., 0.]
    np.testing.assert_allclose(prepared["reference_endpoint_hold_seconds"], [.01, .02])
    with pytest.raises(ValueError, match="coverage"):
        source_camera.prepare_source_replay(arrays, frames=4, fps=25, max_endpoint_hold=0)


def test_fixed_camera_audit_compares_every_frame_and_method_without_mutation():
    left = calibration()
    before = copy.deepcopy(left)
    report = source_camera.validate_camera_sequence(left, frames=4, require_constant=True,
                                                     shared_with=copy.deepcopy(left))
    assert report["constant_all_frames"] and report["shared_all_frames"]
    assert len(report["camera_sha256"]) == 64
    for key in left:
        np.testing.assert_array_equal(left[key], before[key])


@pytest.mark.parametrize("key,index", [("camera_origin_world_m", (2, 0)),
                                      ("world_from_camera_linear", (2, 0, 0)),
                                      ("intrinsics", (2, 0, 0))])
def test_fixed_camera_rejects_interior_drift(key, index):
    changed = calibration()
    changed[key][index] += .01
    with pytest.raises(ValueError):
        source_camera.validate_camera_sequence(changed, frames=4, require_constant=True)
    with pytest.raises(ValueError):
        source_camera.validate_camera_sequence(calibration(), frames=4, shared_with=changed)


def test_moving_camera_remains_supported_unless_constant_is_requested():
    moving = calibration()
    moving["camera_origin_world_m"][1, 0] = .1
    report = source_camera.validate_camera_sequence(moving, frames=4)
    assert not report["constant_all_frames"]
    assert report["shared_all_frames"] is None
    with pytest.raises(ValueError):
        source_camera.validate_camera_sequence(moving, frames=3)


def framehash(pts=(0, 1, 2, 3), durations=(1, 1, 1, 1), timebase="1/25"):
    return [f"#tb 0: {timebase}", "#dimensions 0: 16x12", *[
        f"0, {p}, {p}, {d}, 576, {'a' * 64}" for p, d in zip(pts, durations)]]


def test_full_clock_accepts_all_frames_and_records_span_not_only_fps():
    audit = importlib.import_module("mimicx.visualization.video_audit")
    report = audit.validate_framehash(framehash(), frames=4, fps=25, size=(16, 12))
    assert report["decoded_frames"] == 4
    assert report["duration_seconds"] == pytest.approx(.16)
    assert report["last_pts_seconds"] == pytest.approx(.12)
    assert report["full_clock_passed"]


@pytest.mark.parametrize("lines", [framehash(pts=(1, 2, 3, 4)),
                                   framehash(pts=(0, 1, 3, 4)),
                                   framehash(pts=(0, 1, 1, 3)),
                                   framehash(durations=(1, 2, 1, 1)),
                                   framehash(pts=(0, 1, 2)), framehash(timebase="1/30"),
                                   framehash()[:-1] + [f"0, 3, 3, 1, 575, {'a' * 64}"]])
def test_full_clock_rejects_offsets_gaps_duplicates_missing_tail_and_bad_payload(lines):
    audit = importlib.import_module("mimicx.visualization.video_audit")
    with pytest.raises(ValueError):
        audit.validate_framehash(lines, frames=4, fps=25, size=(16, 12))


def test_full_decode_uses_real_cpu_video_and_rejects_wrong_contract(tmp_path):
    ffmpeg = pytest.importorskip("imageio_ffmpeg")
    audit = importlib.import_module("mimicx.visualization.video_audit")
    video = tmp_path / "tiny.mp4"
    writer = ffmpeg.write_frames(str(video), (16, 12), fps=25, macro_block_size=1)
    writer.send(None)
    try:
        for value in (0, 60, 120, 180):
            writer.send(np.full((12, 16, 3), value, np.uint8))
    finally:
        writer.close()
    report = audit.audit_video(video, frames=4, fps=25, size=(16, 12))
    assert report["full_clock_passed"] and report["decoded_frames"] == 4
    with pytest.raises(ValueError):
        audit.audit_video(video, frames=5, fps=25, size=(16, 12))
    with pytest.raises(ValueError):
        audit.audit_video(video, frames=4, fps=30, size=(16, 12))


@pytest.mark.parametrize("expression", ["PTS+1/TB", "if(eq(N,2),PTS+0.01/TB,PTS)"])
def test_full_decode_does_not_normalize_real_shifted_or_variable_timestamps(tmp_path, expression):
    ffmpeg = pytest.importorskip("imageio_ffmpeg")
    audit = importlib.import_module("mimicx.visualization.video_audit")
    video = tmp_path / "bad-clock.nut"
    subprocess.run([ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-f", "lavfi",
                    "-i", "color=red:size=16x12:rate=25", "-frames:v", "4",
                    "-vf", f"settb=1/1000,setpts='{expression}'", "-fps_mode", "passthrough",
                    "-enc_time_base", "1/1000", "-c:v", "rawvideo", "-threads", "1", str(video)],
                   check=True, capture_output=True,
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES=""))
    with pytest.raises(ValueError, match="clock"):
        audit.audit_video(video, frames=4, fps=25, size=(16, 12))


def test_fixed_camera_export_rejects_drift_before_loading_model(tmp_path):
    moving = calibration()
    moving["camera_origin_world_m"][2, 0] = .1
    poses = np.tile([0., 0, 1, 1, 0, 0, 0], (4, 1))
    with pytest.raises(ValueError, match="constant"):
        source_camera.export_source_video(
            dict(time=np.arange(4) / 25, actual=poses, reference=poses), [], moving,
            model_path=tmp_path / "absent.xml", output=tmp_path / "out.mp4", fps=25,
            source_size=(160, 120), require_constant_camera=True)


def test_display_contrast_preserves_physics_and_excludes_floor_from_ghost(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    display = importlib.import_module("mimicx.visualization.display_scene")
    xml = '''<mujoco><visual><headlight ambient=".1 .1 .1"/></visual>
    <worldbody><body name="robot" pos="0 0 1"><freejoint/>
    <geom name="robot_geom" type="sphere" size=".2" mass="2"/>
    </body></worldbody></mujoco>'''
    path = tmp_path / "robot.xml"
    path.write_text(xml)
    original = mujoco.MjModel.from_xml_path(str(path))
    enhanced, report = display.load_contrast_model(path)
    assert path.read_text() == xml
    assert report["physics_steps"] == 0 and not report["rendered"]
    assert report["dynamic_ghost_excludes_floor"]
    assert report["max_pose_delta"] == 0
    np.testing.assert_array_equal(enhanced.body_mass, original.body_mass)
    np.testing.assert_array_equal(enhanced.qpos0, original.qpos0)
    floor = enhanced.geom(display.FLOOR)
    assert floor.contype == 0 and floor.conaffinity == 0 and floor.bodyid == 0
    np.testing.assert_array_equal(enhanced.geom("robot_geom").rgba, original.geom("robot_geom").rgba)
    assert np.mean(enhanced.vis.headlight.ambient) > np.mean(original.vis.headlight.ambient)


def test_display_contrast_rejects_includes_instead_of_rewriting_external_models(tmp_path):
    display = importlib.import_module("mimicx.visualization.display_scene")
    path = tmp_path / "robot.xml"
    path.write_text('<mujoco><include file="other.xml"/></mujoco>')
    with pytest.raises(ValueError, match="standalone"):
        display.load_contrast_model(path)


@pytest.mark.parametrize("element", ['<geom type="plane" size="1 1 .1"/>',
                                      '<body name="mimicx_display_body"/>'])
def test_display_contrast_refuses_existing_floor_and_reserved_names(tmp_path, element):
    display = importlib.import_module("mimicx.visualization.display_scene")
    path = tmp_path / "robot.xml"
    path.write_text(f'<mujoco><worldbody>{element}</worldbody></mujoco>')
    with pytest.raises(ValueError, match="explicit display scene"):
        display.load_contrast_model(path)


def test_display_floor_does_not_inherit_dark_robot_material(tmp_path):
    pytest.importorskip("mujoco")
    display = importlib.import_module("mimicx.visualization.display_scene")
    path = tmp_path / "robot.xml"
    path.write_text('''<mujoco><asset><material name="dark" rgba=".01 .01 .01 1"/></asset>
    <default><geom material="dark"/></default><worldbody>
    <body pos="0 0 1"><freejoint/><geom name="robot" type="sphere" size=".2" mass="1"/>
    </body></worldbody></mujoco>''')
    enhanced, _ = display.load_contrast_model(path)
    assert enhanced.geom(display.FLOOR).matid == -1
    assert enhanced.geom("robot").matid >= 0


@pytest.mark.parametrize("fps", [50., 29.97, 59.94, 30000 / 1001, 60000 / 1001])
def test_export_preserves_decimal_and_ntsc_clocks_with_cpu_encoding(tmp_path, monkeypatch, fps):
    """Stub only rasterization; exercise real CPU FK, exporter, encoder and audit."""
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("imageio_ffmpeg")
    model_path = tmp_path / "robot.xml"
    model_path.write_text('''<mujoco><worldbody><body pos="0 0 1"><freejoint/>
    <geom type="sphere" size=".2" mass="1"/></body></worldbody></mujoco>''')

    class SyntheticRasterizer:
        def __init__(self, model, *, height, width):
            self.model = model
            self.height, self.width = height, width
            self.scene = mujoco.MjvScene(model, maxgeom=100)
            self.segment = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update_scene(self, data, *, scene_option):
            self.scene.ngeom = 0
            mujoco.mjv_addGeoms(self.model, data, scene_option, mujoco.MjvPerturb(),
                              mujoco.mjtCatBit.mjCAT_ALL, self.scene)

        def enable_segmentation_rendering(self):
            self.segment = True

        def disable_segmentation_rendering(self):
            self.segment = False

        def render(self):
            if self.segment:
                return np.full((self.height, self.width, 2), -1, dtype=np.int32)
            return np.full((self.height, self.width, 3), 100, dtype=np.uint8)

    monkeypatch.setattr(mujoco, "Renderer", SyntheticRasterizer)
    frames = 121  # Long enough to expose the 29.97 versus 30000/1001 drift.
    poses = np.tile([0., 0, 1, 1, 0, 0, 0], (frames, 1))
    prepared = dict(time=np.arange(frames) / fps, actual=poses, reference=poses.copy())
    output = tmp_path / "export.mp4"
    receipt = source_camera.export_source_video(prepared, [], calibration(frames),
                                                model_path=model_path, output=output, fps=fps,
                                                source_size=(16, 12))
    assert receipt["full_clock_passed"] and receipt["frames"] == frames
    assert receipt["clock_audit"]["last_pts_seconds"] == pytest.approx(120 / fps, abs=1e-9)
    if fps == 50:
        assert receipt["clock_audit"]["duration_seconds"] == 2.42
    else:
        from mimicx.visualization.video_audit import audit_video
        other_rate = {29.97: 30000 / 1001, 59.94: 60000 / 1001,
                      30000 / 1001: 29.97, 60000 / 1001: 59.94}[fps]
        with pytest.raises(ValueError, match="clock"):
            audit_video(output, frames=frames, fps=other_rate, size=(16, 12))
