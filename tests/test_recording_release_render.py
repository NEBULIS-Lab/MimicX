"""Explicit opt-in rendering checks, separate from the CPU suite."""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("MIMICX_RENDER_TESTS") != "1", reason="Rendering requires explicit opt-in")


def test_rendered_pixels_match_offcenter_source_intrinsics():
    mujoco = pytest.importorskip("mujoco")
    from mimicx.visualization.source_camera import set_camera

    model = mujoco.MjModel.from_xml_string('''<mujoco><visual><global offwidth="400" offheight="300"/></visual>
    <worldbody><geom type="sphere" size=".025" pos=".6 .5 3" rgba="1 0 0 1"/></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=300, width=400) as renderer:
        renderer.update_scene(data)
        renderer.scene.geoms[0].emission = 1
        set_camera(renderer.scene, np.zeros(3), np.eye(3), [[400, 0, 170], [0, 420, 130], [0, 0, 1]], (400, 300))
        image = renderer.render()
    yy, xx = np.where(image[:, :, 0] > 20)
    assert len(xx) > 0
    assert np.linalg.norm([xx.mean() - 250, yy.mean() - 200]) < 1.5


def test_export_source_video_encodes_four_frames_and_protects_robot(tmp_path, monkeypatch):
    pytest.importorskip("mujoco")
    ffmpeg = pytest.importorskip("imageio_ffmpeg")
    from mimicx.visualization import source_camera

    model_path = tmp_path / "robot.xml"
    model_path.write_text('''<mujoco><visual><headlight ambient="1 1 1" diffuse="0 0 0" specular="0 0 0"/></visual>
    <worldbody><body name="robot"><freejoint name="root"/>
    <geom type="sphere" size=".28" mass="1" rgba=".15 .8 .25 1"/>
    <body name="arm" pos=".28 0 0"><joint name="hinge" axis="0 1 0"/>
    <geom type="box" size=".12 .06 .06" mass=".1" rgba=".15 .8 .25 1"/>
    </body></body></worldbody></mujoco>''', encoding="utf-8")
    actual = np.tile([0., 0, 3, 1, 0, 0, 0, 0], (4, 1))
    actual[:, 0] = [-.3, -.1, .1, .3]
    actual[:, 7] = [0, .1, .2, .3]
    reference = actual.copy()
    reference[:, 0] += .12
    reference[:, 2] -= .15
    prepared = dict(time=np.array([0., .04, .08, .12]), actual=actual, reference=reference)
    calibration = dict(camera_origin_world_m=np.zeros((4, 3)),
                       world_from_camera_linear=np.tile(np.eye(3), (4, 1, 1)),
                       intrinsics=np.tile([[280., 0, 150], [0, 280, 112], [0, 0, 1]], (4, 1, 1)))
    protect = source_camera.protect_robot_pixels
    masks, overlap_changes = [], []

    def inspect_protection(composite, opaque, mask):
        assert opaque.shape == (120, 160, 3)
        assert mask.sum() > 100
        result = protect(composite, opaque, mask)
        np.testing.assert_array_equal(result[mask], opaque[mask])
        masks.append(mask.copy())
        overlap_changes.append(np.count_nonzero(composite[mask] != opaque[mask]))
        return result

    monkeypatch.setattr(source_camera, "protect_robot_pixels", inspect_protection)
    output = tmp_path / "replay.mp4"
    receipt = source_camera.export_source_video(
        prepared, ["hinge"], calibration, model_path=model_path, output=output, fps=25,
        source_size=(320, 240), output_size=(160, 120), ghost_rgba=(.9, .1, .1, .4))
    assert receipt["frames"] == 4 and receipt["fps"] == 25
    assert receipt["full_decode_passed"]
    assert receipt["output_size"] == [160, 120]
    assert receipt["coordinate_transform"] == "identity" and receipt["retime_factor"] == 1
    assert receipt["robot_protected_pixels"] == sum(mask.sum() for mask in masks)
    assert receipt["protected_pixel_max_delta"] == 0
    assert len(masks) == 4 and min(overlap_changes) > 0
    assert np.where(masks[-1])[1].mean() - np.where(masks[0])[1].mean() > 20
    reader = ffmpeg.read_frames(str(output), pix_fmt="rgb24")
    try:
        metadata = next(reader)
        decoded = [np.frombuffer(frame, np.uint8).reshape(120, 160, 3) for frame in reader]
    finally:
        reader.close()
    assert metadata["fps"] == pytest.approx(25)
    assert tuple(metadata["size"]) == (160, 120)
    assert len(decoded) == 4
    assert all(frame.std() > 2 for frame in decoded)
    assert np.abs(decoded[-1].astype(float) - decoded[0]).mean() > .5
    np.testing.assert_array_equal(prepared["actual"][:, 0], [-.3, -.1, .1, .3])
    np.testing.assert_array_equal(prepared["actual"][:, 2], [3, 3, 3, 3])
    np.testing.assert_array_equal(prepared["time"], [0, .04, .08, .12])
    assert not list(tmp_path.glob("*.partial.mp4"))
