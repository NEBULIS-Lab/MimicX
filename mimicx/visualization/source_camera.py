"""Physical-time state replay, source-only camera stabilization and robot-first ghosts."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp

from .baseline_recording import validate_times


def frame_times(frames: int, fps: float) -> np.ndarray:
    if isinstance(frames, bool) or int(frames) != frames or frames < 1 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Positive frame count and FPS are required")
    return np.arange(frames, dtype=float) / fps


def resample(times, root, joints, query, *, max_endpoint_hold=.04):
    """Interpolate translation/joints and SLERP wxyz roots without spatial alignment."""
    times = validate_times(times, "replay clock")
    root, joints, query = (np.asarray(v, dtype=float) for v in (root, joints, query))
    if (root.shape != (len(times), 7) or joints.ndim != 2 or joints.shape[0] != len(times)
            or query.ndim != 1 or not len(query) or not all(np.isfinite(v).all() for v in (root, joints, query))
            or not np.isfinite(max_endpoint_hold) or max_endpoint_hold < 0):
        raise ValueError("Invalid finite replay arrays or endpoint allowance")
    if not np.allclose(np.linalg.norm(root[:, 3:7], axis=1), 1, atol=1e-5, rtol=0):
        raise ValueError("Root quaternions must be unit wxyz")
    if max(times[0] - query.min(), query.max() - times[-1]) > max_endpoint_hold + 1e-12:
        raise ValueError("Missing replay time coverage; cannot stretch or loop motion")
    q = np.clip(query, times[0], times[-1])
    pos = np.column_stack([np.interp(q, times, root[:, j]) for j in range(3)])
    rotation = Slerp(times, Rotation.from_quat(root[:, [4, 5, 6, 3]]))(q).as_quat()[:, [3, 0, 1, 2]]
    angles = np.empty((len(q), 0)) if joints.shape[1] == 0 else np.column_stack(
        [np.interp(q, times, joints[:, j]) for j in range(joints.shape[1])])
    return np.column_stack([pos, rotation, angles])


def prepare_source_replay(arrays, *, frames, fps, max_endpoint_hold=.04):
    """Sample an exact-rollout replay package on the explicit source-video clock."""
    query = frame_times(frames, fps)
    actual = resample(arrays["timestamps"], arrays["robot_root_pose"], arrays["robot_joint_positions"],
                      query, max_endpoint_hold=max_endpoint_hold)
    reference = resample(arrays.get("reference_timestamps", arrays["timestamps"]), arrays["reference_root_pose"],
                         arrays["reference_joint_positions"], query, max_endpoint_hold=max_endpoint_hold)
    if actual.shape != reference.shape:
        raise ValueError("Actual and reference joint dimensions differ")
    return dict(time=query, actual=actual, reference=reference, coordinate_transform="identity",
                endpoint_hold_seconds=[max(0., float(arrays["timestamps"][0] - query[0])),
                                       max(0., float(query[-1] - arrays["timestamps"][-1]))])


def _camera_arrays(origin, linear, intrinsics):
    origin, linear, k = (np.asarray(v, dtype=float) for v in (origin, linear, intrinsics))
    if origin.shape != (3,) or linear.shape != (3, 3) or k.shape != (3, 3):
        raise ValueError("Invalid camera dimensions")
    if not all(np.isfinite(v).all() for v in (origin, linear, k)):
        raise ValueError("Camera values must be finite")
    scale = np.cbrt(np.linalg.det(linear))
    if scale <= 0 or not np.allclose((linear / scale).T @ (linear / scale), np.eye(3), atol=1e-6):
        raise ValueError("Camera basis must be a proper uniformly scaled rotation")
    if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]) or k[0, 1] != 0 or k[1, 0] != 0:
        raise ValueError("Expected positive, zero-skew pinhole intrinsics")
    return origin, linear, k, scale


def project_points(points, origin, world_from_camera_linear, intrinsics):
    origin, linear, k, _ = _camera_arrays(origin, world_from_camera_linear, intrinsics)
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Expected finite Nx3 world points")
    camera = np.linalg.solve(linear, (points - origin).T).T
    if np.any(camera[:, 2] <= 0):
        raise ValueError("Points must lie in front of the camera")
    pixels = camera @ k.T
    return pixels[:, :2] / pixels[:, 2:]


def set_camera(scene, origin, rotation, intrinsics, size, *, near=.02, far=100.):
    origin, rotation, k, scale = _camera_arrays(origin, rotation, intrinsics)
    width, height = size
    if not 0 < near < far or min(width, height) <= 0:
        raise ValueError("Invalid camera dimensions or clipping planes")
    rotation = rotation / scale
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    for camera in scene.camera:
        camera.pos[:] = origin
        camera.forward[:] = rotation[:, 2]
        camera.up[:] = -rotation[:, 1]
        camera.frustum_near, camera.frustum_far = near, far
        camera.frustum_bottom = -(height - cy) / fy * near
        camera.frustum_top = cy / fy * near
        camera.frustum_center = (width / 2 - cx) / fx * near
        # MuJoCo uses this field as a half-width, despite its name.
        camera.frustum_width = width / (2 * fx) * near
        camera.orthographic = 0


def protect_robot_pixels(composite, opaque, mask):
    composite, opaque, mask = np.asarray(composite), np.asarray(opaque), np.asarray(mask, dtype=bool)
    if composite.shape != opaque.shape or composite.ndim != 3 or mask.shape != opaque.shape[:2]:
        raise ValueError("Image and robot-mask shapes differ")
    result = composite.copy()
    result[mask] = opaque[mask]
    return result


def composite_ghost(opaque, ghost, ghost_mask, robot_mask, *, alpha=.23, guard_pixels=1):
    if not 0 < alpha < 1 or int(guard_pixels) != guard_pixels or guard_pixels < 0:
        raise ValueError("Ghost alpha must be translucent and guard size nonnegative")
    opaque, ghost = np.asarray(opaque), np.asarray(ghost)
    mask, robot = np.asarray(ghost_mask, dtype=bool), np.asarray(robot_mask, dtype=bool)
    if opaque.shape != ghost.shape or opaque.ndim != 3 or mask.shape != opaque.shape[:2] or robot.shape != mask.shape:
        raise ValueError("Ghost/image/mask shapes differ")
    blended = opaque.copy()
    blended[mask] = ((1 - alpha) * opaque[mask] + alpha * ghost[mask]).astype(opaque.dtype)
    if guard_pixels:
        robot = binary_dilation(robot, structure=np.ones((3, 3)), iterations=guard_pixels)
    return protect_robot_pixels(blended, opaque, robot)


def stabilize(calibration, background, *, sigma_frames=10., static_step_px=.8,
              static_drift_fraction=.03, min_valid_fraction=.9):
    """Smooth cameras using source-background evidence, never robot rollout poses."""
    origin = np.asarray(calibration["camera_origin_world_m"], dtype=float)
    basis = np.asarray(calibration["world_from_camera_linear"], dtype=float)
    intrinsics = np.asarray(calibration["intrinsics"], dtype=float)
    if origin.ndim != 2 or len(origin) < 2 or basis.shape != (len(origin), 3, 3) or intrinsics.shape != basis.shape:
        raise ValueError("Camera calibration frame counts differ or are too short")
    for o, r, k in zip(origin, basis, intrinsics):
        _camera_arrays(o, r, k)
    values = [background[k] for k in ("valid_fraction", "step_p95_px", "accumulated_max_px", "source_width")]
    if (not np.isfinite(values).all() or not 0 <= values[0] <= 1 or min(values[1:3]) < 0
            or values[3] <= 0 or not np.isfinite(sigma_frames) or sigma_frames <= 0):
        raise ValueError("Invalid background evidence or smoothing scale")
    scales = np.cbrt(np.linalg.det(basis))
    rotations = Rotation.from_matrix(basis / scales[:, None, None])
    static = values[0] >= min_valid_fraction and values[1] < static_step_px and values[2] < static_drift_fraction * values[3]
    if static:
        new_origin = np.broadcast_to(np.median(origin, axis=0), origin.shape).copy()
        new_rotation = np.broadcast_to(rotations.mean().as_matrix(), basis.shape).copy()
        intrinsics = np.broadcast_to(np.median(intrinsics, axis=0), intrinsics.shape).copy()
        policy = "locked source-anchored display camera"
    else:
        anchor = rotations.mean()
        residual = (anchor.inv() * rotations).as_rotvec()
        new_rotation = (anchor * Rotation.from_rotvec(gaussian_filter1d(
            residual, sigma_frames, axis=0, mode="nearest"))).as_matrix()
        new_origin = gaussian_filter1d(origin, sigma_frames, axis=0, mode="nearest")
        intrinsics = gaussian_filter1d(intrinsics, sigma_frames, axis=0, mode="nearest")
        policy = "source-anchored low-pass display camera"
    old_step, new_step = (np.linalg.norm(np.diff(x, axis=0), axis=1) for x in (origin, new_origin))
    return dict(calibration, camera_origin_world_m=new_origin.tolist(),
                world_from_camera_linear=(new_rotation * np.median(scales)).tolist(), intrinsics=intrinsics.tolist(),
                display_camera_policy=policy, sigma_frames=sigma_frames,
                raw_camera_modified_for_display=True, stabilization_uses_policy_rollout=False,
                source_background_audit={k: v for k, v in background.items() if k != "frames"},
                display_stability=dict(old_step_max_cm=float(old_step.max() * 100),
                                       new_step_max_cm=float(new_step.max() * 100)))


def export_source_video(prepared, joint_names, calibration, *, model_path, output, fps,
                        source_size, output_size=None, ghost_rgba=(.23, .47, .66, .23)):
    """Render saved states only; caller configures the graphics backend explicitly.

    The output is published only after a full decode verifies frame count, FPS and
    dimensions. Only uniform resizing is accepted. Existing outputs are never
    replaced, including when another export publishes first. This function never
    advances physics or invokes a controller.
    """
    import os
    from pathlib import Path
    import tempfile
    from .baseline_recording import named_joint_addresses

    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    actual, target = np.asarray(prepared["actual"]), np.asarray(prepared["reference"])
    frames = len(actual)
    if actual.shape != target.shape or actual.shape != (frames, 7 + len(joint_names)):
        raise ValueError("Prepared poses do not match joint names")
    if not np.allclose(prepared["time"], frame_times(frames, fps), atol=1e-10, rtol=0):
        raise ValueError("Prepared replay and output frame clocks differ")
    camera_keys = ("camera_origin_world_m", "world_from_camera_linear", "intrinsics")
    if any(len(calibration[key]) != frames for key in camera_keys):
        raise ValueError("Camera/source timeline mismatch")
    for index in range(frames):
        _camera_arrays(*(calibration[key][index] for key in camera_keys))
    rgba = np.asarray(ghost_rgba, dtype=float)
    if rgba.shape != (4,) or not np.isfinite(rgba).all() or np.any((rgba < 0) | (rgba > 1)) or not 0 < rgba[3] < 1:
        raise ValueError("Ghost color must be RGBA with translucent alpha")
    source = np.asarray(source_size, dtype=float)
    if source.shape != (2,) or not np.isfinite(source).all() or np.any(source < 1) or np.any(source != np.floor(source)):
        raise ValueError("Source dimensions must be positive integer width and height")
    size = np.asarray(source_size if output_size is None else output_size, dtype=float)
    if size.shape != (2,) or not np.isfinite(size).all() or np.any(size < 2) or np.any(size % 2):
        raise ValueError("Video dimensions must be positive even integers")
    width, height = map(int, size)
    source_width, source_height = map(int, source)
    if width * source_height != height * source_width:
        raise ValueError("Output aspect ratio must match source; only uniform resizing is supported")
    import imageio_ffmpeg
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    addresses, _ = named_joint_addresses(model, joint_names)
    if model.nq != actual.shape[1] or model.jnt_type[0] != 0:
        raise ValueError("Rendering requires one free-root robot and its named scalar joints")
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    data, ghost = mujoco.MjData(model), mujoco.MjData(model)
    options, perturb = mujoco.MjvOption(), mujoco.MjvPerturb()
    output.parent.mkdir(parents=True, exist_ok=True)
    protected_pixels = 0
    protected_pixel_max_delta = 0
    with tempfile.TemporaryDirectory(prefix=f".{output.stem}-", dir=output.parent) as temporary:
        partial = Path(temporary) / "video.mp4"
        writer = None
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            try:
                writer = imageio_ffmpeg.write_frames(str(partial), (width, height), fps=fps, codec="libx264",
                                                     macro_block_size=1, output_params=["-crf", "18", "-movflags", "+faststart"])
                writer.send(None)
                for index in range(frames):
                    data.qpos[:7], data.qpos[addresses] = actual[index, :7], actual[index, 7:]
                    ghost.qpos[:7], ghost.qpos[addresses] = target[index, :7], target[index, 7:]
                    mujoco.mj_forward(model, data)
                    mujoco.mj_forward(model, ghost)
                    renderer.update_scene(data, scene_option=options)
                    set_camera(renderer.scene, *(calibration[key][index] for key in camera_keys), source_size)
                    opaque = renderer.render().copy()
                    renderer.enable_segmentation_rendering()
                    segmentation = renderer.render()
                    renderer.disable_segmentation_rendering()
                    ids = segmentation[:, :, 0]
                    valid = (segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM) & (ids >= 0) & (ids < model.ngeom)
                    robot_mask = valid & (model.geom_bodyid[np.clip(ids, 0, model.ngeom - 1)] > 0)
                    robot_mask = binary_dilation(robot_mask, structure=np.ones((3, 3)))
                    begin = renderer.scene.ngeom
                    mujoco.mjv_addGeoms(model, ghost, options, perturb, mujoco.mjtCatBit.mjCAT_DYNAMIC, renderer.scene)
                    for geom in renderer.scene.geoms[begin:renderer.scene.ngeom]:
                        geom.rgba[:] = rgba
                        geom.matid = -1
                        geom.emission = .12
                    rgb = protect_robot_pixels(renderer.render(), opaque, robot_mask)
                    protected_pixels += int(robot_mask.sum())
                    if robot_mask.any():
                        protected_pixel_max_delta = max(protected_pixel_max_delta, int(np.max(
                            np.abs(rgb[robot_mask].astype(int) - opaque[robot_mask].astype(int)))))
                    writer.send(rgb)
            finally:
                if writer is not None:
                    writer.close()
        reader = imageio_ffmpeg.read_frames(str(partial), pix_fmt="rgb24")
        try:
            metadata = next(reader)
            count = sum(1 for _ in reader)
        finally:
            reader.close()
        if count != frames or tuple(metadata["size"]) != (width, height) or abs(metadata["fps"] - fps) > .011:
            raise ValueError("Encoded frame count, dimensions or FPS differ from source")
        # Same-filesystem hard-link publication is atomic and never replaces a
        # destination created after the preflight check. Temporary files are owned.
        os.link(partial, output)
    return dict(frames=frames, fps=fps, output_size=[width, height], full_decode_passed=True,
                coordinate_transform="identity", retime_factor=1., policy_or_reference_modified=False,
                robot_protected_pixels=protected_pixels, protected_pixel_max_delta=protected_pixel_max_delta,
                ghost_rgba=rgba.tolist(), camera_policy=calibration.get("display_camera_policy", "raw source camera"))
