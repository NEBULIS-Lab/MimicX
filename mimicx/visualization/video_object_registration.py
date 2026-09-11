"""CPU source registration and explicitly separate held-prop display transforms."""

from __future__ import annotations

import numpy as np


def fit_similarity(source, target, weights=None):
    """Fit target = scale * rotation @ source + translation (no reflection)."""
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("similarity requires matching Nx3 points")
    if len(source) < 3 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("similarity requires at least three finite points")
    weight = np.ones(len(source)) if weights is None else np.asarray(weights, dtype=float)
    if weight.shape != (len(source),) or not np.isfinite(weight).all() or (weight < 0).any() or weight.sum() <= 0:
        raise ValueError("invalid similarity weights")
    weight = weight / weight.sum()
    x_mean, y_mean = weight @ source, weight @ target
    x, y = source - x_mean, target - y_mean
    if np.linalg.matrix_rank(x * np.sqrt(weight[:, None])) < 2:
        raise ValueError("degenerate calibration points")
    left, singular, right = np.linalg.svd((x * weight[:, None]).T @ y)
    sign = np.ones(3)
    sign[-1] = np.linalg.det(right.T @ left.T)
    rotation = right.T @ np.diag(sign) @ left.T
    scale = float(np.dot(singular, sign) / np.sum(weight[:, None] * x * x))
    if scale <= 0:
        raise ValueError("nonpositive similarity scale")
    return scale, rotation, y_mean - scale * rotation @ x_mean


def match_reference_phase(recorded_joints, motion_joints, tolerance=1e-5):
    """Recover the logged command phase, including resets, from reference qpos."""
    from scipy.spatial.distance import cdist
    recorded, motion = np.asarray(recorded_joints), np.asarray(motion_joints)
    if recorded.ndim != 2 or motion.ndim != 2 or recorded.shape[1] != motion.shape[1]:
        raise ValueError("joint trajectories must have matching joint dimensions")
    if not np.isfinite(recorded).all() or not np.isfinite(motion).all() or len(motion) == 0:
        raise ValueError("joint trajectories must be finite and reference nonempty")
    distances = cdist(recorded, motion, metric="sqeuclidean")
    phase = distances.argmin(axis=1)
    rms = np.sqrt(distances[np.arange(len(recorded)), phase] / recorded.shape[1])
    if (rms > tolerance).any():
        raise ValueError(f"reference phase is not exact: max joint RMS {rms.max():.8g}")
    if np.any((distances <= tolerance**2 * recorded.shape[1]).sum(axis=1) > 1):
        raise ValueError("ambiguous reference phase: multiple matching reference poses")
    return phase, rms


def source_frames_from_phase(phase, reference_fps, source_fps, source_count, offset_seconds=0.0):
    """Nearest observed frame at physical source time; never stretch the horizon."""
    if reference_fps <= 0 or source_fps <= 0 or source_count < 1:
        raise ValueError("positive frame rates and source count required")
    times = np.asarray(phase, dtype=float) / reference_fps + offset_seconds
    frames = np.floor(times * source_fps + 0.5).astype(int)
    return [int(f) if 0 <= f < source_count and t >= 0 else None for f, t in zip(frames, times)]


def project_points(points, intrinsics):
    points = np.asarray(points, dtype=float)
    if points.shape[-1] != 3 or not np.isfinite(points).all() or (points[..., 2] <= 0).any():
        raise ValueError("projection requires finite points in front of the camera")
    homogeneous = points @ np.asarray(intrinsics, dtype=float).T
    return homogeneous[..., :2] / homogeneous[..., 2:3]


def lift_to_camera_depth(center_xy, intrinsics, depth):
    if not np.isfinite(depth) or depth <= 0:
        raise ValueError("camera depth must be positive")
    center = np.asarray(center_xy, dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("ball center must be finite xy")
    ray = np.linalg.solve(np.asarray(intrinsics, dtype=float), np.r_[center, 1.0])
    return ray * (depth / ray[2])


def ground_ray_depth(origin, direction, center_height):
    origin, direction = np.asarray(origin, dtype=float), np.asarray(direction, dtype=float)
    if abs(direction[2]) < 1e-8:
        return None
    depth = float((center_height - origin[2]) / direction[2])
    return depth if depth > 0 and np.isfinite(depth) else None


def fit_camera_translation(points_camera, pixels, intrinsics, confidence):
    """Correct SMPL body-origin reprojection bias using source 2D joints only."""
    from scipy.optimize import least_squares
    points, pixels = np.asarray(points_camera, dtype=float), np.asarray(pixels, dtype=float)
    weight = np.sqrt(np.asarray(confidence, dtype=float))
    if points.shape != (len(pixels), 3) or pixels.shape[1:] != (2,) or weight.shape != (len(points),):
        raise ValueError("camera correction requires matching skeleton observations")
    if (weight > .5).sum() < 4:
        raise ValueError("too few reliable source joints for camera correction")
    result = least_squares(
        lambda translation: ((project_points(points + translation, intrinsics) - pixels) * weight[:, None]).ravel(),
        np.zeros(3), loss="soft_l1", f_scale=5.0,
        bounds=([-1, -1, max(-1, -float(points[:, 2].min()) + .1)], [1, 1, 1]),
    )
    if not result.success:
        raise ValueError("source camera translation fit did not converge")
    return result.x


def sample_source_trajectory(trajectory, source_frames):
    return [None if frame is None else trajectory[frame] for frame in source_frames]


def display_ball_transform(track, index, body_names, body_pose, held_config=None, world_offset=(0.,0.,0.)):
    """Display-only held prop; never alters reference-anchored source observations."""
    from mimicx.visualization.racket_calibration import quaternion_matrix_wxyz
    xyz = track['ball_world'][index] if track else None
    result = {'displayed_ball_world':None, 'ball_role':'unobserved', 'held_attachment':None}
    if xyz is None:
        return result
    sourceframe = track['sourceframe'][index]
    if held_config is not None and sourceframe in held_config.get('render_omission_source_frames',[]):
        result['ball_role'] = 'depth_or_identity_ambiguous'
        result['display_omission_reason'] = held_config['render_omission_reason']
        return result
    held = held_config is not None and any(start <= sourceframe <= stop
           for start,stop in held_config['held_source_intervals_inclusive'])
    if held:
        body = held_config['attachment_body']
        pose = np.asarray(body_pose[body_names.index(body)],dtype=float)
        local = np.asarray(held_config['local_offset_m'],dtype=float)
        quaternion = pose[3:] / np.linalg.norm(pose[3:])
        position = pose[:3] + quaternion_matrix_wxyz(quaternion) @ local
        result.update(ball_role='held_left', held_attachment={'body':body,'local_offset_m':local.tolist()})
    else:
        position = np.asarray(xyz,dtype=float)+np.asarray(world_offset,dtype=float)
        result['ball_role'] = 'source_registered'
    result['displayed_ball_world'] = position.tolist()
    return result
