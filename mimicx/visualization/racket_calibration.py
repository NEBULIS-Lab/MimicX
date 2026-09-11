from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCHEMA = "mimicx.tennis-racket-calibration.v1"
ATTACHMENT_SCHEMA = "mimicx.tennis-racket-attachment.v2"


def _vector(payload: dict[str, object], key: str, length: int) -> np.ndarray:
    value = np.asarray(payload.get(key), dtype=np.float64)
    if value.shape != (length,) or not np.isfinite(value).all():
        raise ValueError(f"{key} must contain {length} finite values")
    return value


def quaternion_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class RacketCalibration:
    schema: str
    version: int
    wrist_body: str
    translation_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    uniform_scale: float
    racket_axis_local: tuple[float, float, float]
    grip_point_local_m: tuple[float, float, float]
    source_video: str
    source_frame_count: int
    replay_frame_count: int
    annotations: tuple[dict[str, object], ...]
    median_projected_axis_error_deg: float | None
    max_projected_axis_error_deg: float | None = None
    asset_path: str = ""
    asset_sha256: str = ""
    mount_authority: str = ""
    asset_mesh_grip_local_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    asset_to_racket_quaternion_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


def load_racket_calibration(path: str | Path) -> RacketCalibration:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    schema = str(payload.get("schema"))
    if schema not in {SCHEMA, ATTACHMENT_SCHEMA}:
        raise ValueError(f"unsupported racket calibration schema: {payload.get('schema')}")
    translation = _vector(payload, "translation_m", 3)
    quaternion = _vector(payload, "quaternion_wxyz", 4)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-8:
        raise ValueError("quaternion_wxyz must have nonzero norm")
    quaternion /= norm
    axis = _vector(payload, "racket_axis_local", 3)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-8:
        raise ValueError("racket_axis_local must have nonzero norm")
    axis /= axis_norm
    grip = _vector(payload, "grip_point_local_m", 3)
    scale = float(payload.get("uniform_scale", 0.0))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("uniform_scale must be positive and finite")
    raw_error = payload.get("median_projected_axis_error_deg")
    error = None if raw_error is None else float(raw_error)
    if error is not None and (not np.isfinite(error) or error < 0.0):
        raise ValueError("median_projected_axis_error_deg must be finite and nonnegative")
    raw_max_error = payload.get("max_projected_axis_error_deg")
    max_error = None if raw_max_error is None else float(raw_max_error)
    if max_error is not None and (not np.isfinite(max_error) or max_error < 0.0):
        raise ValueError("max_projected_axis_error_deg must be finite and nonnegative")
    mesh_grip = _vector(payload, "asset_mesh_grip_local_m", 3) if "asset_mesh_grip_local_m" in payload else np.zeros(3)
    asset_rotation = _vector(payload, "asset_to_racket_quaternion_wxyz", 4) if "asset_to_racket_quaternion_wxyz" in payload else np.asarray([1.0, 0.0, 0.0, 0.0])
    asset_rotation_norm = float(np.linalg.norm(asset_rotation))
    if asset_rotation_norm <= 1e-8:
        raise ValueError("asset_to_racket_quaternion_wxyz must have nonzero norm")
    asset_rotation /= asset_rotation_norm
    annotations = payload.get("annotations", [])
    if not isinstance(annotations, list):
        raise ValueError("annotations must be a list")
    return RacketCalibration(
        schema=schema,
        version=int(payload.get("version", 0)),
        wrist_body=str(payload.get("wrist_body", "")),
        translation_m=tuple(float(value) for value in translation),
        quaternion_wxyz=tuple(float(value) for value in quaternion),
        uniform_scale=scale,
        racket_axis_local=tuple(float(value) for value in axis),
        grip_point_local_m=tuple(float(value) for value in grip),
        source_video=str(payload.get("source_video", "")),
        source_frame_count=int(payload.get("source_frame_count", 0)),
        replay_frame_count=int(payload.get("replay_frame_count", 0)),
        annotations=tuple(dict(row) for row in annotations),
        median_projected_axis_error_deg=error,
        max_projected_axis_error_deg=max_error,
        asset_path=str(payload.get("asset_path", "")),
        asset_sha256=str(payload.get("asset_sha256", "")),
        mount_authority=str(payload.get("mount_authority", "")),
        asset_mesh_grip_local_m=tuple(float(value) for value in mesh_grip),
        asset_to_racket_quaternion_wxyz=tuple(float(value) for value in asset_rotation),
    )


def racket_world_matrix(
    body_pose: np.ndarray,
    wrist_index: int,
    calibration: RacketCalibration,
) -> np.ndarray:
    poses = np.asarray(body_pose, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError("body_pose must have shape [body, 7]")
    if wrist_index < 0 or wrist_index >= len(poses):
        raise IndexError("wrist_index outside body_pose")
    wrist = poses[wrist_index]
    body = np.eye(4, dtype=np.float64)
    body[:3, :3] = quaternion_matrix_wxyz(wrist[3:])
    body[:3, 3] = wrist[:3]
    mount = np.eye(4, dtype=np.float64)
    mount[:3, :3] = quaternion_matrix_wxyz(np.asarray(calibration.quaternion_wxyz))
    mount[:3, 3] = np.asarray(calibration.translation_m)
    scale = np.diag([calibration.uniform_scale] * 3 + [1.0])
    grip_offset = np.eye(4, dtype=np.float64)
    grip_offset[:3, 3] = -np.asarray(calibration.grip_point_local_m)
    return body @ mount @ scale @ grip_offset


def projected_axis_error_degrees(predicted_endpoints: np.ndarray, observed_endpoints: np.ndarray) -> float:
    predicted = np.asarray(predicted_endpoints, dtype=np.float64)
    observed = np.asarray(observed_endpoints, dtype=np.float64)
    if predicted.shape != (2, 2) or observed.shape != (2, 2):
        raise ValueError("endpoint arrays must have shape [2, 2]")
    predicted_axis = predicted[1] - predicted[0]
    observed_axis = observed[1] - observed[0]
    predicted_norm = float(np.linalg.norm(predicted_axis))
    observed_norm = float(np.linalg.norm(observed_axis))
    if predicted_norm <= 1e-8 or observed_norm <= 1e-8:
        raise ValueError("projected axes must have nonzero length")
    cosine = float(np.dot(predicted_axis, observed_axis) / (predicted_norm * observed_norm))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def included_error_objective(
    errors: dict[int, float],
    annotations: tuple[dict[str, object], ...],
) -> float:
    values = np.asarray(
        [errors[int(row["source_frame"])] for row in annotations if bool(row.get("included"))],
        dtype=np.float64,
    )
    if values.size == 0:
        raise ValueError("at least one included racket annotation is required")
    return float(values.max())
