from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPLAY_ARRAY_NAMES = (
    "timestamps",
    "robot_root_pose",
    "robot_joint_positions",
    "robot_body_pose",
    "reference_root_pose",
    "reference_joint_positions",
    "reference_body_pose",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate(
    arrays: Mapping[str, np.ndarray],
    body_names: Sequence[str],
    reference_body_names: Sequence[str],
    joint_names: Sequence[str],
) -> int:
    missing = sorted(set(REPLAY_ARRAY_NAMES) - set(arrays))
    if missing:
        raise ValueError(f"missing replay arrays: {missing}")
    frame_counts = {name: int(np.asarray(arrays[name]).shape[0]) for name in REPLAY_ARRAY_NAMES}
    if len(set(frame_counts.values())) != 1:
        raise ValueError(f"inconsistent replay frame count: {frame_counts}")
    frames = next(iter(frame_counts.values()))
    if np.asarray(arrays["robot_body_pose"]).shape != (frames, len(body_names), 7):
        raise ValueError("robot_body_pose shape does not match body names")
    if np.asarray(arrays["reference_body_pose"]).shape != (frames, len(reference_body_names), 7):
        raise ValueError("reference_body_pose shape does not match reference body names")
    if np.asarray(arrays["robot_joint_positions"]).shape != (frames, len(joint_names)):
        raise ValueError("robot_joint_positions shape does not match joint names")
    if np.asarray(arrays["reference_joint_positions"]).shape != (frames, len(joint_names)):
        raise ValueError("reference_joint_positions shape does not match joint names")
    return frames


def write_replay_package(
    output: Path,
    *,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    body_names: Sequence[str],
    joint_names: Sequence[str],
    reference_body_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Write an exact simulator-state replay package for offline rendering."""
    reference_body_names = list(reference_body_names or body_names)
    frames = _validate(arrays, body_names, reference_body_names, joint_names)
    output.mkdir(parents=True, exist_ok=True)
    array_records: dict[str, dict[str, Any]] = {}
    for name in REPLAY_ARRAY_NAMES:
        path = output / f"{name}.npy"
        array = np.asarray(arrays[name])
        np.save(path, array, allow_pickle=False)
        array_records[name] = {
            "filename": path.name,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": _sha256(path),
        }
    (output / "body_names.json").write_text(json.dumps(list(body_names), indent=2) + "\n", encoding="utf-8")
    (output / "reference_body_names.json").write_text(
        json.dumps(list(reference_body_names), indent=2) + "\n", encoding="utf-8"
    )
    (output / "joint_names.json").write_text(json.dumps(list(joint_names), indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema": "mimicx.exact-rollout-replay.v1",
        **dict(metadata),
        "frames": frames,
        "body_count": len(body_names),
        "robot_body_count": len(body_names),
        "reference_body_count": len(reference_body_names),
        "joint_count": len(joint_names),
        "quaternion_order": "wxyz",
        "coordinate_system": "MuJoCo world frame; meters; z-up",
        "arrays": array_records,
        "body_names": "body_names.json",
        "reference_body_names": "reference_body_names.json",
        "joint_names": "joint_names.json",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
