"""Common rollout metrics for SONIC MuJoCo executions."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np


REFERENCE_FILES = {
    "joint_pos": ("joint_pos.csv", 1),
    "joint_vel": ("joint_vel.csv", 1),
    "body_pos": ("body_pos.csv", 3),
    "body_quat": ("body_quat.csv", 4),
    "body_lin_vel": ("body_lin_vel.csv", 3),
    "body_ang_vel": ("body_ang_vel.csv", 3),
}


def _load_csv(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    data = np.atleast_2d(data).astype(np.float64)
    if data.size == 0 or not np.isfinite(data).all():
        raise ValueError(f"empty or non-finite reference data: {path}")
    return data


def _load_reference(reference_dir: Path) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for key, (filename, width) in REFERENCE_FILES.items():
        data = _load_csv(reference_dir / filename)
        if width > 1:
            if data.shape[1] % width:
                raise ValueError(f"invalid {key} width in {reference_dir / filename}")
            data = data.reshape(data.shape[0], -1, width)
        result[key] = data
    return result


def _best_start(reference: np.ndarray, actual: np.ndarray, horizon: int) -> int:
    """Find playback onset in a recorder stream that includes controller startup."""
    probe = min(25, horizon, reference.shape[0])
    if actual.shape[0] <= horizon:
        return 0
    max_start = actual.shape[0] - horizon
    scores = [
        float(np.mean(np.square(actual[start : start + probe] - reference[:probe])))
        for start in range(max_start + 1)
    ]
    return int(np.argmin(scores))


def _mean_l2(error: np.ndarray) -> np.ndarray:
    return np.linalg.norm(error, axis=-1).mean(axis=-1)


def compute_sonic_metrics(
    reference_dir: Path,
    sim_state_npz: Path,
    *,
    horizon: int | None = None,
    failure_threshold: float = 0.5,
) -> dict[str, object]:
    reference = _load_reference(Path(reference_dir))
    with np.load(sim_state_npz, allow_pickle=False) as archive:
        actual = {key: np.asarray(archive[key], dtype=np.float64) for key in REFERENCE_FILES}
        times = np.asarray(archive["time"], dtype=np.float64)
    for key, value in actual.items():
        if value.size == 0 or not np.isfinite(value).all():
            raise ValueError(f"empty or non-finite SONIC state array: {key}")
    if not np.isfinite(times).all():
        raise ValueError("non-finite SONIC timestamps")

    requested = int(horizon or reference["joint_pos"].shape[0])
    available = min(requested, *(value.shape[0] for value in reference.values()))
    if actual["joint_pos"].shape[0] < available:
        available = actual["joint_pos"].shape[0]
    if available <= 0:
        raise ValueError("no aligned SONIC rollout frames")
    start = _best_start(reference["joint_pos"], actual["joint_pos"], available)
    aligned = {key: value[start : start + available] for key, value in actual.items()}
    for key in reference:
        reference[key] = reference[key][:available]
        if aligned[key].shape != reference[key].shape:
            raise ValueError(
                f"shape mismatch for {key}: reference={reference[key].shape}, actual={aligned[key].shape}"
            )

    joint_pos_step = np.sqrt(np.mean(np.square(aligned["joint_pos"] - reference["joint_pos"]), axis=1))
    joint_vel_step = np.sqrt(np.mean(np.square(aligned["joint_vel"] - reference["joint_vel"]), axis=1))
    body_pos_step = _mean_l2(aligned["body_pos"] - reference["body_pos"])
    body_lin_step = _mean_l2(aligned["body_lin_vel"] - reference["body_lin_vel"])
    body_ang_step = _mean_l2(aligned["body_ang_vel"] - reference["body_ang_vel"])
    root_step = np.linalg.norm(aligned["body_pos"][:, 0] - reference["body_pos"][:, 0], axis=1)
    failed = np.flatnonzero((body_pos_step > failure_threshold) | (root_step > failure_threshold))
    first_done = int(failed[0]) if failed.size else available
    strict_success = bool(not failed.size and available >= requested)

    metrics: dict[str, object] = {
        "schema": "mimicx.sonic-common-metrics.v1",
        "rollout_ok": True,
        "strict_success": strict_success,
        "requested_horizon": requested,
        "evaluated_frames": available,
        "alignment_start_frame": start,
        "steps_completed": first_done,
        "first_done_step": first_done,
        "termination_count": int(bool(failed.size)),
        "joint_pos_error_mean": float(joint_pos_step.mean()),
        "joint_vel_error_mean": float(joint_vel_step.mean()),
        "body_pos_error_mean": float(body_pos_step.mean()),
        "body_pos_error_max": float(body_pos_step.max()),
        "root_pos_error_mean": float(root_step.mean()),
        "root_pos_error_max": float(root_step.max()),
        "body_lin_vel_error_mean": float(body_lin_step.mean()),
        "body_ang_vel_error_mean": float(body_ang_step.mean()),
        "contact_error_mean": None,
        "unsupported_metrics": ["contact_error_mean"],
    }
    metrics["_steps"] = {
        "frame": np.arange(available),
        "joint_pos_error": joint_pos_step,
        "joint_vel_error": joint_vel_step,
        "body_pos_error": body_pos_step,
        "root_pos_error": root_step,
        "body_lin_vel_error": body_lin_step,
        "body_ang_vel_error": body_ang_step,
    }
    return metrics


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def write_sonic_metrics(
    reference_dir: Path,
    sim_state_npz: Path,
    output_dir: Path,
    *,
    horizon: int | None = None,
    failure_threshold: float = 0.5,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    result = compute_sonic_metrics(
        reference_dir, sim_state_npz, horizon=horizon, failure_threshold=failure_threshold
    )
    steps = result.pop("_steps")
    metrics_path = output_dir / "metrics.json"
    steps_path = output_dir / "steps.csv"
    _atomic_text(metrics_path, json.dumps(result, indent=2, sort_keys=True) + "\n")
    columns = list(steps)
    rows = zip(*(steps[name].tolist() for name in columns))
    lines = [",".join(columns)] + [",".join(f"{value:.9g}" for value in row) for row in rows]
    _atomic_text(steps_path, "\n".join(lines) + "\n")
    return metrics_path, steps_path
