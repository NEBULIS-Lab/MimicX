from pathlib import Path

import numpy as np
import pytest

from mimicx.evaluation.sonic_metrics import compute_sonic_metrics, write_sonic_metrics


def _write_csv(path: Path, values: np.ndarray) -> None:
    values = np.asarray(values)
    header = ",".join(f"c{i}" for i in range(values.shape[1]))
    np.savetxt(path, values, delimiter=",", header=header, comments="")


def _fixture(tmp_path: Path, *, failed: bool = False) -> tuple[Path, Path]:
    reference = tmp_path / "reference"
    reference.mkdir()
    frames = 5
    joint_pos = np.zeros((frames, 2))
    joint_vel = np.zeros((frames, 2))
    body_pos = np.zeros((frames, 2, 3))
    body_pos[:, 0, 2] = 0.8
    body_pos[:, 1, 2] = 0.2
    body_vel = np.zeros_like(body_pos)
    body_quat = np.zeros((frames, 2, 4))
    body_quat[..., 0] = 1.0
    _write_csv(reference / "joint_pos.csv", joint_pos)
    _write_csv(reference / "joint_vel.csv", joint_vel)
    _write_csv(reference / "body_pos.csv", body_pos.reshape(frames, -1))
    _write_csv(reference / "body_quat.csv", body_quat.reshape(frames, -1))
    _write_csv(reference / "body_lin_vel.csv", body_vel.reshape(frames, -1))
    _write_csv(reference / "body_ang_vel.csv", body_vel.reshape(frames, -1))

    actual_body = body_pos.copy()
    if failed:
        actual_body[3:, 0, 0] = 0.75
    state_path = tmp_path / "sim_state.npz"
    np.savez(
        state_path,
        time=np.arange(frames) / 50.0,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos=actual_body,
        body_quat=body_quat,
        body_lin_vel=body_vel,
        body_ang_vel=body_vel,
        body_names=np.asarray(["pelvis", "left_wrist_yaw_link"]),
    )
    return reference, state_path


def test_aligned_rollout_is_strict_success(tmp_path: Path) -> None:
    reference, state = _fixture(tmp_path)
    result = compute_sonic_metrics(reference, state, horizon=5, failure_threshold=0.5)
    assert result["strict_success"] is True
    assert result["first_done_step"] == 5
    assert result["steps_completed"] == 5
    assert result["body_pos_error_mean"] == pytest.approx(0.0)
    assert result["root_pos_error_mean"] == pytest.approx(0.0)
    assert result["contact_error_mean"] is None


def test_failure_frontier_and_horizon_are_reported(tmp_path: Path) -> None:
    reference, state = _fixture(tmp_path, failed=True)
    result = compute_sonic_metrics(reference, state, horizon=5, failure_threshold=0.5)
    assert result["strict_success"] is False
    assert result["first_done_step"] == 3
    assert result["steps_completed"] == 3
    assert result["termination_count"] == 1


def test_nonfinite_state_is_rejected(tmp_path: Path) -> None:
    reference, state = _fixture(tmp_path)
    payload = dict(np.load(state))
    payload["joint_pos"][2, 0] = np.nan
    np.savez(state, **payload)
    with pytest.raises(ValueError, match="non-finite"):
        compute_sonic_metrics(reference, state, horizon=5)


def test_writer_emits_metrics_and_steps(tmp_path: Path) -> None:
    reference, state = _fixture(tmp_path)
    metrics_path, steps_path = write_sonic_metrics(reference, state, tmp_path / "out", horizon=4)
    assert metrics_path.is_file()
    assert steps_path.is_file()
    assert len(steps_path.read_text().strip().splitlines()) == 5
