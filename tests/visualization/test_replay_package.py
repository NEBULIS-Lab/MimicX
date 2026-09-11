from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mimicx.visualization.replay_package import REPLAY_ARRAY_NAMES, write_replay_package


def test_write_replay_package_preserves_exact_arrays_and_manifest(tmp_path: Path) -> None:
    frames = 4
    arrays = {
        "timestamps": np.arange(frames, dtype=np.float64) / 50.0,
        "robot_root_pose": np.zeros((frames, 7), dtype=np.float32),
        "robot_joint_positions": np.zeros((frames, 3), dtype=np.float32),
        "robot_body_pose": np.zeros((frames, 2, 7), dtype=np.float32),
        "reference_root_pose": np.ones((frames, 7), dtype=np.float32),
        "reference_joint_positions": np.ones((frames, 3), dtype=np.float32),
        "reference_body_pose": np.ones((frames, 2, 7), dtype=np.float32),
    }
    output = tmp_path / "replay"

    manifest = write_replay_package(
        output,
        arrays=arrays,
        metadata={"task": "tennis", "checkpoint": "/tmp/model.pt", "fps": 50.0},
        body_names=["pelvis", "torso"],
        joint_names=["j0", "j1", "j2"],
    )

    assert manifest["frames"] == frames
    assert manifest["body_count"] == 2
    assert manifest["joint_count"] == 3
    assert set(manifest["arrays"]) == set(REPLAY_ARRAY_NAMES)
    assert json.loads((output / "manifest.json").read_text())["task"] == "tennis"
    for name, expected in arrays.items():
        np.testing.assert_array_equal(np.load(output / f"{name}.npy"), expected)


def test_write_replay_package_rejects_inconsistent_frame_counts(tmp_path: Path) -> None:
    arrays = {
        "timestamps": np.zeros(4),
        "robot_root_pose": np.zeros((3, 7)),
        "robot_joint_positions": np.zeros((4, 1)),
        "robot_body_pose": np.zeros((4, 1, 7)),
        "reference_root_pose": np.zeros((4, 7)),
        "reference_joint_positions": np.zeros((4, 1)),
        "reference_body_pose": np.zeros((4, 1, 7)),
    }

    try:
        write_replay_package(
            tmp_path / "bad",
            arrays=arrays,
            metadata={"task": "tennis"},
            body_names=["pelvis"],
            joint_names=["j0"],
        )
    except ValueError as error:
        assert "frame count" in str(error)
    else:
        raise AssertionError("inconsistent replay arrays must be rejected")


def test_write_replay_package_supports_full_robot_and_tracked_reference_bodies(tmp_path: Path) -> None:
    frames = 2
    arrays = {
        "timestamps": np.zeros(frames),
        "robot_root_pose": np.zeros((frames, 7)),
        "robot_joint_positions": np.zeros((frames, 1)),
        "robot_body_pose": np.zeros((frames, 3, 7)),
        "reference_root_pose": np.zeros((frames, 7)),
        "reference_joint_positions": np.zeros((frames, 1)),
        "reference_body_pose": np.zeros((frames, 1, 7)),
    }

    manifest = write_replay_package(
        tmp_path / "subset",
        arrays=arrays,
        metadata={"task": "tennis"},
        body_names=["pelvis", "left_knee", "right_knee"],
        reference_body_names=["pelvis"],
        joint_names=["j0"],
    )

    assert manifest["robot_body_count"] == 3
    assert manifest["reference_body_count"] == 1
    assert json.loads((tmp_path / "subset/reference_body_names.json").read_text()) == ["pelvis"]
