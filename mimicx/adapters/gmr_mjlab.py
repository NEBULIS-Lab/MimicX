"""Bridge GMR robot motion pickles to Unitree RL Mjlab motion CSV files."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GmrMotion:
    """Minimal GMR robot motion representation.

    GMR stores root quaternions in xyzw order when saving to pickle. Unitree RL
    Mjlab's csv_to_npz.py expects CSV quaternions in the same xyzw order and
    converts them internally to wxyz.
    """

    fps: float
    root_pos: np.ndarray
    root_rot_xyzw: np.ndarray
    dof_pos: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.root_pos.shape[0])

    @property
    def num_dofs(self) -> int:
        return int(self.dof_pos.shape[1])


def load_gmr_motion(path: str | Path) -> GmrMotion:
    """Load a GMR `.pkl` robot motion file."""

    p = Path(path)
    with p.open("rb") as f:
        data = pickle.load(f)

    required = ("fps", "root_pos", "root_rot", "dof_pos")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{p} is missing required GMR fields: {missing}")

    motion = GmrMotion(
        fps=float(data["fps"]),
        root_pos=np.asarray(data["root_pos"], dtype=np.float32),
        root_rot_xyzw=np.asarray(data["root_rot"], dtype=np.float32),
        dof_pos=np.asarray(data["dof_pos"], dtype=np.float32),
    )
    validate_gmr_motion(motion, source=p)
    return motion


def validate_gmr_motion(motion: GmrMotion, source: str | Path | None = None) -> None:
    """Validate shape consistency before writing backend files."""

    label = f"{source}: " if source is not None else ""
    if motion.root_pos.ndim != 2 or motion.root_pos.shape[1] != 3:
        raise ValueError(f"{label}root_pos must have shape [T, 3], got {motion.root_pos.shape}")
    if motion.root_rot_xyzw.ndim != 2 or motion.root_rot_xyzw.shape[1] != 4:
        raise ValueError(f"{label}root_rot must have shape [T, 4], got {motion.root_rot_xyzw.shape}")
    if motion.dof_pos.ndim != 2:
        raise ValueError(f"{label}dof_pos must have shape [T, DoF], got {motion.dof_pos.shape}")
    frame_counts = {motion.root_pos.shape[0], motion.root_rot_xyzw.shape[0], motion.dof_pos.shape[0]}
    if len(frame_counts) != 1:
        raise ValueError(
            f"{label}frame count mismatch: root_pos={motion.root_pos.shape[0]}, "
            f"root_rot={motion.root_rot_xyzw.shape[0]}, dof_pos={motion.dof_pos.shape[0]}"
        )
    if not np.isfinite(motion.root_pos).all():
        raise ValueError(f"{label}root_pos contains NaN or inf")
    if not np.isfinite(motion.root_rot_xyzw).all():
        raise ValueError(f"{label}root_rot contains NaN or inf")
    if not np.isfinite(motion.dof_pos).all():
        raise ValueError(f"{label}dof_pos contains NaN or inf")


def gmr_motion_to_mjlab_csv_array(motion: GmrMotion) -> np.ndarray:
    """Return Unitree RL Mjlab-compatible CSV array.

    Output columns are:
    root position xyz, root quaternion xyzw, joint positions.
    """

    validate_gmr_motion(motion)
    return np.concatenate([motion.root_pos, motion.root_rot_xyzw, motion.dof_pos], axis=1).astype(np.float32)


def save_mjlab_csv(motion: GmrMotion, output_path: str | Path, delimiter: str = ",") -> Path:
    """Write a Unitree RL Mjlab-compatible CSV file."""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, gmr_motion_to_mjlab_csv_array(motion), delimiter=delimiter)
    return out


def convert_gmr_pkl_to_mjlab_csv(input_path: str | Path, output_path: str | Path) -> Path:
    """Convert one GMR pickle to one Unitree RL Mjlab-compatible CSV."""

    return save_mjlab_csv(load_gmr_motion(input_path), output_path)
