from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraPreset:
    lens_mm: float
    target_height: float
    expected_robot_fraction: tuple[float, float]


@dataclass(frozen=True)
class TaskPresentation:
    surface_z: float
    prop: str
    event_step: int
    prop_anchor: tuple[float, float, float]
    wide_target: tuple[float, float, float]
    wide_location: tuple[float, float, float]
    action_target: tuple[float, float, float]
    action_location: tuple[float, float, float]
    sequence_target: tuple[float, float, float]
    sequence_location: tuple[float, float, float]


CAMERA_PRESETS = {
    "environment_wide": CameraPreset(
        lens_mm=42.0,
        target_height=0.9,
        expected_robot_fraction=(0.18, 0.42),
    ),
    "action_detail": CameraPreset(
        lens_mm=58.0,
        target_height=0.9,
        expected_robot_fraction=(0.38, 0.68),
    ),
    "motion_sequence": CameraPreset(
        lens_mm=46.0,
        target_height=0.9,
        expected_robot_fraction=(0.16, 0.38),
    ),
}


TASK_PRESENTATION = {
    "tennis": TaskPresentation(
        surface_z=0.0,
        prop="tennis",
        event_step=201,
        prop_anchor=(0.66, -7.04, 1.16),
        wide_target=(0.0, -3.6, 0.85),
        wide_location=(8.4, -15.2, 6.0),
        action_target=(0.0, -7.35, 0.88),
        action_location=(4.8, -13.2, 3.45),
        sequence_target=(0.0, -6.9, 0.85),
        sequence_location=(0.0, -17.0, 6.2),
    ),
    "football1": TaskPresentation(
        surface_z=0.4,
        prop="football",
        event_step=225,
        prop_anchor=(0.0, 0.0, 0.51),
        wide_target=(3.2, 0.0, 0.95),
        wide_location=(7.2, -12.5, 5.5),
        action_target=(-0.2, 0.0, 0.92),
        action_location=(4.0, -7.2, 3.2),
        sequence_target=(2.5, 0.0, 0.92),
        sequence_location=(2.5, -18.0, 6.2),
    ),
}


def registration_offset(lowest_visual_z: np.ndarray, *, surface_z: float) -> float:
    values = np.asarray(lowest_visual_z, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("lowest_visual_z must be one finite nonempty vector")
    return float(surface_z - values.min())


def validate_clearance(
    lowest_visual_z: np.ndarray,
    *,
    surface_z: float,
    tolerance: float = 0.02,
) -> float:
    values = np.asarray(lowest_visual_z, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("lowest_visual_z must be one finite nonempty vector")
    penetration = float(max(0.0, surface_z - values.min()))
    if penetration > tolerance:
        raise ValueError(
            f"visible robot penetration {penetration:.4f} m exceeds tolerance {tolerance:.4f} m"
        )
    return penetration


def sequence_layout_offsets(task: str, selected_roots: np.ndarray) -> np.ndarray:
    roots = np.asarray(selected_roots, dtype=np.float64)
    if roots.ndim != 2 or roots.shape[1] != 3 or roots.shape[0] < 2:
        raise ValueError("selected_roots must have shape [N, 3] with N >= 2")
    count = roots.shape[0]
    if task == "tennis":
        target_x = np.linspace(-3.3, 3.3, count)
        target_y = -7.2 + 0.22 * np.sin(np.linspace(-0.7, 0.7, count) * np.pi)
    elif task == "football1":
        target_x = np.linspace(-2.5, 4.5, count)
        target_y = np.linspace(-0.75, 0.65, count)
    else:
        raise ValueError(f"no motion-sequence layout for task: {task}")
    targets = np.column_stack((target_x, target_y, roots[:, 2]))
    offsets = targets - roots
    offsets[:, 2] = 0.0
    return offsets


def chronological_opacities(count: int, *, first: float = 0.18) -> np.ndarray:
    if count < 2:
        raise ValueError("chronological sequence requires at least two states")
    if not 0.0 < first < 1.0:
        raise ValueError("first opacity must be between zero and one")
    return np.linspace(first, 1.0, count, dtype=np.float64)
