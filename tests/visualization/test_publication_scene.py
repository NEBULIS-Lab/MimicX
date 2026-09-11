from __future__ import annotations

import numpy as np

from mimicx.visualization.publication_scene import (
    CAMERA_PRESETS,
    TASK_PRESENTATION,
    chronological_opacities,
    registration_offset,
    sequence_layout_offsets,
    validate_clearance,
)


def test_registration_aligns_lowest_visual_vertex_to_surface() -> None:
    lowest = np.asarray([-0.012, 0.031, 0.044, -0.004])

    offset = registration_offset(lowest, surface_z=0.4)

    assert np.isclose(offset, 0.412)
    assert np.isclose((lowest + offset).min(), 0.4)


def test_clearance_gate_rejects_visible_penetration() -> None:
    assert np.isclose(
        validate_clearance(np.asarray([0.399, 0.42]), surface_z=0.4, tolerance=0.002),
        0.001,
    )
    try:
        validate_clearance(np.asarray([0.39, 0.42]), surface_z=0.4, tolerance=0.002)
    except ValueError as error:
        assert "penetration" in str(error)
    else:
        raise AssertionError("penetrated scene should fail validation")


def test_sports_tasks_define_props_surface_and_wide_camera() -> None:
    assert TASK_PRESENTATION["tennis"].surface_z == 0.0
    assert TASK_PRESENTATION["tennis"].prop == "tennis"
    assert TASK_PRESENTATION["football1"].surface_z == 0.4
    assert TASK_PRESENTATION["football1"].prop == "football"
    assert CAMERA_PRESETS["environment_wide"].lens_mm < CAMERA_PRESETS["action_detail"].lens_mm


def test_sequence_layout_spreads_poses_without_changing_height() -> None:
    roots = np.zeros((7, 3), dtype=np.float64)

    tennis = sequence_layout_offsets("tennis", roots)
    football = sequence_layout_offsets("football1", roots)

    assert tennis.shape == (7, 3)
    assert football.shape == (7, 3)
    assert np.allclose(tennis[:, 2], 0.0)
    assert np.allclose(football[:, 2], 0.0)
    assert np.min(np.diff(tennis[:, 0])) >= 1.0
    assert np.min(np.diff(football[:, 0])) >= 1.0


def test_sequence_opacity_increases_from_context_to_current_state() -> None:
    values = chronological_opacities(7)

    assert np.isclose(values[0], 0.18)
    assert np.isclose(values[-1], 1.0)
    assert np.all(np.diff(values) > 0.0)
