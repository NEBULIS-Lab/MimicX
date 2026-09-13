from __future__ import annotations

from pathlib import Path

import numpy as np

from mimicx.visualization.dex3_grip import (
    POWER_GRASP_JOINTS,
    load_closed_grip_visuals,
)


def test_production_dex3_grip_contains_palm_and_seven_finger_links() -> None:
    root = Path(__file__).resolve().parents[2]
    visuals = load_closed_grip_visuals(
        root / "dependencies/assets/unitree_dex3_grip/g1_29dof_with_hand_rev_1_0.xml"
    )

    assert len(visuals) == 8
    assert {visual.mesh_name for visual in visuals} == {
        "right_hand_palm_link",
        "right_hand_thumb_0_link",
        "right_hand_thumb_1_link",
        "right_hand_thumb_2_link",
        "right_hand_middle_0_link",
        "right_hand_middle_1_link",
        "right_hand_index_0_link",
        "right_hand_index_1_link",
    }
    palm = next(visual for visual in visuals if visual.mesh_name == "right_hand_palm_link")
    assert np.allclose(palm.local_matrix[:3, 3], [0.0415, -0.003, 0.0])
    assert all(visual.mesh_path.is_file() for visual in visuals)


def test_power_grasp_matches_public_g1_closed_hand_targets() -> None:
    assert POWER_GRASP_JOINTS == {
        "right_hand_thumb_0_joint": 0.8,
        "right_hand_thumb_1_joint": -0.9,
        "right_hand_thumb_2_joint": -1.5,
        "right_hand_index_0_joint": 1.4,
        "right_hand_index_1_joint": 1.5,
        "right_hand_middle_0_joint": 1.4,
        "right_hand_middle_1_joint": 1.5,
    }
    visuals = load_closed_grip_visuals(
        Path(__file__).resolve().parents[2]
        / "dependencies/assets/unitree_dex3_grip/g1_29dof_with_hand_rev_1_0.xml"
    )
    assert all(np.isfinite(visual.local_matrix).all() for visual in visuals)
