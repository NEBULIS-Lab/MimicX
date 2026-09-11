from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from mimicx.visualization.racket_calibration import quaternion_matrix_wxyz


POWER_GRASP_JOINTS = {
    "right_hand_thumb_0_joint": 0.8,
    "right_hand_thumb_1_joint": -0.9,
    "right_hand_thumb_2_joint": -1.5,
    "right_hand_index_0_joint": 1.4,
    "right_hand_index_1_joint": 1.5,
    "right_hand_middle_0_joint": 1.4,
    "right_hand_middle_1_joint": 1.5,
}


@dataclass(frozen=True)
class GripVisual:
    mesh_name: str
    mesh_path: Path
    local_matrix: np.ndarray


def _floats(value: str | None, default: tuple[float, ...]) -> np.ndarray:
    return np.asarray(tuple(float(item) for item in value.split()) if value else default)


def _transform(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_matrix_wxyz(quaternion)
    matrix[:3, 3] = position
    return matrix


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    direction = np.asarray(axis, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    x, y, z = direction
    c, s, one_minus_c = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    rotation = np.asarray(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    return matrix


def _find_body(root: ET.Element, name: str) -> ET.Element:
    for body in root.findall(".//body"):
        if body.get("name") == name:
            return body
    raise ValueError(f"missing MJCF body: {name}")


def load_closed_grip_visuals(xml_path: str | Path) -> tuple[GripVisual, ...]:
    source = Path(xml_path).expanduser().resolve()
    root = ET.parse(source).getroot()
    compiler = root.find("compiler")
    mesh_dir = source.parent / (compiler.get("meshdir", "") if compiler is not None else "")
    mesh_files = {
        mesh.get("name"): (mesh_dir / mesh.get("file", "")).resolve()
        for mesh in root.findall("./asset/mesh")
    }
    wrist = _find_body(root, "right_wrist_yaw_link")
    visuals: dict[str, GripVisual] = {}

    def add_visuals(body: ET.Element, body_matrix: np.ndarray) -> None:
        for geom in body.findall("geom"):
            mesh_name = geom.get("mesh")
            if not mesh_name or not mesh_name.startswith("right_hand_") or mesh_name in visuals:
                continue
            if mesh_name not in mesh_files:
                raise ValueError(f"unknown hand mesh {mesh_name}")
            geom_matrix = _transform(
                _floats(geom.get("pos"), (0.0, 0.0, 0.0)),
                _floats(geom.get("quat"), (1.0, 0.0, 0.0, 0.0)),
            )
            visuals[mesh_name] = GripVisual(mesh_name, mesh_files[mesh_name], body_matrix @ geom_matrix)

    def visit(body: ET.Element, parent_matrix: np.ndarray) -> None:
        body_matrix = parent_matrix @ _transform(
            _floats(body.get("pos"), (0.0, 0.0, 0.0)),
            _floats(body.get("quat"), (1.0, 0.0, 0.0, 0.0)),
        )
        joint = body.find("joint")
        if joint is not None and joint.get("name") in POWER_GRASP_JOINTS:
            joint_position = _floats(joint.get("pos"), (0.0, 0.0, 0.0))
            before = _transform(joint_position, np.asarray((1.0, 0.0, 0.0, 0.0)))
            after = _transform(-joint_position, np.asarray((1.0, 0.0, 0.0, 0.0)))
            body_matrix = body_matrix @ before @ _axis_rotation(
                _floats(joint.get("axis"), (0.0, 0.0, 1.0)),
                POWER_GRASP_JOINTS[joint.get("name")],
            ) @ after
        add_visuals(body, body_matrix)
        for child in body.findall("body"):
            if child.get("name", "").startswith("right_hand_"):
                visit(child, body_matrix)

    add_visuals(wrist, np.eye(4, dtype=np.float64))
    for child in wrist.findall("body"):
        if child.get("name", "").startswith("right_hand_"):
            visit(child, np.eye(4, dtype=np.float64))
    return tuple(visuals[name] for name in sorted(visuals))
