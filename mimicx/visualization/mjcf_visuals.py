from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class VisualGeom:
    body_name: str
    mesh_name: str
    mesh_path: Path
    material: str
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]


def _floats(value: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split()) if value else default


def parse_visual_geoms(xml_path: Path) -> list[VisualGeom]:
    """Resolve mesh visuals and their owning body/local transform from a MuJoCo XML."""
    xml_path = xml_path.expanduser().resolve()
    root = ET.parse(xml_path).getroot()
    compiler = root.find("compiler")
    mesh_dir = xml_path.parent / (compiler.get("meshdir", "") if compiler is not None else "")
    mesh_files = {
        mesh.get("name"): (mesh_dir / mesh.get("file", "")).resolve()
        for mesh in root.findall("./asset/mesh")
    }
    geoms: list[VisualGeom] = []

    def visit(body: ET.Element) -> None:
        body_name = body.get("name")
        if not body_name:
            raise ValueError("visual body without a name")
        for geom in body.findall("geom"):
            mesh_name = geom.get("mesh")
            if not mesh_name:
                continue
            if mesh_name not in mesh_files:
                raise ValueError(f"unknown mesh {mesh_name} in {xml_path}")
            geoms.append(
                VisualGeom(
                    body_name=body_name,
                    mesh_name=mesh_name,
                    mesh_path=mesh_files[mesh_name],
                    material=geom.get("material", "silver"),
                    position=_floats(geom.get("pos"), (0.0, 0.0, 0.0)),
                    quaternion=_floats(geom.get("quat"), (1.0, 0.0, 0.0, 0.0)),
                )
            )
        for child in body.findall("body"):
            visit(child)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"missing worldbody in {xml_path}")
    for body in worldbody.findall("body"):
        visit(body)
    return geoms
