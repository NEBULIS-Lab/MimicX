"""Opt-in neutral contrast for saved-state rendering, never an experiment model."""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

FLOOR = "mimicx_display_floor"


def _equal(left, right, name):
    if not np.array_equal(left, right):
        raise ValueError(f"Display model invariant changed: {name}")


def load_contrast_model(model_path):
    """Compile a display-only wrapper without writing or modifying source assets.

    Standalone MJCF with mesh/texture assets is supported. Includes, plugins,
    attachments, existing planes and skyboxes require an explicitly prepared
    scene instead. The added world plane cannot contact or enter dynamic ghosts.
    CPU compilation/FK checks do not claim a visual review or new physics result.
    """
    model_path = Path(model_path).resolve(strict=True)
    root = ET.parse(model_path).getroot()
    if (root.tag != "mujoco" or any(root.findall(f".//{tag}") for tag in ("include", "attach", "plugin"))):
        raise ValueError("Contrast requires standalone MJCF without includes, attachments or plugins")
    if (root.findall(".//geom[@type='plane']") or root.findall(".//texture[@type='skybox']")
            or any(e.get("name", "").startswith("mimicx_display_") for e in root.iter())):
        raise ValueError("Existing floor, skybox or display names require an explicit display scene")
    compiler = root.find("compiler")
    attrs = {} if compiler is None else compiler.attrib
    if attrs.get("strippath", "false") != "false":
        raise ValueError("Contrast does not support strippath asset rewriting")
    if any(e.get("file") and e.tag not in ("mesh", "texture") for e in root.findall("./asset/*")):
        raise ValueError("Contrast supports only mesh and texture file assets")
    meshdir = (model_path.parent / attrs.get("meshdir", attrs.get("assetdir", ""))).resolve()
    texturedir = (model_path.parent / attrs.get("texturedir", attrs.get("assetdir", ""))).resolve()
    wrapper = ET.Element("mujoco", model="mimicx_display_only")
    ET.SubElement(wrapper, "include", file=str(model_path))
    ET.SubElement(wrapper, "compiler", meshdir=str(meshdir), texturedir=str(texturedir))
    visual = ET.SubElement(wrapper, "visual")
    ET.SubElement(visual, "headlight", active="1", ambient="0.45 0.45 0.45",
                  diffuse="0.65 0.65 0.65", specular="0.1 0.1 0.1")
    asset = ET.SubElement(wrapper, "asset")
    ET.SubElement(asset, "texture", name="mimicx_display_sky", type="skybox", builtin="flat",
                  rgb1="0.78 0.80 0.82", width="256", height="1536")
    world = ET.SubElement(wrapper, "worldbody")
    ET.SubElement(world, "geom", name=FLOOR, type="plane", pos="0 0 0", quat="1 0 0 0",
                  size="0 0 0.05", rgba="0.68 0.70 0.72 1", material="",
                  contype="0", conaffinity="0", density="0", group="0")
    import mujoco

    original = mujoco.MjModel.from_xml_path(str(model_path))
    enhanced = mujoco.MjModel.from_xml_string(ET.tostring(wrapper, encoding="unicode"))
    for name in ("nq", "nv", "njnt", "nbody", "nu", "na", "nsite", "ncam", "nlight",
                 "nmesh", "ntendon", "nsensor", "neq", "npair", "nexclude"):
        _equal(getattr(original, name), getattr(enhanced, name), name)
    _equal(enhanced.ngeom, original.ngeom + 1, "ngeom")
    floor = enhanced.geom(FLOOR).id
    _equal([enhanced.geom_bodyid[floor], enhanced.geom_contype[floor], enhanced.geom_conaffinity[floor]],
           [0, 0, 0], "contact-free world floor")
    mapping = np.delete(np.arange(enhanced.ngeom), floor)
    # World geometry changes address/BVH bookkeeping, not robot physics.
    excluded = {"body_geomadr", "body_geomnum", "body_bvhadr", "body_bvhnum"}
    prefixes = ("geom_", "body_", "jnt_", "dof_", "actuator_", "mesh_", "site_", "cam_", "light_")
    checked = []
    for name in dir(original):
        if name in excluded or not name.startswith(prefixes):
            continue
        left, right = getattr(original, name), getattr(enhanced, name)
        if isinstance(left, np.ndarray):
            _equal(left, right[mapping] if name.startswith("geom_") else right, name)
            checked.append(name)
    for name in ("qpos0", "qpos_spring"):
        _equal(getattr(original, name), getattr(enhanced, name), name)
    for name in dir(original.opt):
        value = getattr(original.opt, name)
        if not name.startswith("_") and isinstance(value, (int, float, np.ndarray)):
            _equal(value, getattr(enhanced.opt, name), "option." + name)
    a, b = mujoco.MjData(original), mujoco.MjData(enhanced)
    mujoco.mj_forward(original, a)
    mujoco.mj_forward(enhanced, b)
    for name in ("xpos", "xquat", "xmat", "xipos", "ximat", "subtree_com", "site_xpos", "site_xmat"):
        _equal(getattr(a, name), getattr(b, name), "FK." + name)
    _equal(a.geom_xpos, b.geom_xpos[mapping], "FK.geom_xpos")
    _equal(a.geom_xmat, b.geom_xmat[mapping], "FK.geom_xmat")
    scene = mujoco.MjvScene(enhanced, maxgeom=2 * enhanced.ngeom + 100)
    mujoco.mjv_addGeoms(enhanced, b, mujoco.MjvOption(), mujoco.MjvPerturb(),
                      mujoco.mjtCatBit.mjCAT_DYNAMIC, scene)
    if any(g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid == floor for g in scene.geoms[:scene.ngeom]):
        raise ValueError("Display floor entered dynamic ghost geometry")
    return enhanced, dict(compared_arrays=checked, poses_checked=1, max_pose_delta=0.,
                          dynamic_ghost_excludes_floor=True, physics_steps=0, rendered=False,
                          display_only=True, floor_z=0., visual_review="pending")
