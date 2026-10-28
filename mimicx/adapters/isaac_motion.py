"""Strict, CPU-only conversion of named world-frame motion arrays for Isaac Lab.

Body arrays must contain the *complete* articulation, not just tracked bodies.
This adapter reorders existing FK data; it does not compute or certify FK.
"""

from pathlib import Path
import numpy as np


FIELDS = {"joint_pos": None, "joint_vel": None, "body_pos_w": 3,
          "body_quat_w": 4, "body_lin_vel_w": 3, "body_ang_vel_w": 3}


def names(value, label):
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "US":
        raise ValueError(f"{label} must be a one-dimensional string array")
    result = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in array.tolist()]
    if not result or len(set(result)) != len(result) or any(not x for x in result):
        raise ValueError(f"{label} must be nonempty and unique")
    return result


def validate_motion(data, *, joint_names=None, body_names=None, fps=None):
    joints = names(data["joint_names"], "joint_names")
    bodies = names(data["body_names"], "body_names")
    if joint_names is not None and joints != list(joint_names):
        raise ValueError("joint_names do not match runtime articulation order")
    if body_names is not None and bodies != list(body_names):
        raise ValueError("body_names do not match full runtime articulation order")
    rate = np.asarray(data["fps"])
    if rate.size != 1 or not np.isfinite(rate).all() or float(rate.item()) <= 0:
        raise ValueError("fps must be a finite positive scalar")
    rate = float(rate.item())
    if fps is not None and not np.isclose(rate, fps, rtol=0, atol=1e-5):
        raise ValueError(f"Motion fps {rate} != control frequency {fps}; resample before conversion")
    positions = np.asarray(data["joint_pos"])
    if positions.ndim != 2:
        raise ValueError("joint_pos must have shape [T, joints]")
    frames = positions.shape[0]
    if frames < 2:
        raise ValueError("motion must have at least two frames")
    for key, width in FIELDS.items():
        array = np.asarray(data[key])
        expected = (frames, len(joints)) if width is None else (frames, len(bodies), width)
        if array.shape != expected or not np.isfinite(array).all():
            raise ValueError(f"{key} must be finite with shape {expected}")
    if not np.allclose(np.linalg.norm(data["body_quat_w"], axis=-1), 1, atol=1e-3):
        raise ValueError("body_quat_w contains non-unit rotations")
    if "timestamps" in data:
        ts = np.asarray(data["timestamps"])
        if ts.shape != (frames,) or not np.isfinite(ts).all() or not np.allclose(np.diff(ts), 1 / rate, atol=1e-6, rtol=1e-4):
            raise ValueError("timestamps must be uniformly spaced at fps")
    return frames, rate


def convert_motion(source, destination, *, source_joints=None, source_bodies=None,
                   target_joints, target_bodies, quaternion_order="wxyz",
                   units="SI", angular_velocity_frame="world", expected_fps=50.0,
                   linear_velocity_reference="link_origin"):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.exists():
        raise FileExistsError("Conversion must create a new output file")
    if units != "SI" or angular_velocity_frame != "world":
        raise ValueError("Require SI units and world-frame angular velocities; convert at source")
    if linear_velocity_reference != "link_origin":
        raise ValueError("Require link-origin body linear velocities, not COM velocities")
    if quaternion_order not in {"wxyz", "xyzw"}:
        raise ValueError("quaternion_order must be wxyz or xyzw")
    with np.load(source, allow_pickle=False) as raw:
        for key, declared in (("quaternion_order", quaternion_order), ("units", units),
                              ("angular_velocity_frame", angular_velocity_frame),
                              ("velocity_reference", linear_velocity_reference)):
            if key in raw:
                stored = raw[key].item()
                stored = stored.decode("utf-8") if isinstance(stored, bytes) else str(stored)
                if stored != declared:
                    raise ValueError(f"Declared {key} contradicts input metadata")
        data = {key: raw[key] for key in FIELDS.keys() | {"fps"}}
        for key, explicit in (("joint_names", source_joints), ("body_names", source_bodies)):
            if explicit is not None and key in raw and names(raw[key], key) != list(explicit):
                raise ValueError(f"Explicit {key} contradict input metadata")
            if explicit is None and key not in raw:
                raise ValueError(f"Missing {key}; explicit source mapping required")
            data[key] = np.asarray(explicit) if explicit is not None else raw[key]
        if "timestamps" in raw:
            data["timestamps"] = raw["timestamps"]
    validate_motion(data, fps=expected_fps)
    for key, target, family in (("joint_names", target_joints, "joint"), ("body_names", target_bodies, "body")):
        src = names(data[key], key)
        dst = names(target, f"target_{key}")
        if set(src) != set(dst):
            raise ValueError(f"{key} must match the complete target articulation")
        order = [src.index(name) for name in dst]
        for field in FIELDS:
            if field.startswith(family + "_"):
                data[field] = data[field][:, order].astype(np.float32)
        data[key] = np.asarray(dst)
    if quaternion_order == "xyzw":
        data["body_quat_w"] = data["body_quat_w"][..., [3, 0, 1, 2]]
    data["quaternion_order"] = np.asarray("wxyz")
    data["units"] = np.asarray("SI")
    data["angular_velocity_frame"] = np.asarray("world")
    data["velocity_reference"] = np.asarray("link_origin")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        np.savez_compressed(handle, **data)
    return destination
