"""Controller-neutral recording guards. Importing this module needs only NumPy."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time
from types import MethodType

import numpy as np


def require_forward_time(previous: float, current: float) -> None:
    if not math.isfinite(previous) or not math.isfinite(current) or current <= previous:
        raise ValueError(f"Simulator clock reset/stalled: {previous} -> {current}")


def validate_times(values, name="clock") -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
        raise ValueError(f"{name} must be finite and strictly increasing with at least two samples")
    return values


def joint_permutation(source_names, target_names) -> np.ndarray:
    source, target = list(source_names), list(target_names)
    if (len(set(source)) != len(source) or len(set(target)) != len(target)
            or set(source) != set(target)):
        raise ValueError("Joint names must form a complete, unique bijection")
    return np.asarray([source.index(name) for name in target], dtype=int)


def named_joint_addresses(model, names, *, require_complete=True):
    """Resolve scalar joints by name, not by actuator or backend storage order."""
    names = list(names)
    if len(set(names)) != len(names):
        raise ValueError("Duplicate joint names")
    try:
        ids = np.asarray([model.joint(name).id for name in names], dtype=int)
    except KeyError as exc:
        raise ValueError(f"Unknown joint: {exc}") from exc
    # MuJoCo joint types: free=0, ball=1, slide=2, hinge=3.
    scalar = np.flatnonzero(np.isin(model.jnt_type, [2, 3]))
    if not set(ids).issubset(set(scalar)) or (require_complete and set(ids) != set(scalar)):
        raise ValueError("Named joint mapping must cover the model's scalar joints")
    return np.asarray(model.jnt_qposadr[ids]), np.asarray(model.jnt_dofadr[ids])


def install_no_reset_fall_monitor(sim_env, *, height_threshold=.2):
    if not math.isfinite(height_threshold) or height_threshold <= 0:
        raise ValueError("Fall height threshold must be positive")

    def check_fall(env):
        env.fall = bool(env.mj_data.qpos[2] < height_threshold)

    sim_env.check_fall = MethodType(check_fall, sim_env)


def release_if_requested(sim_env, request_path) -> bool:
    """Disable startup support and clear its residual wrench before acknowledging.

    Use a fresh request filename per run. An acknowledgement from a previous run
    must not silently authorize playback while the band has been enabled again.
    Unrelated applied forces are not erased.
    """
    if not request_path:
        return False
    request = Path(request_path)
    if not request.is_file():
        return False
    ack = Path(str(request) + ".ack.json")
    band = getattr(sim_env, "elastic_band", None)
    enabled = bool(band is not None and band.enable)
    if ack.exists():
        residual = (band is not None and np.any(
            sim_env.mj_data.xfrc_applied[sim_env.band_attached_link] != 0))
        if enabled or residual:
            raise ValueError("Stale release acknowledgement while startup support is active")
        return False
    if band is not None:
        sim_env.mj_data.xfrc_applied[sim_env.band_attached_link] = 0
        band.enable = False
    record = dict(previously_enabled=enabled, enabled=False,
                  simulation_time=float(sim_env.mj_data.time), monotonic_time=time.monotonic())
    temporary = ack.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(ack)
    return True


def validate_playback(states, playback_wall, *, clock_audit=None, ratio_bounds=(.97, 1.03),
                      wrench_tolerance=1e-8, joint_rmse_tolerance=.1):
    """Audit actual simulator coverage and support, including interpolation brackets.

    Returns elapsed physical playback time, never a fabricated nominal logger
    timeline. A supplied legacy audit is checked in addition to measured clocks.
    """
    required = {"time", "monotonic_time", "elastic_band_enabled", "external_wrench_max"}
    if not required.issubset(states):
        raise ValueError(f"Missing recording fields: {sorted(required - set(states))}")
    sim = validate_times(states["time"], "simulator clock")
    wall = validate_times(states["monotonic_time"], "state wall clock")
    play = validate_times(playback_wall, "playback wall clock")
    if len(sim) != len(wall) or play[0] < wall[0] or play[-1] > wall[-1]:
        raise ValueError("Incomplete simulator coverage of playback")
    low = max(0, int(np.searchsorted(wall, play[0], side="right")) - 1)
    high = min(len(wall), int(np.searchsorted(wall, play[-1], side="left")) + 1)
    band = np.asarray(states["elastic_band_enabled"])
    force = np.asarray(states["external_wrench_max"], dtype=float)
    if (band.shape != wall.shape or force.shape != wall.shape
            or not np.isfinite(band).all() or not np.isfinite(force).all()
            or np.any(force < 0) or not np.isin(band, [0, 1]).all()):
        raise ValueError("Invalid startup-force telemetry")
    if np.any(band[low:high]) or np.any(force[low:high] > wrench_tolerance):
        raise ValueError("External startup support during playback")
    physical = np.interp(play, wall, sim)
    ratio = float((physical[-1] - physical[0]) / (play[-1] - play[0]))
    if not ratio_bounds[0] <= ratio <= ratio_bounds[1]:
        raise ValueError(f"Invalid simulation-to-wall clock ratio: {ratio}")
    if clock_audit is not None:
        audit_ratio = float(clock_audit["simulation_to_wall_ratio"])
        rmse = float(clock_audit["state_logger_vs_named_sim_joints_rmse_rad"])
        if not ratio_bounds[0] <= audit_ratio <= ratio_bounds[1] or not 0 <= rmse <= joint_rmse_tolerance:
            raise ValueError("Invalid clock/named-joint audit")
    return dict(playback_time=physical - physical[0], simulation_time=physical,
                simulation_to_wall_ratio=ratio, startup_band_released=True,
                maximum_external_wrench=float(force[low:high].max()))


class StateRecorder:
    """Capture a caller-owned simulator, without stepping, resetting or rendering it.

    Call after forward kinematics at each chosen sample. The caller owns sampling
    cadence and controller startup; measured times, not nominal FPS, are saved.
    """

    def __init__(self, model, joint_names, *, body_names=()):
        self.joint_names, self.body_names = list(joint_names), list(body_names)
        self.qpos_ids, self.qvel_ids = named_joint_addresses(model, self.joint_names)
        if len(set(self.body_names)) != len(self.body_names):
            raise ValueError("Duplicate body names")
        try:
            self.body_ids = np.asarray([model.body(name).id for name in body_names], dtype=int)
        except KeyError as exc:
            raise ValueError(f"Unknown body: {exc}") from exc
        self.body_root_ids = (model.body_rootid[self.body_ids].copy()
                              if len(self.body_ids) else np.empty(0, dtype=int))
        self.samples = []

    def append(self, data, *, monotonic_time, elastic_band_enabled, fall_detected=False):
        if not np.isfinite([data.time, monotonic_time]).all():
            raise ValueError("Recording clocks must be finite")
        if self.samples:
            require_forward_time(self.samples[-1]["time"], data.time)
            require_forward_time(self.samples[-1]["monotonic_time"], monotonic_time)
        if not all(np.isfinite(value).all() for value in (data.qpos, data.qvel, data.xfrc_applied)):
            raise ValueError("Nonfinite simulator state")
        row = dict(time=float(data.time), monotonic_time=float(monotonic_time),
                   qpos=data.qpos.copy(), qvel=data.qvel.copy(),
                   joint_pos=data.qpos[self.qpos_ids].copy(), joint_vel=data.qvel[self.qvel_ids].copy(),
                   elastic_band_enabled=bool(elastic_band_enabled),
                   external_wrench_max=float(np.abs(data.xfrc_applied).max(initial=0)),
                   fall_detected=bool(fall_detected))
        if len(self.body_ids):
            # MuJoCo cvel is referenced to each body's subtree COM, not its origin.
            angular = data.cvel[self.body_ids, :3].copy()
            offset = data.xpos[self.body_ids] - data.subtree_com[self.body_root_ids]
            row.update(body_pos=data.xpos[self.body_ids].copy(), body_quat=data.xquat[self.body_ids].copy(),
                       body_ang_vel=angular,
                       body_lin_vel=data.cvel[self.body_ids, 3:] + np.cross(angular, offset))
        self.samples.append(row)

    def save(self, path):
        if len(self.samples) < 2:
            raise ValueError("A recording needs at least two samples")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{k: np.asarray([row[k] for row in self.samples]) for k in self.samples[0]},
                            joint_names=np.asarray(self.joint_names, dtype=str),
                            body_names=np.asarray(self.body_names, dtype=str))
