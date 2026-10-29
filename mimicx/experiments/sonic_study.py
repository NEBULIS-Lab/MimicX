"""Portable, event-clock SONIC evaluation. Optional physics imports are lazy.

Historical corrected sources are optional provenance inputs, never launchers.
The public vendored release/continuous-recording helpers are copied per run.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import pty
import re
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

SCHEMA = "mimicx.sonic-study.v1"
BODY_INDEX = dict(zip((
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
), (0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29)))
JOINT_NAMES = tuple(
    [f"{s}_{j}_joint" for s in ("left", "right")
     for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    + [f"{s}_{j}_joint" for s in ("left", "right")
       for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                 "wrist_roll", "wrist_pitch", "wrist_yaw")])
FIELDS = {"joint_pos": "joint_pos", "joint_vel": "joint_vel", "body_pos": "body_pos_w",
          "body_quat": "body_quat_w", "body_lin_vel": "body_lin_vel_w",
          "body_ang_vel": "body_ang_vel_w"}
PATH_KEYS = ("sonic_root", "policy_parameters", "canonical_model",
             "simulator_config", "simulator_model", "deploy_binary", "decoder", "encoder",
             "observation_config", "sim_python")
HELPERS = ("run_sonic_sim_record.py", "direct_policy_comparison.py", "sonic_release.py",
           "sonic_recording.py")


class InputGateError(ValueError):
    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = audit


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def protocol_sha256(protocol):
    return hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def bind_protocol(manifest, runtime):
    """Bind scientific semantics to a declared object/hash, including older manifests."""
    p = manifest.get("protocol")
    if p is None:
        if not runtime.get("protocol_file"):
            raise ValueError("manifest has no protocol; supply an explicit frozen protocol_file")
        p = read(runtime["protocol_file"])
    digest = protocol_sha256(p)
    if runtime.get("protocol_sha256") != digest:
        raise ValueError("runtime protocol_sha256 does not match the frozen protocol")
    if "protocol" in manifest:
        if manifest.get("protocol_sha256") != digest:
            raise ValueError("manifest protocol hash mismatch")
        for t in manifest["tasks"]:
            if t.get("protocol_id") != p["protocol_id"] or t.get("protocol_sha256") != digest:
                raise ValueError("task protocol identity/hash mismatch")
    if runtime.get("protocol_file") and read(runtime["protocol_file"]) != p:
        raise ValueError("sidecar and manifest protocol differ")
    for key, value in runtime["protocol"].items():
        if key not in p or value != p[key]:
            raise ValueError(f"runtime protocol differs from frozen contract: {key}")
    expected = {"fps": 50, "root_body": "pelvis", "comparison": "strict_greater_than",
                "combination": "OR_union_before_persistence", "grace_frames": 0,
                "recovery_after_failure": False, "duration_ratio_bounds": [.97, 1.03],
                "startup_failure_frame": 0, "native_mask": "prefix_only; no postreset reentry"}
    if any(key in p and p[key] != value for key, value in expected.items()):
        raise ValueError("unsupported common-protocol semantics")
    if not isinstance(p.get("protocol_id"), str) or not p["protocol_id"]:
        raise ValueError("frozen protocol_id required")
    validate_protocol(p)
    runtime["protocol"] = copy.deepcopy(p)
    return digest


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def names(value):
    result = np.asarray(value).tolist()
    if (not isinstance(result, list) or not result or any(not isinstance(n, str) or not n for n in result)
            or len(set(result)) != len(result)):
        raise ValueError("names must be a unique, nonempty string vector")
    return result


def permutation(source, target):
    source, target = names(source), names(target)
    if set(source) != set(target):
        raise ValueError("named joint bijection mismatch")
    return np.asarray([source.index(n) for n in target])


def clocks(values, label):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all()
            or np.any(np.diff(values) <= 0)):
        raise ValueError(f"{label}: finite strictly increasing timestamps required")
    return values


def runtime_config(path):
    path = Path(path).resolve()
    config = read(path)
    for key in PATH_KEYS:
        p = Path(config[key]).expanduser()
        config[key] = str((path.parent / p).absolute())
        if not Path(config[key]).exists():
            raise ValueError(f"missing {key}: {config[key]}")
    for key in ("sonic_root",):
        if not Path(config[key]).is_dir():
            raise ValueError(f"{key} must be a directory")
    if config.get("protocol_file"):
        config["protocol_file"] = str((path.parent / config["protocol_file"]).absolute())
        if not Path(config["protocol_file"]).is_file():
            raise ValueError("missing frozen protocol_file")
    if config.get("corrected_source"):
        config["corrected_source"] = str((path.parent / config["corrected_source"]).absolute())
        if not Path(config["corrected_source"]).is_dir():
            raise ValueError("optional corrected_source must be a directory")
    config.setdefault("reference_mode", "reference_to_deployment_fk")
    if config["reference_mode"] != "reference_to_deployment_fk":
        raise ValueError("only the explicit reference_to_deployment_fk asset bridge is supported")
    config.setdefault("environment", {})
    if (not isinstance(config["environment"], dict)
            or any(not isinstance(k, str) or not isinstance(v, str) for k, v in config["environment"].items())):
        raise ValueError("environment must map strings to strings")
    if any(k in config["environment"] for k in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH")):
        raise ValueError("device visibility belongs to the caller; PYTHONPATH is constructed explicitly")
    config.setdefault("interface", "lo")
    config.setdefault("dds_domain", 0)
    if config["dds_domain"] != 0:
        raise ValueError("released deploy binary uses DDS domain 0; isolate via network namespace/interface")
    if config.get("dds_isolation") not in ("dedicated_network_namespace", "parent_serialized_domain_zero"):
        raise ValueError("declare dds_isolation; a domain environment variable does not isolate this binary")
    config.setdefault("sim_frequency", 200)
    if config["sim_frequency"] < 50 or config["sim_frequency"] % 50:
        raise ValueError("sim_frequency must be an integer multiple of 50")
    config.setdefault("startup_timeout_s", 330.)
    config.setdefault("startup_mode", "reference_phase_zero")
    if config["startup_mode"] not in ("reference_phase_zero", "legacy_paused_reference"):
        raise ValueError("unknown startup_mode")
    phase_zero = config["startup_mode"] == "reference_phase_zero"
    config.setdefault("stand_settle_s", 0. if phase_zero else 5.)
    config.setdefault("release_settle_s", 0. if phase_zero else 3.)
    for key in ("startup_timeout_s", "stand_settle_s", "release_settle_s"):
        if not np.isfinite(config[key]) or config[key] < 0 or (key == "startup_timeout_s" and config[key] == 0):
            raise ValueError(f"invalid {key}")
    if phase_zero and (config["stand_settle_s"] or config["release_settle_s"]):
        raise ValueError("reference_phase_zero forbids unsupported pre-playback settling holds")
    validate_protocol(config["protocol"])
    return config


def validate_protocol(p):
    required = ("root_height_m", "relative_ee_height_m", "root_orientation_rad",
                "persistence_frames", "clock_tolerance_s", "state_gap_s")
    if any(not np.isfinite(p[k]) or p[k] <= 0 for k in required):
        raise ValueError("positive finite protocol thresholds required")
    if int(p["persistence_frames"]) != p["persistence_frames"]:
        raise ValueError("persistence_frames must be an integer")
    critical = names(p["critical_bodies"])
    if not set(critical).issubset(BODY_INDEX) or "pelvis" in critical:
        raise ValueError("unknown critical end effector")
    if not np.isfinite(p["common_prefix_seconds"]) or not 0 < p["common_prefix_seconds"]:
        raise ValueError("a prospective common-prefix duration is required")


def environment(config, *, cpu=False):
    env = dict(os.environ)
    env.update(config["environment"])
    env.update(PYTHONPATH=config["sonic_root"], PYTHONUNBUFFERED="1")
    if cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def preflight(config):
    """Inspect the dynamic loader and CPU packages, never execute the policy."""
    result = {"status": "cpu_preflight_passed_runtime_unverified", "gpu_started": False,
              "host_libc": list(platform.libc_ver()), "paths": {k: config[k] for k in PATH_KEYS}}
    checks = {}
    commands = {
        "loader": ["ldd", config["deploy_binary"]],
        "python": [config["sim_python"], "-c",
                   "import importlib.util,json,mujoco,numpy,yaml; "
                   "assert importlib.util.find_spec('unitree_sdk2py'); "
                   "print(json.dumps(dict(mujoco=mujoco.__version__,numpy=numpy.__version__)))"],
    }
    for name, command in commands.items():
        try:
            proc = subprocess.run(command, env=environment(config, cpu=True), capture_output=True,
                                  text=True, timeout=45, cwd=config["sonic_root"])
            checks[name] = {"command": command, "exit_code": proc.returncode,
                            "output": proc.stdout + proc.stderr}
            if proc.returncode or "not found" in checks[name]["output"]:
                result["status"] = "blocked"
        except (OSError, subprocess.TimeoutExpired) as exc:
            checks[name] = {"command": command, "error": str(exc)}
            result["status"] = "blocked"
    result["checks"] = checks
    loader = checks["loader"].get("output", "")
    result["glibc_blocked"] = bool(re.search(r"GLIBC[^\n]*not found", loader))
    return result


def asset_files(xml):
    """Hash XML includes and file-backed assets, not just the scene's top XML."""
    result = set()

    def visit(path, meshdir="", texturedir=""):
        path = Path(path).resolve()
        if path in result:
            return
        result.add(path)
        root = ET.parse(path).getroot()
        compiler = root.find("compiler")
        if compiler is not None:
            meshdir = compiler.get("meshdir", compiler.get("assetdir", meshdir))
            texturedir = compiler.get("texturedir", compiler.get("assetdir", texturedir))
        for node in root.iter():
            filename = node.get("file")
            if not filename:
                continue
            if node.tag == "include":
                visit(path.parent / filename, meshdir, texturedir)
            else:
                directory = meshdir if node.tag == "mesh" else texturedir if node.tag == "texture" else ""
                asset = (path.parent / directory / filename).resolve()
                if not asset.is_file():
                    raise ValueError(f"missing model asset: {asset}")
                result.add(asset)
    visit(xml)
    return result


def parameter_mapping(path):
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", Path(path).read_text(), flags=re.S)
    maps = []
    for name in ("mujoco_to_isaaclab", "isaaclab_to_mujoco"):
        match = re.search(r"\b" + name + r"\s*=\s*\{([^}]+)\}", text)
        if not match:
            raise ValueError(f"missing deployment mapping {name}")
        values = [int(x.strip()) for x in match[1].split(",") if x.strip()]
        if sorted(values) != list(range(29)):
            raise ValueError(f"{name} is not a 29-joint permutation")
        maps.append(np.asarray(values))
    if not np.array_equal(maps[0][maps[1]], np.arange(29)):
        raise ValueError("deployment mappings are not inverses")
    return maps[0]


def model_addresses(model, joint_names):
    ids = [model.joint(n).id for n in joint_names]
    if any(int(model.jnt_type[i]) not in (2, 3) for i in ids):
        raise ValueError("named mapping requires scalar joints")
    return model.jnt_qposadr[ids], model.jnt_dofadr[ids]


def physical_body_trajectory(model, qpos, qvel, body_names):
    """Postprocess measured physical states without advancing their simulation."""
    import mujoco
    bids = np.asarray([model.body(b).id for b in body_names])
    root_ids, data = model.body_rootid[bids], mujoco.MjData(model)
    rows = []
    for position, velocity in zip(qpos, qvel):
        data.qpos[:], data.qvel[:] = position, velocity
        mujoco.mj_forward(model, data)
        angular = data.cvel[bids, :3].copy()
        linear = data.cvel[bids, 3:] + np.cross(angular, data.xpos[bids] - data.subtree_com[root_ids])
        rows.append((data.xpos[bids].copy(), data.xquat[bids].copy(), linear, angular))
    return {key: np.asarray([r[i] for r in rows])
            for i, key in enumerate(("body_pos", "body_quat", "body_lin_vel", "body_ang_vel"))}


def free_model(model):
    if (list(model.jnt_type).count(0) != 1 or int(model.jnt_type[0]) != 0
            or model.neq or model.nmocap):
        raise ValueError("evaluation requires one unconstrained free root, no equality/mocap support")


def quaternion_error(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    an, bn = np.linalg.norm(a, axis=-1, keepdims=True), np.linalg.norm(b, axis=-1, keepdims=True)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(an == 0) or np.any(bn == 0):
        raise ValueError("invalid quaternion for geodesic error")
    a, b = a / an, b / bn
    return 2 * np.arccos(np.clip(np.abs(np.sum(a * b, axis=-1)), 0, 1))


def deployment_fk(model, target):
    import mujoco
    free_model(model)
    qids, _ = model_addresses(model, names(target["joint_names"]))
    bids = [model.body(n).id for n in names(target["body_names"])]
    root = names(target["body_names"]).index("pelvis")
    d = mujoco.MjData(model)
    indices = np.unique(np.linspace(0, len(target["joint_pos"]) - 1, min(9, len(target["joint_pos"])), dtype=int))
    body_maximum, angle = np.zeros(len(bids)), 0.
    for frame in indices:
        d.qpos[:7] = np.r_[target["body_pos"][frame, root], target["body_quat"][frame, root]]
        d.qpos[qids] = target["joint_pos"][frame]
        mujoco.mj_forward(model, d)
        body_maximum = np.maximum(body_maximum, np.linalg.norm(d.xpos[bids] - target["body_pos"][frame], axis=-1))
        angle = max(angle, float(quaternion_error(d.xquat[bids], target["body_quat"][frame]).max()))
    maximum = float(body_maximum.max())
    audit = dict(deployment_fk_position_m=maximum, deployment_fk_rotation_rad=angle,
                 deployment_fk_by_body_m=dict(zip(names(target["body_names"]), body_maximum.tolist())),
                 deployment_fk_frames=indices.tolist(), mujoco_version=mujoco.__version__)
    if maximum > .002 or angle > .002:
        raise InputGateError(f"deployment FK gate failed: {maximum} m, {angle} rad", audit)
    return audit


def kinematic_trajectory(model, trajectory, joint_names, body_names):
    """Exact named-asset FK from immutable/measured root, 29 joints and velocities."""
    import mujoco
    root = body_names.index("pelvis")
    qids, vids = model_addresses(model, joint_names)
    bids = np.asarray([model.body(b).id for b in body_names])
    d = mujoco.MjData(model)
    rows = []
    for i, joints in enumerate(trajectory["joint_pos"]):
        quat = np.asarray(trajectory["body_quat"][i, root], dtype=float)
        quat = quat / np.linalg.norm(quat)
        d.qpos[:7] = np.r_[trajectory["body_pos"][i, root], quat]
        d.qpos[qids] = joints
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quat)
        d.qvel[:3] = trajectory["body_lin_vel"][i, root]
        d.qvel[3:6] = rotation.reshape(3, 3).T @ trajectory["body_ang_vel"][i, root]
        d.qvel[vids] = trajectory["joint_vel"][i]
        mujoco.mj_forward(model, d)
        angular = d.cvel[bids, :3].copy()
        velocity = d.cvel[bids, 3:] + np.cross(angular, d.xpos[bids] - d.subtree_com[model.body_rootid[bids]])
        rows.append((d.xpos[bids].copy(), d.xquat[bids].copy(), velocity, angular))
    result = {k: np.asarray(trajectory[k]).copy() for k in ("joint_pos", "joint_vel")}
    for i, key in enumerate(("body_pos", "body_quat", "body_lin_vel", "body_ang_vel")):
        result[key] = np.asarray([row[i] for row in rows])
    result.update(joint_names=np.asarray(joint_names), body_names=np.asarray(body_names))
    return result


def reference_to_deployment_fk(arrays, config):
    import mujoco
    target = {k: arrays[v] for k, v in FIELDS.items()}
    model = mujoco.MjModel.from_xml_path(config["simulator_model"])
    free_model(model)
    return kinematic_trajectory(model, target, names(arrays["joint_names"]), names(arrays["body_names"]))


def validate_reference(motion, config):
    from mimicx.datasets.motion_study import CanonicalG1, read_motion
    a = read_motion(Path(motion))
    robot = CanonicalG1(Path(config["canonical_model"]))
    for key, canonical in (("joint_names", robot.joint_names), ("body_names", robot.body_names)):
        if key not in a:
            raise ValueError(f"audited reference must supply {key}")
        order = permutation(a[key], canonical)
        for field in (FIELDS.values()):
            if field.startswith("joint") == key.startswith("joint"):
                a[field] = a[field][:, order]
        a[key] = np.asarray(canonical)
    if abs(float(a["fps"].ravel()[0]) - 50) > 1e-8:
        raise ValueError("reference must be 50 Hz; no retiming is permitted")
    audit = robot.audit(a)
    if not audit["fk_passed"] or not audit["joint_limits_passed"]:
        raise InputGateError(f"canonical FK/limits gate failed: {audit}", audit)
    mj = robot.mj
    deployed = mj.MjModel.from_xml_path(config["simulator_model"])
    control = reference_to_deployment_fk(a, config)
    audit.update(deployment_fk(deployed, control))
    audit.update(reference_mode="reference_to_deployment_fk", motion_repaired=False, retimed=False,
                 joint_values_modified=False, root_poses_modified=False,
                 cross_asset_body_origin_difference_by_name_m=dict(zip(robot.body_names,
                     np.linalg.norm(control["body_pos"] - a["body_pos_w"], axis=-1).max(axis=0).tolist())),
                 cross_asset_origin_difference_is_mapping_failure=False)
    return a, audit


def perturbation(seed, spec):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an explicit uint32")
    if spec.get("mode") == "nominal" and set(spec) == {"mode"}:
        joint, velocity = 0., 0.
    elif spec.get("mode") == "release_state_uniform":
        if set(spec) != {"mode", "joint_position_rad", "root_velocity_m_s"}:
            raise ValueError("explicit perturbation amplitudes required")
        joint, velocity = spec["joint_position_rad"], spec["root_velocity_m_s"]
        if not np.isfinite([joint, velocity]).all() or min(joint, velocity) < 0:
            raise ValueError("invalid perturbation amplitudes")
    else:
        raise ValueError("unsupported perturbation; seed alone is not an execution perturbation")
    rng = np.random.default_rng(seed)
    return {"seed": seed, "distribution": spec, "event": "startup_release_before_ack",
            "random_generator": "numpy.random.Generator(PCG64); joint vector then root-linear-velocity vector",
            "joint_names": list(JOINT_NAMES),
            "joint_position_delta_rad": rng.uniform(-joint, joint, 29).tolist(),
            "root_velocity_delta_m_s": rng.uniform(-velocity, velocity, 3).tolist(),
            "reference_modified": False, "forces_applied": False}


def initialize_phase_zero(model, data, target, perturb):
    """One episode initialization before CONTROL, never a scored-state correction."""
    import mujoco
    qids, vids = model_addresses(model, JOINT_NAMES)
    order = permutation(target["joint_names"], JOINT_NAMES)
    root = names(target["body_names"]).index("pelvis")
    quat = np.asarray(target["body_quat"][0, root], dtype=float)
    quat = quat / np.linalg.norm(quat)
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quat)
    before = dict(root_pose=data.qpos[:7].tolist(), simulation_time=float(data.time))
    data.qpos[:7] = np.r_[target["body_pos"][0, root], quat]
    data.qpos[qids] = target["joint_pos"][0, order] + perturb["joint_position_delta_rad"]
    data.qvel[:] = 0
    data.qvel[:3] = target["body_lin_vel"][0, root] + perturb["root_velocity_delta_m_s"]
    data.qvel[3:6] = rotation.reshape(3, 3).T @ target["body_ang_vel"][0, root]
    data.qvel[vids] = target["joint_vel"][0, order]
    mujoco.mj_forward(model, data)
    return dict(mode="reference_phase_zero_once", count=1, before_control=True, reference_frame=0,
                before=before, simulation_time=float(data.time), monotonic_time=time.monotonic(),
                root_pose_wxyz=data.qpos[:7].tolist(), root_link_linear_velocity_world=data.qvel[:3].tolist(),
                root_link_angular_velocity_world=target["body_ang_vel"][0, root].tolist(),
                free_joint_angular_velocity_local=data.qvel[3:6].tolist(), joint_names=list(JOINT_NAMES),
                joint_positions=data.qpos[qids].tolist(), joint_velocities=data.qvel[vids].tolist(),
                perturbation=perturb, simulation_clock_changed=False, policy_or_physics_modified=False)


def selected_tasks(manifest, task_ids=None):
    manifest = Path(manifest).resolve()
    data = read(manifest)
    if data.get("schema") != "mimicx.motion-study.v1" or not data.get("tasks"):
        raise ValueError("audited motion-study manifest required")
    ids = [t["task_id"] for t in data["tasks"]]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"[a-z0-9_]+", x) for x in ids):
        raise ValueError("duplicate/unsafe task identities")
    chosen = set(ids if task_ids is None else task_ids)
    if not chosen or not chosen.issubset(ids):
        raise ValueError("unknown or empty task selection")
    base = manifest.parent
    tasks = []
    for task in data["tasks"]:
        if task["task_id"] not in chosen:
            continue
        if task.get("quality_status") != "passed" or task.get("quality", {}).get("status") != "passed":
            raise ValueError(f"ineligible audited task: {task['task_id']}")
        p = (base / task["motion_path"]).resolve()
        if not p.is_relative_to(base) or sha(p) != task["sha256"]:
            raise ValueError(f"motion integrity/path failure: {task['task_id']}")
        tasks.append(dict(task, resolved_motion=str(p)))
    return tasks


def prepare(manifest, runtime_path, output, *, seed, perturb, task_ids=None):
    config = runtime_config(runtime_path)
    manifest_digest = sha(manifest)
    perturb = perturbation(seed, perturb)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("prepare requires an empty destination; existing runs are never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    dump(output / "preflight.json", preflight(config))
    preparation = {"status": "auditing", "launched": False, "tasks": [],
                   "cohort": [{k: t.get(k) for k in ("task_id", "split", "quality_status", "quality")}
                              for t in read(manifest)["tasks"]]}
    try:
        protocol_digest = bind_protocol(read(manifest), config)
        tasks = selected_tasks(manifest, task_ids)
    except (ValueError, OSError) as exc:
        preparation.update(status="blocked", error=str(exc))
        dump(output / "preparation.json", preparation)
        raise
    # Audit the entire selected cohort before producing executable run records.
    audited = []
    for task in tasks:
        try:
            arrays, audit = validate_reference(task["resolved_motion"], config)
            audited.append((task, arrays, audit))
            preparation["tasks"].append({"task_id": task["task_id"], "status": "passed", "audit": audit})
        except (ValueError, OSError, KeyError) as exc:
            preparation["tasks"].append({"task_id": task["task_id"], "status": "blocked", "error": str(exc),
                                          "audit": getattr(exc, "audit", None)})
    if len(audited) != len(tasks):
        preparation["status"] = "blocked"
        dump(output / "preparation.json", preparation)
        raise ValueError("selected input/FK gate blocked; inspect preparation.json")
    mapping = parameter_mapping(config["policy_parameters"])
    canonical = names(audited[0][1]["joint_names"])
    hardware_map = permutation(canonical, JOINT_NAMES)
    files = {Path(config[k]) for k in PATH_KEYS if Path(config[k]).is_file()}
    if config.get("protocol_file"):
        files.add(Path(config["protocol_file"]))
    files |= asset_files(config["canonical_model"]) | asset_files(config["simulator_model"])
    sonic_root = Path(config["sonic_root"])
    files.update((sonic_root / "gear_sonic").rglob("*.py"))
    # Bind deployment source provenance as well as the actual executable bytes.
    # Build products and model caches are not runtime source inputs.
    deployment = sonic_root / "gear_sonic_deploy"
    for subtree in ("src", "include"):
        for source in (deployment / subtree).rglob("*"):
            if source.is_file() and (source.suffix in {".c", ".cc", ".cpp", ".h", ".hpp", ".cmake"}
                                     or source.name == "CMakeLists.txt"):
                files.add(source)
    if (deployment / "CMakeLists.txt").is_file():
        files.add(deployment / "CMakeLists.txt")
    identities = {str(p): sha(p) for p in sorted(files)}
    runs = []
    for task, arrays, audit in audited:
        folder = output / task["task_id"] / f"seed_{seed}"
        code, ref = folder / "code", folder / "reference" / task["task_id"]
        code.mkdir(parents=True)
        ref.mkdir(parents=True)
        if config.get("corrected_source"):
            for name in HELPERS:
                source = Path(config["corrected_source"]) / name
                if source.is_file():
                    shutil.copyfile(source, code / name)
        shutil.copyfile(Path(__file__).with_name("sonic_helpers.py"), code / "sonic_helpers.py")
        shutil.copyfile(__file__, code / "study.py")
        (code / "entry.py").write_text("from study import main\nraise SystemExit(main())\n")
        weights = folder / "weights"
        weights.mkdir()
        if Path(config["decoder"]).name == Path(config["encoder"]).name:
            raise ValueError("encoder and decoder basenames must differ for per-run engine caches")
        for key in ("decoder", "encoder"):
            destination = weights / Path(config[key]).name
            shutil.copyfile(config[key], destination)
            if sha(destination) != identities[config[key]]:
                raise ValueError(f"{key} changed during copy")
        shutil.copyfile(task["resolved_motion"], folder / "motion.npz")
        if sha(folder / "motion.npz") != task["sha256"]:
            raise ValueError("motion changed during preparation")
        body_names = names(arrays["body_names"])
        control_body_map = [body_names.index(n) for n in BODY_INDEX]
        target = {k: arrays[v][:, hardware_map] if k.startswith("joint") else arrays[v]
                  for k, v in FIELDS.items()}
        np.savez_compressed(folder / "target.npz", **target, joint_names=JOINT_NAMES,
                            body_names=body_names, fps=50.)
        control = reference_to_deployment_fk(arrays, config)
        joint_order = permutation(control["joint_names"], JOINT_NAMES)
        for key in ("joint_pos", "joint_vel"):
            control[key] = control[key][:, joint_order]
        control["joint_names"] = np.asarray(JOINT_NAMES)
        np.savez_compressed(folder / "control_target.npz", **control, fps=50.)
        for name in FIELDS:
            value = control[name]
            if name.startswith("joint"):
                value = value[:, mapping]
            else:
                value = value[:, control_body_map]
            flat = value.reshape(len(value), -1)
            np.savetxt(ref / f"{name}.csv", flat, delimiter=",", fmt="%.12g",
                       header=",".join(f"{name}_{i}" for i in range(flat.shape[1])), comments="")
        (ref / "metadata.txt").write_text(
            f"Metadata for: {task['task_id']}\n==============================\nBody part indexes:\n["
            + " ".join(map(str, BODY_INDEX.values())) + f"]\nTotal timesteps: {len(target['joint_pos'])}\n")
        n = len(target["joint_pos"])
        if (n != task["frames"] or abs(task["fps"] - 50) > 1e-8
                or abs(task["duration"] - (n - 1) / 50) > 1e-6):
            raise ValueError("manifest elapsed timing differs from reference")
        run = {"schema": SCHEMA, "task": task, "runtime": config, "seed": seed,
               "perturbation": perturb, "frames": n, "fps": 50, "duration": (n - 1) / 50,
               "input_audit": audit, "reference_joint_order": "IsaacLab CSV; hardware named target NPZ",
               "protocol_sha256": protocol_digest,
               "scoring_body_names": body_names, "control_body_names": list(BODY_INDEX),
               "asset_frame_bridge": {"mode": "reference_to_deployment_fk", "joint_root_time_preserved": True,
                   "source_motion_sha256": task["sha256"], "protocol_sha256": protocol_digest,
                   "canonical_model_sha256": identities[config["canonical_model"]],
                   "deployment_model_sha256": identities[config["simulator_model"]],
                   "motion_repair": False, "retiming": False, "controller_or_physics_modified": False},
               "source_files": identities, "same_input_physics": {
                   "reference_sha256": task["sha256"], "reference_repaired": False,
                   "sim_frequency": config["sim_frequency"], "controller_hz": 50,
                   "asset_differences": "reference_to_deployment_fk body frame bridge; released scene/gains unchanged",
                   "scoring_representation": "same canonical FK of measured root and 29 joints, all 30 bodies",
                   "simulator_model": config["simulator_model"]},
               "manifest_sha256": manifest_digest, "manifest_tasks": [t["task_id"] for t in read(manifest)["tasks"]]}
        startup = dict(schema="mimicx.sonic-startup.v4", mode=config["startup_mode"],
                       source_motion_sha256=task["sha256"], common_protocol_sha256=protocol_digest,
                       canonical_model_sha256=identities[config["canonical_model"]],
                       deployment_model_sha256=identities[config["simulator_model"]],
                       decoder_sha256=identities[config["decoder"]], encoder_sha256=identities[config["encoder"]],
                       observation_config_sha256=identities[config["observation_config"]],
                       deploy_binary_sha256=identities[config["deploy_binary"]],
                       causal_source_schema="mimicx.sonic-causal-startup.v1",
                       torso_identity="all causally prior exact-payload candidates must be fresh and post-reset",
                       velocity_mapping="physical joint velocity rounded to float32; 2e-8 absolute tolerance",
                       telemetry_storage="run_directory; external staging requires separately frozen provenance",
                       perturbation=perturb, initialization="once before CONTROL; no scored reset or teleport",
                       dds_receipt="mode_machine application echo restored before CONTROL; unchanged motor commands",
                       heading_history="fresh CONTROL logger history; official I reinitializes heading",
                       activation="I] after restored DDS echo; T immediately after first CONTROL heading log",
                       sampling_gap_scope="reset/release through scored endpoint, including interpolation brackets",
                       state_identity="actual used LowState tick/receipt plus exact causal source payload; torso independent",
                       source_max_age_s=.02, source_payload_quantization_tolerance=2e-8,
                       unsupported_paused_reference_hold_seconds=config["release_settle_s"])
        run.update(startup_contract=startup, startup_contract_sha256=protocol_sha256(startup))
        if sha(manifest) != manifest_digest:
            raise ValueError("manifest changed during preparation; prepare again in a new destination")
        run["frozen_files"] = {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob("*")) if p.is_file()}
        dump(folder / "run.json", run)
        runs.append(str((folder / "run.json").relative_to(output)))
    dump(output / "prepared.json", {"schema": SCHEMA, "runs": runs, "selected_count": len(tasks),
                                    "run_sha256": {p: sha(output / p) for p in runs},
                                    "source_manifest_count": len(read(manifest)["tasks"]),
                                    "cohort": [{k: t.get(k) for k in ("task_id", "split", "quality_status", "quality")}
                                               for t in read(manifest)["tasks"]], "launched": False})
    preparation["status"] = "prepared_runtime_unverified"
    dump(output / "preparation.json", preparation)
    return runs


def verify_run(path):
    path = Path(path).resolve()
    run = read(path)
    if run.get("schema") != SCHEMA:
        raise ValueError("unsupported SONIC run schema")
    root = path.parent.parent.parent
    index = read(root / "prepared.json")
    relative = str(path.relative_to(root))
    if (index.get("schema") != SCHEMA or relative not in index["runs"]
            or index["run_sha256"].get(relative) != sha(path)):
        raise ValueError("prepared run contract changed or incomplete preparation")
    for relative, expected in run["frozen_files"].items():
        p = (path.parent / relative).resolve()
        if not p.is_relative_to(path.parent) or sha(p) != expected:
            raise ValueError(f"frozen run artifact changed: {relative}")
    for origin, expected in run["source_files"].items():
        if sha(origin) != expected:
            raise ValueError(f"runtime source changed: {origin}; prepare a new run")
    return run


def commands(path, run):
    path = Path(path).resolve()
    folder, c = path.parent, run["runtime"]
    sim = [c["sim_python"], str(folder / "code/entry.py"), "_simulate", "--run", str(path)]
    deploy = [c["deploy_binary"], c["interface"], str(folder / "weights" / Path(c["decoder"]).name), str(folder / "reference"),
              "--obs-config", c["observation_config"], "--encoder-file", str(folder / "weights" / Path(c["encoder"]).name),
              "--input-type", "keyboard", "--disable-crc-check", "--enable-csv-logs",
              "--logs-dir", str(folder / "state_logs"), "--target-motion-logfile", str(folder / "target_motion.csv")]
    return {"sim": sim, "deploy": deploy, "sim_cwd": c["sonic_root"],
            "deploy_cwd": str(Path(c["sonic_root"]) / "gear_sonic_deploy"),
            "environment": c["environment"], "states_only": True}


@contextmanager
def duplicate_guard(folder):
    with (folder / ".run.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("this exact run is already active") from exc
        yield


def stop(process):
    if process is None or process.poll() is not None:
        return
    for sig, timeout in ((signal.SIGINT, 30), (signal.SIGTERM, 10), (signal.SIGKILL, 10)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue
    process.wait()


def wait_for(predicate, processes, timeout, label, *, poll_interval=.1):
    deadline = time.monotonic() + timeout
    while True:
        if any(p.poll() is not None for p in processes):
            raise RuntimeError(f"subprocess exited during {label}; inspect sim.log/deploy.log")
        if predicate():
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(label)
        time.sleep(poll_interval)


def evaluate(path):
    path = Path(path).resolve()
    run, folder = verify_run(path), path.parent
    with duplicate_guard(folder):
        if any((folder / f).exists() for f in ("attempt.json", "sim_state.npz", "result.json")):
            raise ValueError("existing attempt retained; prepare a new destination for reruns")
        feasible = preflight(run["runtime"])
        dump(folder / "execution_preflight.json", feasible)
        if feasible["status"] == "blocked":
            raise ValueError("CPU runtime preflight blocked; inspect execution_preflight.json")
        cmd = commands(path, run)
        dump(folder / "commands.json", cmd)
        attempt = {"status": "starting", "pid": os.getpid(),
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "tmux": os.environ.get("TMUX"), "tmux_session": os.environ.get("TMUX_SESSION_NAME"),
                   "command": cmd, "output": str(folder),
                   "network_namespace": os.readlink("/proc/self/ns/net"),
                   "dds_isolation": run["runtime"]["dds_isolation"],
                   "started_unix": time.time(), "run_sha256": sha(path)}
        dump(folder / "attempt.json", attempt)
        start = time.monotonic()
        sim = deploy = None
        master, slave = pty.openpty()
        try:
            with (folder / "sim.log").open("w") as slog, (folder / "deploy.log").open("w") as dlog:
                env = environment(run["runtime"])
                env["MIMICX_STARTUP_STATE_LOG"] = str(folder / "controller_startup_source.csv")
                sim = subprocess.Popen(cmd["sim"], cwd=cmd["sim_cwd"], env=env, stdout=slog,
                                       stderr=subprocess.STDOUT, start_new_session=True)
                attempt["sim_pid"] = sim.pid
                dump(folder / "attempt.json", attempt)
                wait_for(lambda: (folder / "sim_ready.json").exists(), [sim], 60, "simulator ready")
                deploy = subprocess.Popen(cmd["deploy"], cwd=cmd["deploy_cwd"], env=env, stdin=slave,
                                          stdout=dlog, stderr=subprocess.STDOUT, start_new_session=True)
                attempt.update(deploy_pid=deploy.pid, status="controller_startup")
                dump(folder / "attempt.json", attempt)
                wait_for(lambda: "Initialized keyboard input interface" in (folder / "deploy.log").read_text(),
                         [sim, deploy], run["runtime"]["startup_timeout_s"], "controller startup")
                phase_zero = run["runtime"]["startup_mode"] == "reference_phase_zero"
                startup_events = dict(keyboard_ready_observed_monotonic=time.monotonic())
                if phase_zero:
                    wait_for(lambda: "Init Done" in (folder / "deploy.log").read_text(),
                             [sim, deploy], 30, "released SDK initial homing complete")
                    startup_events["sdk_init_done_observed_monotonic"] = time.monotonic()
                else:
                    os.write(master, b"]")
                    time.sleep(run["runtime"]["stand_settle_s"])
                startup_events["reference_reset_release_request_monotonic"] = time.monotonic()
                (folder / "release-band").touch(exist_ok=False)
                ack_path = folder / "release-band.ack.json"
                wait_for(ack_path.exists, [sim, deploy], 15, "startup release acknowledgement", poll_interval=.001)
                ack = read(ack_path)
                if (ack["enabled"] or ack["external_wrench_max"] > 1e-8
                        or ack["qfrc_applied_max"] > 1e-8):
                    raise ValueError("startup support not released")
                startup_events["release_ack_observed_monotonic"] = time.monotonic()
                if phase_zero:
                    if (ack["initialization"]["mode"] != "reference_phase_zero_once"
                            or not ack["dds_reset_echo"]["original_header_restored"]
                            or not ack["dds_reset_echo"]["motor_commands_unchanged"]):
                        raise ValueError("missing phase-zero initialization or post-reset DDS receipt")
                else:
                    time.sleep(run["runtime"]["release_settle_s"])
                if phase_zero:
                    control_time = time.monotonic()
                    dump(folder / "control_request.json", {"monotonic_time": control_time, "keys": "I]"})
                    startup_events["control_request_monotonic"] = control_time
                    os.write(master, b"I]")
                    wait_for(lambda: "Reset heading state to" in (folder / "deploy.log").read_text(),
                             [sim, deploy], .25, "first CONTROL heading initialization", poll_interval=.001)
                    startup_events["first_control_heading_observed_monotonic"] = time.monotonic()
                request_time = time.monotonic()
                dump(folder / "play_request.json", {"monotonic_time": request_time})
                if not phase_zero:
                    dump(folder / "control_request.json", {"monotonic_time": request_time, "keys": "T"})
                startup_events["play_request_monotonic"] = request_time
                dump(folder / "startup_events.json", startup_events)
                os.write(master, b"T")
                attempt["status"] = "playback_requested_not_yet_verified"
                dump(folder / "attempt.json", attempt)
                wait_for(lambda: "completed." in (folder / "deploy.log").read_text()
                         and "Playing motion" in (folder / "deploy.log").read_text(),
                         [sim, deploy], run["duration"] + 45, "full playback completion")
                time.sleep(.5)
            stop(deploy)
            stop(sim)
            if sim.returncode != 0:
                raise RuntimeError(f"simulator exit {sim.returncode}; recording may be partial")
            result = score_recording(path)
            attempt["status"] = "evaluated"
            return result
        except BaseException as exc:
            attempt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            stop(deploy)
            stop(sim)
            os.close(master)
            os.close(slave)
            attempt.update(wall_seconds=time.monotonic() - start, finished_unix=time.time(),
                           sim_exit_code=None if sim is None else sim.returncode,
                           deploy_exit_code=None if deploy is None else deploy.returncode,
                           gpu_compute_seconds=None, energy_joules=None)
            dump(folder / "attempt.json", attempt)


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def simulate(path):
    """Adapted states-only corrected native recorder; no graphics or reset path."""
    import mujoco
    import yaml
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig, override_wbc_config
    from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory

    path, run = Path(path).resolve(), read(path)
    folder, c = path.parent, run["runtime"]
    helper = import_file(folder / "code/sonic_helpers.py", "study_helpers")
    recording = helper
    # Repeat FK in the actual simulator's MuJoCo version before opening DDS.
    checked_model = mujoco.MjModel.from_xml_path(c["simulator_model"])
    target = load_npz(folder / "target.npz")
    scoring_bodies = names(target["body_names"])
    runtime_audit = deployment_fk(checked_model, load_npz(folder / "control_target.npz"))
    dump(folder / "runtime_fk_audit.json", runtime_audit)
    config = SimLoopConfig(interface=c["interface"], sim_frequency=c["sim_frequency"],
                           enable_onscreen=False, enable_offscreen=False, enable_image_publish=False)
    wbc = override_wbc_config(yaml.safe_load(Path(c["simulator_config"]).read_text()), config)
    wbc.update(INTERFACE=c["interface"], DOMAIN_ID=0, ROBOT_SCENE=c["simulator_model"],
               SIMULATE_DT=1 / c["sim_frequency"], ENABLE_ONSCREEN=False, ENABLE_OFFSCREEN=False,
               ENV_NAME=config.env_name)
    sim = SimulatorFactory.create_simulator(config=wbc, env_name=config.env_name, onscreen=False,
                                            offscreen=False, enable_image_publish=False)
    env = sim.sim_env
    model, data = env.mj_model, env.mj_data
    free_model(model)
    recording.install_no_reset_fall_monitor(env)
    qids, vids = model_addresses(model, JOINT_NAMES)
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    publications = helper.StartupPublications(sim.unitree_bridge, lambda: dict(
        physical_q=data.qpos[qids].copy(), physical_dq=data.qvel[vids].copy(),
        physical_base_quat=data.qpos[3:7].copy(), physical_torso_quat=data.xquat[torso_id].copy()),
        joint_names=JOINT_NAMES)
    samples = []
    pending_ack, reset_echo = None, None
    requested, acked = folder / "release-band", folder / "release-band.ack.json"
    private_request = folder / "release-internal"
    period, dt = .02, float(env.sim_dt)
    if abs(dt - 1 / c["sim_frequency"]) > 1e-9:
        raise ValueError("simulator physics timestep differs from frozen configuration")
    dump(folder / "sim_ready.json", {"pid": os.getpid(), "sim_dt": dt, "states_only": True,
                                    "reset_on_fall": False, "body_fk_deferred_until_after_capture": True,
                                    "mujoco_version": mujoco.__version__})
    wall_start, previous, next_sample = time.monotonic(), float(data.time), float(data.time) + period
    maximum_time = c["startup_timeout_s"] + run["duration"] + 90
    try:
        while data.time < maximum_time:
            if requested.exists() and not acked.exists() and pending_ack is None:
                private_request.touch(exist_ok=False)
                if not helper.release_if_requested(env, private_request):
                    raise ValueError("corrected startup release did not execute")
                p = run["perturbation"]
                phase_zero = c["startup_mode"] == "reference_phase_zero"
                if phase_zero:
                    initialization = initialize_phase_zero(model, data, target, p)
                    publications.start(initialization)
                else:
                    data.qpos[qids] += np.asarray(p["joint_position_delta_rad"])
                    data.qvel[:3] += np.asarray(p["root_velocity_delta_m_s"])
                    mujoco.mj_forward(model, data)
                    initialization = None
                ack = read(str(private_request) + ".ack.json")
                ack.update(monotonic_time=time.monotonic(), simulation_time=float(data.time),
                           external_wrench_max=float(np.abs(data.xfrc_applied).max(initial=0)),
                           qfrc_applied_max=float(np.abs(data.qfrc_applied).max(initial=0)),
                           perturbation=p, root_constraints=False)
                if initialization is not None:
                    ack["initialization"] = initialization
                    reset_echo = helper.DdsResetEcho(sim.unitree_bridge, monotonic_time=time.monotonic())
                if ack["enabled"] or ack["external_wrench_max"] > 1e-8 or ack["qfrc_applied_max"] > 1e-8:
                    raise ValueError("external forces remain at release")
                pending_ack = ack
                if not phase_zero:
                    dump(acked, ack)
            env.sim_step()
            recording.require_forward_time(previous, float(data.time))
            previous = float(data.time)
            if reset_echo is not None and not acked.exists():
                receipt = reset_echo.observe(monotonic_time=time.monotonic())
                if receipt is not None:
                    pending_ack.update(dds_reset_echo=receipt, monotonic_time=time.monotonic(),
                                       simulation_time=float(data.time))
                    dump(acked, pending_ack)
            if pending_ack is not None and (np.any(data.xfrc_applied) or np.any(data.qfrc_applied)
                                   or bool(getattr(env, "elastic_band", None) and env.elastic_band.enable)):
                raise ValueError("external support or applied force after release")
            if data.time + 1e-9 >= next_sample:
                # Copy only measured generalized state on the real-time control clock.
                row = dict(time=float(data.time), monotonic_time=time.monotonic(), qpos=data.qpos.copy(),
                           qvel=data.qvel.copy(), joint_pos=data.qpos[qids].copy(), joint_vel=data.qvel[vids].copy(),
                           elastic_band_enabled=bool(getattr(env, "elastic_band", None)
                                                                           and env.elastic_band.enable),
                           external_wrench_max=float(np.abs(data.xfrc_applied).max(initial=0)),
                           qfrc_applied_max=float(np.abs(data.qfrc_applied).max(initial=0)),
                           fall_detected=bool(env.fall))
                if any(not np.isfinite(v).all() for v in row.values()):
                    raise ValueError("nonfinite simulated state")
                samples.append(row)
                next_sample += period
            if round(data.time / dt) % max(round(period / dt), 1) == 0:
                env.update_reward()
            sleep = wall_start + data.time - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        sim.close()
        publications.save(folder)
        if samples:
            final_start = time.monotonic()
            arrays = {k: np.asarray([r[k] for r in samples]) for k in samples[0]}
            arrays.update(physical_body_trajectory(model, arrays["qpos"], arrays["qvel"], scoring_bodies))
            np.savez_compressed(folder / "sim_state.npz",
                                **arrays,
                                joint_names=JOINT_NAMES, body_names=scoring_bodies)
            dump(folder / "recording_finalization.json", {
                "body_fk_after_capture": True, "simulation_steps_added": 0,
                "measured_clocks_modified": False, "wall_seconds": time.monotonic() - final_start})


def resample(values, source, query, *, quaternion=False):
    values = np.asarray(values, dtype=float)
    if len(values) != len(source) or not np.isfinite(values).all():
        raise ValueError("invalid recording values")
    if query[0] < source[0] or query[-1] > source[-1]:
        raise ValueError("incomplete fixed-time coverage; extrapolation/endpoint holds forbidden")
    lo = np.clip(np.searchsorted(source, query, side="right") - 1, 0, len(source) - 2)
    alpha = ((query - source[lo]) / (source[lo + 1] - source[lo])).reshape((-1,) + (1,) * (values.ndim - 1))
    a, b = values[lo], values[lo + 1]
    if not quaternion:
        return a + alpha * (b - a)
    if np.max(np.abs(np.linalg.norm(values, axis=-1) - 1)) > 1e-3:
        raise ValueError("nonunit WXYZ recording quaternion")
    b = np.where(np.sum(a * b, axis=-1, keepdims=True) < 0, -b, b)
    theta = np.arccos(np.clip(np.sum(a * b, axis=-1, keepdims=True), -1, 1))
    sine = np.sin(theta)
    safe = np.where(sine < 1e-8, 1, sine)
    q = np.where(sine < 1e-8, a + alpha * (b - a),
                 np.sin((1 - alpha) * theta) / safe * a + np.sin(alpha * theta) / safe * b)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def csv_log(path):
    a = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    if not len(a) or not a.dtype.names or not all(np.isfinite(a[k]).all() for k in a.dtype.names):
        raise ValueError(f"invalid controller CSV: {path}")
    return a


def target_log(path):
    """Read the SDK's 36 numeric columns, allowing its empty trailing CSV field."""
    rows = []
    with Path(path).open(newline="") as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            if len(row) == 37 and not row[-1].strip():
                row = row[:-1]
            if len(row) != 36:
                raise ValueError("target log must have 36 numeric columns and at most one empty trailing field")
            rows.append([float(value) for value in row])
    data = np.asarray(rows, dtype=float)
    if data.ndim != 2 or not len(data) or not np.isfinite(data).all():
        raise ValueError("empty/nonfinite target log")
    return data


def canonical_body_states(states, target, model_xml):
    """Recover omitted body telemetry from actual saved physical state, not targets."""
    wanted, recorded = names(target["body_names"]), names(states["body_names"])
    if set(wanted).issubset(recorded):
        return states, {"reconstructed": False, "scoring_body_count": len(wanted)}
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    n = len(states["time"])
    if (np.shape(states.get("qpos")) != (n, model.nq)
            or np.shape(states.get("qvel")) != (n, model.nv)):
        raise ValueError("all 30 scoring bodies require full saved physical qpos/qvel, not a control subset")
    if not np.isfinite(states["qpos"]).all() or not np.isfinite(states["qvel"]).all():
        raise ValueError("nonfinite physical state for body reconstruction")
    free_model(model)
    qids, vids = model_addresses(model, names(states["joint_names"]))
    if (not np.allclose(states["joint_pos"], states["qpos"][:, qids], atol=1e-7, rtol=0)
            or not np.allclose(states["joint_vel"], states["qvel"][:, vids], atol=1e-7, rtol=0)):
        raise ValueError("saved named joints disagree with physical qpos/qvel")
    result = dict(states, body_names=np.asarray(wanted))
    result.update(physical_body_trajectory(model, states["qpos"], states["qvel"], wanted))
    # An omitted body's physical contact label cannot be recovered from another label.
    result.pop("body_contact", None)
    return result, {"reconstructed": True, "scoring_body_count": len(wanted),
                    "basis": "saved physical qpos/qvel and configured deployment scene FK; no reference fitting",
                    "mujoco_version": mujoco.__version__, "model_sha256": sha(model_xml),
                    "original_recorded_body_names": recorded}


def align_recording(states, target, q, playing, logged_target, ack, play_request, protocol):
    """Event onset + nominal 50 Hz grid, never motion-error onset fitting."""
    sim, wall = clocks(states["time"], "simulation"), clocks(states["monotonic_time"], "wall")
    fall = np.asarray(states["fall_detected"])
    if fall.shape != wall.shape or not np.isin(fall, [0, 1]).all():
        raise ValueError("invalid native fall telemetry")
    if len(sim) != len(wall):
        raise ValueError("simulation/wall sample counts differ")
    if len(q) != len(playing) or len(q) != len(logged_target):
        raise ValueError("controller/target logs are not row synchronized")
    for key in ("index", "time_monotonic_ms"):
        if not np.array_equal(q[key], playing[key]):
            raise ValueError("controller/playback row identity mismatch")
    if not np.isin(playing["playing_0"], [0, 1]).all():
        raise ValueError("invalid playback flags")
    ids = np.flatnonzero(playing["playing_0"] > .5)
    n = len(target["joint_pos"])
    if n < 2:
        raise ValueError("at least two reference frames required")
    if len(ids) != n or not np.array_equal(ids, np.arange(ids[0], ids[0] + n)):
        raise ValueError("playback must be one complete, unpaused reference pass")
    event = clocks(q["time_monotonic_ms"][ids] / 1000, "playback event")
    grid = event[0] + np.arange(n) / 50
    controller_ratio = float((event[-1] - event[0]) / ((n - 1) / 50))
    if not .97 <= controller_ratio <= 1.03 or np.max(np.abs(event - grid)) > protocol["clock_tolerance_s"]:
        raise ValueError("controller playback differs from fixed 50 Hz clock")
    expected = np.concatenate([target["body_pos"][:, 0], target["body_quat"][:, 0], target["joint_pos"]], axis=1)
    if logged_target.shape[1] != 36 or not np.allclose(logged_target[ids], expected, atol=1e-4, rtol=0):
        raise ValueError("played target differs from frozen input; no temporal/heading repair allowed")
    if (not np.isfinite([ack["external_wrench_max"], ack["qfrc_applied_max"], ack["monotonic_time"],
                         ack["simulation_time"], play_request["monotonic_time"]]).all()
            or ack["enabled"] or ack.get("root_constraints", True)
            or ack["external_wrench_max"] > 1e-8 or ack["qfrc_applied_max"] > 1e-8
            or not wall[0] <= ack["monotonic_time"] <= play_request["monotonic_time"] <= event[0]):
        raise ValueError("missing/stale/late release acknowledgement")
    if abs(ack["simulation_time"] - np.interp(ack["monotonic_time"], wall, sim)) > protocol["clock_tolerance_s"]:
        raise ValueError("release acknowledgement disagrees with physical clock")
    if grid[0] < wall[0] or max(grid[-1], event[-1]) > wall[-1]:
        raise ValueError("incomplete fixed-time simulation coverage")
    low = max(int(np.searchsorted(wall, ack["monotonic_time"], side="right")) - 1, 0)
    # The ACK can lie between samples: audit only samples at/after release here.
    low_release = int(np.searchsorted(wall, ack["monotonic_time"], side="left"))
    high = min(int(np.searchsorted(wall, max(grid[-1], event[-1]))) + 1, len(wall))
    clock_start = ack.get("initialization", {}).get("monotonic_time", ack["monotonic_time"])
    if not np.isfinite(clock_start) or not wall[0] <= clock_start <= ack["monotonic_time"]:
        raise ValueError("invalid reset/release clock for sample-gap audit")
    clock_low = max(int(np.searchsorted(wall, clock_start, side="right")) - 1, 0)
    sim_gap, wall_gap = np.diff(sim), np.diff(wall)
    if np.max(sim_gap[clock_low:high-1]) > protocol["state_gap_s"]:
        raise ValueError("missing simulator samples in reset/release/scored interval")
    if np.max(wall_gap[clock_low:high-1]) > protocol["state_gap_s"]:
        raise ValueError("missing wall-clock samples in reset/release/scored interval")
    for key in ("elastic_band_enabled", "external_wrench_max", "qfrc_applied_max"):
        value = np.asarray(states[key])
        if value.shape != wall.shape or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError("invalid support telemetry")
        if key == "elastic_band_enabled" and not np.isin(value, [0, 1]).all():
            raise ValueError("invalid band telemetry")
        if np.any(value[low_release:high] > (0 if key == "elastic_band_enabled" else 1e-8)):
            raise ValueError("external support/forces after startup release")
    bracket = max(int(np.searchsorted(wall, grid[0], side="right")) - 1, low)
    if (states["elastic_band_enabled"][bracket] or states["external_wrench_max"][bracket] > 1e-8
            or states["qfrc_applied_max"][bracket] > 1e-8):
        raise ValueError("supported interpolation bracket at playback onset")
    physical = np.interp(grid, wall, sim)
    ratio = float((physical[-1] - physical[0]) / (grid[-1] - grid[0]))
    if not .97 <= ratio <= 1.03 or np.max(np.abs(physical - physical[0] - np.arange(n) / 50)) > protocol["clock_tolerance_s"]:
        raise ValueError("simulation clock slow/stalled; time warping forbidden")
    source_joint_order = permutation(states["joint_names"], target["joint_names"])
    source_bodies = names(states["body_names"])
    body_order = [source_bodies.index(b) for b in names(target["body_names"])]
    actual = {}
    for key in FIELDS:
        value = states[key][:, source_joint_order if key.startswith("joint") else body_order]
        actual[key] = resample(value, wall, grid, quaternion=key == "body_quat")
    measured = np.column_stack([q[f"q_{i}"][ids] for i in range(29)])
    event_joints = resample(states["joint_pos"][:, source_joint_order], wall, event)
    rmse = float(np.sqrt(np.mean((event_joints - measured)**2)))
    if rmse > .1:
        raise ValueError("controller logger does not match named simulator joints")
    if "body_contact" in states:
        if (np.shape(states["body_contact"]) != (len(wall), len(source_bodies))
                or not np.isin(states["body_contact"], [0, 1]).all()):
            raise ValueError("invalid physical contact telemetry")
        # Nearest physical sample, no interpolation of discrete contact labels.
        idx = np.clip(np.searchsorted(wall, grid), 0, len(wall) - 1)
        actual["body_contact"] = states["body_contact"][idx][:, body_order]
    onset = int(np.searchsorted(wall, event[0]))
    initialized = ack.get("initialization", {}).get("mode") == "reference_phase_zero_once"
    if initialized:
        root = names(target["body_names"]).index("pelvis")
        critical = [names(target["body_names"]).index(b) for b in protocol["critical_bodies"]]
        pos = states["body_pos"][low_release:onset]
        rotation = states["body_quat"][low_release:onset, root]
        relative = (pos[:, critical, 2] - pos[:, root:root+1, 2]
                    - target["body_pos"][0, critical, 2] + target["body_pos"][0, root, 2])
        bad = ((np.abs(pos[:, root, 2] - target["body_pos"][0, root, 2]) > protocol["root_height_m"])
               | (np.abs(relative).max(axis=1) > protocol["relative_ee_height_m"])
               | (quaternion_error(rotation, target["body_quat"][0, root]) > protocol["root_orientation_rad"]))
        length = int(protocol["persistence_frames"])
        startup_failure = bool(len(bad) >= length and np.any(np.convolve(bad.astype(int), np.ones(length), mode="valid") == length))
    else:
        startup_failure = bool(np.any(states["fall_detected"][low_release:onset]))
    return actual, {"alignment": "first recorded playing event plus fixed 50 Hz clock", "fitted_onset": False,
                    "event_start_monotonic": float(event[0]), "simulation_to_wall_ratio": ratio,
                    "controller_to_nominal_duration_ratio": controller_ratio,
                    "state_logger_vs_named_sim_joints_rmse_rad": rmse,
                    "maximum_event_jitter_s": float(np.max(np.abs(event - grid))),
                    "sampling_gap_audit": dict(scope="reset/release through scored endpoint and interpolation brackets",
                        start_monotonic=float(clock_start), bracket_first_monotonic=float(wall[clock_low]),
                        bracket_last_monotonic=float(wall[high-1]),
                        maximum_simulation_gap_s=float(sim_gap[clock_low:high-1].max()),
                        maximum_wall_gap_s=float(wall_gap[clock_low:high-1].max()),
                        cold_start_maximum_wall_gap_s=float(wall_gap[:clock_low].max(initial=0)),
                        full_capture_maximum_wall_gap_s=float(wall_gap.max())),
                    "fixed_frames": n, "endpoint_hold_seconds": 0, "startup_failure": startup_failure}


def causal_publication_audit(evidence, controller, *, reset_monotonic, reset_simulation_time,
                             original_header, max_age_s=.02):
    """Bind actual CONTROL inputs to source publications, never fit a state clock."""
    tolerance = 2e-8  # Float32 wire values printed by the SDK with nine decimals.
    epsilon = 1e-6  # The existing controller log clock is rounded to microseconds.
    required_controller = ("control_read_monotonic", "log_monotonic", "q", "dq", "base_quat", "torso_quat",
        "torso_gyro", "torso_accel", "lowstate_tick", "mode_machine", "lowstate_received_monotonic", "torso_received_monotonic")
    if any(k not in controller for k in required_controller):
        raise ValueError("missing actual used CONTROL source provenance")
    read_time, log_time = controller["control_read_monotonic"], controller["log_monotonic"]
    if (not np.isfinite([read_time, log_time, reset_monotonic, reset_simulation_time, max_age_s]).all()
            or not 0 < max_age_s <= .02 or read_time > log_time + epsilon
            or log_time - read_time > .02 or read_time < reset_monotonic):
        raise ValueError("invalid CONTROL/source audit clocks or freshness bound")
    if (names(evidence.get("joint_names")) != list(JOINT_NAMES)
            or not np.array_equal(evidence.get("motor_indices"), np.arange(29))):
        raise ValueError("publication joint names do not match hardware mapping")
    result = {}
    for topic in ("lowstate", "torso"):
        table = {k[len(topic)+1:]: np.asarray(v) for k, v in evidence.items() if k.startswith(topic + "_")}
        sequence = np.asarray(table.get("sequence", []))
        if sequence.ndim != 1 or not len(sequence) or not np.isfinite(sequence).all() or np.any(np.diff(sequence) <= 0):
            raise ValueError("invalid publication sequence")
        n = len(sequence)
        required = dict(source_time=(), source_monotonic=(), write_start=(), write_end=(), write_success=(), quat=(4,))
        required.update(dict(q=(29,), dq=(29,), tick=(), mode_machine=(), wire_motor_count=(), physical_q=(29,),
                             physical_dq=(29,), physical_base_quat=(4,)) if topic == "lowstate"
                        else dict(gyro=(3,), accel=(3,), physical_torso_quat=(4,)))
        for key, shape in required.items():
            if table.get(key, np.empty(0)).shape != (n, *shape) or not np.isfinite(table[key]).all():
                raise ValueError("incomplete/nonfinite source publication " + topic + "/" + key)
        for key in ("source_time", "source_monotonic"):
            if np.any(np.diff(table[key]) <= 0):
                raise ValueError("non-increasing source publication " + key)
        if topic == "lowstate" and np.any(table["wire_motor_count"] != 35):
            raise ValueError("released LowState requires 35 motor slots with named first29")
        for key in ("quat", "physical_base_quat" if topic == "lowstate" else "physical_torso_quat"):
            if np.max(np.abs(np.linalg.norm(table[key], axis=1) - 1)) > 1e-3:
                raise ValueError("invalid source publication quaternion")
        if (np.any(table["write_start"] < table["source_monotonic"])
                or np.any(table["write_end"] < table["write_start"])
                or not np.isin(table["write_success"], [1]).all()):
            raise ValueError("invalid publication write clocks or failed transport")
        same = np.ones(n, dtype=bool)
        received = controller[topic + "_received_monotonic"]
        if not np.isfinite(received):
            raise ValueError("nonfinite source receipt clock")
        payload = dict(q=controller["q"], dq=controller["dq"], quat=controller["base_quat"]) if topic == "lowstate" else dict(
            quat=controller["torso_quat"], gyro=controller["torso_gyro"], accel=controller["torso_accel"])
        for key, observed in payload.items():
            observed = np.asarray(observed)
            if observed.shape != table[key].shape[1:] or not np.isfinite(observed).all():
                raise ValueError("invalid controller source payload " + key)
            same &= np.max(np.abs(table[key] - observed), axis=1) <= tolerance
        if topic == "lowstate":
            if not np.array_equal(table["tick"], np.floor(table["source_time"] * 1000)):
                raise ValueError("LowState tick disagrees with source physics time")
            same &= table["tick"] == controller["lowstate_tick"]
            same &= table["mode_machine"] == controller["mode_machine"]
        # Search all causally prior history before applying age/reset filters.
        same &= table["write_start"] <= received + epsilon
        indices = np.flatnonzero(same)
        if not len(indices) or (topic == "lowstate" and len(indices) != 1):
            raise ValueError("missing or ambiguous exact causal publication identity: " + topic)
        i = int(indices[0])  # Earliest candidate supplies the conservative worst age.
        source, start = float(table["source_monotonic"][i]), float(table["write_start"][i])
        if (np.any(table["source_monotonic"][indices] < reset_monotonic)
                or np.any(table["source_time"][indices] < reset_simulation_time)
                or received > read_time + epsilon
                or np.any(read_time - table["source_monotonic"][indices] > max_age_s)
                or read_time - received > max_age_s):
            raise ValueError("pre-reset, future or overage source publication: " + topic)
        if topic == "lowstate" and table["mode_machine"][i] != original_header:
            raise ValueError("CONTROL did not consume the restored DDS header")
        if topic == "lowstate":
            q_error = float(np.sqrt(np.mean((table["q"][i] - table["physical_q"][i])**2)))
            physical_wire_dq = table["physical_dq"][i].astype(np.float32).astype(float)
            dq_error = float(np.max(np.abs(table["dq"][i] - physical_wire_dq)))
            if not np.isfinite(physical_wire_dq).all() or dq_error > tolerance:
                raise ValueError("source-message joint velocity mapping guard failed")
            angle = float(quaternion_error(table["quat"][i], table["physical_base_quat"][i]))
            if q_error > .02 or angle > .03:
                raise ValueError("source-message joint/base mapping guard failed")
        else:
            q_error = None
            dq_error = None
            angle = float(quaternion_error(table["quat"][indices], table["physical_torso_quat"][indices]).max())
            if angle > .03:
                raise ValueError("source-message torso mapping guard failed")
        result[topic] = dict(sequence=int(sequence[i]) if len(indices) == 1 else None,
            candidate_sequences=sequence[indices].astype(int).tolist(), candidate_count=len(indices),
            all_candidates_fresh=True, source_time=float(table["source_time"][i]),
            source_monotonic=source, write_start=start, write_end=float(table["write_end"][i]),
            received_monotonic=float(received), source_age_s=float(read_time-source),
            receipt_age_s=float(read_time-received), q_mapping_rmse_rad=q_error, imu_mapping_error_rad=angle,
            dq_mapping_max_rad_s=dq_error,
            identity="actual used tick and exact float32/CSV payload" if topic == "lowstate" else "causal torso payload-equivalence set; worst age over all candidates")
    return dict(schema="mimicx.sonic-causal-startup.v1", max_age_s=max_age_s,
                payload_quantization_tolerance=tolerance, publications=result, rollout_clock_fitted=False)


def verify_startup_evidence(folder, run, states, target, q, playing, ack):
    """Verify reset receipt, first CONTROL state and official heading reset causally."""
    if run["runtime"]["startup_mode"] != "reference_phase_zero":
        return {"mode": "legacy_paused_reference", "comparative_initialization_compatible": False}
    required = ("controller_startup_source.csv", "startup_publications.npz", "startup_publications.json", "state_logs/dq.csv")
    missing = [name for name in required if not (folder / name).is_file()]
    if missing:
        raise ValueError("missing causal startup provenance: " + ", ".join(missing) + "; old captures cannot be silently mixed")
    init, echo = ack["initialization"], ack["dds_reset_echo"]
    events = read(folder / "startup_events.json")
    control = read(folder / "control_request.json")
    times = [events["sdk_init_done_observed_monotonic"],
             events["reference_reset_release_request_monotonic"], init["monotonic_time"],
             echo["marker_received_monotonic"], echo["restored_received_monotonic"],
             ack["monotonic_time"], control["monotonic_time"], float(q["time_monotonic_ms"][0] / 1000)]
    if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("reset/receipt/CONTROL clocks are incomplete or out of order")
    if (init["mode"] != "reference_phase_zero_once" or init["count"] != 1 or not init["before_control"]
            or init["reference_frame"] != 0 or init["simulation_clock_changed"]
            or not echo["original_header_restored"] or not echo["motor_commands_unchanged"]
            or init["perturbation"] != run["perturbation"] or control["keys"] != "I]"):
        raise ValueError("invalid one-time initialization or DDS receipt proof")
    if times[6] - times[5] > run["runtime"]["protocol"]["clock_tolerance_s"] or times[7] - times[2] > .25:
        raise ValueError("unsupported pre-CONTROL hold after phase-zero initialization")
    play_request = read(folder / "play_request.json")["monotonic_time"]
    heading_seen = events["first_control_heading_observed_monotonic"]
    if not (times[-1] <= heading_seen <= play_request <= times[-1] + .03):
        raise ValueError("playback was not requested immediately after first CONTROL heading")
    root = names(target["body_names"]).index("pelvis")
    if (not np.allclose(init["root_pose_wxyz"][:3], target["body_pos"][0, root], atol=1e-7, rtol=0)
            or quaternion_error(np.asarray(init["root_pose_wxyz"])[3:], target["body_quat"][0, root]) > 1e-6
            or not np.allclose(init["joint_positions"], target["joint_pos"][0] + run["perturbation"]["joint_position_delta_rad"], atol=1e-7, rtol=0)
            or not np.allclose(init["joint_velocities"], target["joint_vel"][0], atol=1e-7, rtol=0)
            or not np.allclose(init["root_link_linear_velocity_world"], target["body_lin_vel"][0, root] + run["perturbation"]["root_velocity_delta_m_s"], atol=1e-7, rtol=0)
            or not np.allclose(init["root_link_angular_velocity_world"], target["body_ang_vel"][0, root], atol=1e-7, rtol=0)):
        raise ValueError("initialized state differs from immutable phase zero and declared noise")
    recorded = np.asarray([q[f"q_{i}"][0] for i in range(29)])
    dq = csv_log(folder / "state_logs/dq.csv")
    used = csv_log(folder / "controller_startup_source.csv")
    if (len(used) != 1 or used["index"][0] != q["index"][0]
            or dq["index"][0] != q["index"][0] or dq["time_monotonic_ms"][0] != q["time_monotonic_ms"][0]):
        raise ValueError("used-source and CONTROL logger identities do not agree")
    evidence = load_npz(folder / "startup_publications.npz")
    source_meta = read(folder / "startup_publications.json")
    if (source_meta["reset"] != {k: init[k] for k in ("monotonic_time", "simulation_time")}
            or source_meta["schema"] != "mimicx.sonic-startup-publications.v1"):
        raise ValueError("source capture reset provenance differs from initialization")
    controller = {key: float(used[key][0]) for key in ("control_read_monotonic", "lowstate_received_monotonic",
                  "torso_received_monotonic", "lowstate_tick", "mode_machine")}
    controller.update(q=recorded, dq=np.asarray([dq[f"dq_{i}"][0] for i in range(29)]), log_monotonic=times[-1],
                      torso_gyro=[used[f"torso_gyro_{i}"][0] for i in range(3)],
                      torso_accel=[used[f"torso_accel_{i}"][0] for i in range(3)])
    base = csv_log(folder / "state_logs/base_quat.csv")
    torso = csv_log(folder / "state_logs/torso_quat.csv")
    quaternion_checks = {}
    metadata = {"index", "time_ms", "time_realtime_ms", "time_monotonic_ms", "ros_timestamp"}
    for name, log, body in (("base", base, "pelvis"), ("torso", torso, "torso_link")):
        columns = [c for c in log.dtype.names if c not in metadata]
        if (len(columns) != 4 or log["index"][0] != q["index"][0]
                or log["time_monotonic_ms"][0] != q["time_monotonic_ms"][0]):
            raise ValueError("initial CONTROL IMU and joint log rows do not agree")
        observed = np.asarray([log[c][0] for c in columns])
        controller[name + "_quat"] = observed
        quaternion_checks[name] = dict(controller_wxyz=observed.tolist())
    causal = causal_publication_audit(evidence, controller, reset_monotonic=init["monotonic_time"],
                reset_simulation_time=init["simulation_time"], original_header=echo["original_header"])
    q_rmse = causal["publications"]["lowstate"]["q_mapping_rmse_rad"]
    for name, topic in (("base", "lowstate"), ("torso", "torso")):
        quaternion_checks[name]["error_rad"] = causal["publications"][topic]["imu_mapping_error_rad"]
    text = (folder / "deploy.log").read_text().split("Playing motion", 1)[0]
    headings = re.findall(r"Reset heading state to ([^\n]+)", text)
    if not headings:
        raise ValueError("official heading reinitialization evidence is missing")
    heading = np.asarray([float(v) for v in headings[-1].split(",")])
    if quaternion_error(heading, np.asarray(quaternion_checks["base"]["controller_wxyz"])) > .03:
        raise ValueError("heading reset used stale suspended root orientation")
    first_play = int(np.flatnonzero(playing["playing_0"] > .5)[0])
    return dict(mode="reference_phase_zero_once", startup_contract_sha256=run["startup_contract_sha256"],
                clock_sequence_monotonic=times, controller_first_q_rmse_rad=q_rmse,
                controller_initial_imu=quaternion_checks, official_heading_reset_wxyz=heading.tolist(),
                actual_CONTROL_monotonic=times[-1], actual_playback_monotonic=float(q["time_monotonic_ms"][first_play]/1000),
                actual_first_CONTROL_q=recorded.tolist(), reference_first_q=target["joint_pos"][0].tolist(),
                causal_source_audit=causal,
                comparative_initialization_compatible=True, reference_frame_at_first_playback=0)


def common_metrics(actual, target, *, model_xml, joint_names, body_names, protocol,
                   startup_failure=False, reference_frames=None, native_failure_frame=None,
                   valid_mask=None):
    """Same fixed-clock metric vector for released and task-trained controllers.

    Root-local FK is recomputed with the canonical asset, with an identity root.
    Prefix duration is prospective, not selected by a method's survival time.
    """
    import mujoco
    validate_protocol(protocol)
    joint_names, body_names = names(joint_names), names(body_names)
    n = len(target["joint_pos"])
    if n < 2 or (reference_frames is not None and not np.array_equal(reference_frames, np.arange(n))):
        raise ValueError("common scoring requires complete phase-zero fixed-clock reference frames")
    valid = np.ones(n, dtype=bool) if valid_mask is None else np.asarray(valid_mask)
    if (valid.shape != (n,) or not np.isin(valid, [0, 1]).all()
            or np.any(np.diff(valid.astype(int)) > 0)):
        raise ValueError("native validity must be a prefix; postreset rows cannot reenter scoring")
    valid_count = int(valid.sum())
    if native_failure_frame is not None and (isinstance(native_failure_frame, bool)
            or int(native_failure_frame) != native_failure_frame or not 0 <= native_failure_frame < n):
        raise ValueError("invalid native failure frame")
    widths = {"joint_pos": (len(joint_names),), "joint_vel": (len(joint_names),),
              "body_pos": (len(body_names), 3), "body_quat": (len(body_names), 4),
              "body_lin_vel": (len(body_names), 3), "body_ang_vel": (len(body_names), 3)}
    for data in (actual, target):
        for key, width in widths.items():
            if np.shape(data[key]) != (n, *width) or not np.isfinite(data[key]).all():
                raise ValueError(f"invalid common-metric {key} shape/values")
        if np.max(np.abs(np.linalg.norm(data["body_quat"], axis=-1) - 1)) > 1e-3:
            raise ValueError("invalid common-metric rotations")
    root = body_names.index("pelvis")
    critical = [body_names.index(b) for b in protocol["critical_bodies"]]
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    free_model(model)
    canonical_bodies = [model.body(i).name for i in range(1, model.nbody)]
    if len(canonical_bodies) != 30 or set(body_names) != set(canonical_bodies):
        raise ValueError("common scoring requires all 30 named canonical bodies; control-target subsets are invalid")
    # Different released XMLs may name different body origins: measure both in one asset.
    actual = kinematic_trajectory(model, actual, joint_names, body_names)
    target = kinematic_trajectory(model, target, joint_names, body_names)
    qids, _ = model_addresses(model, joint_names)
    bids = [model.body(b).id for b in body_names]
    d = mujoco.MjData(model)

    def fk(joints):
        result = []
        for frame in joints:
            d.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
            d.qpos[qids] = frame
            mujoco.mj_forward(model, d)
            result.append(d.xpos[bids].copy())
        return np.asarray(result)

    error = actual["body_pos"] - target["body_pos"]
    root_angle = quaternion_error(actual["body_quat"][:, root], target["body_quat"][:, root])
    relative = (actual["body_pos"][:, critical, 2] - actual["body_pos"][:, root:root+1, 2]
                - target["body_pos"][:, critical, 2] + target["body_pos"][:, root:root+1, 2])
    bad = ((np.abs(error[:, root, 2]) > protocol["root_height_m"])
           | (np.abs(relative).max(axis=1) > protocol["relative_ee_height_m"])
           | (root_angle > protocol["root_orientation_rad"]))
    persistence = int(protocol["persistence_frames"])
    observed_bad = bad[:valid_count]
    fail = (np.flatnonzero(np.convolve(observed_bad.astype(int), np.ones(persistence, dtype=int), mode="valid") == persistence)
            if persistence <= valid_count else np.array([], dtype=int))
    first = min(int(fail[0]) if len(fail) else n, valid_count,
                n if native_failure_frame is None else int(native_failure_frame))
    first = 0 if startup_failure else first
    completed = not startup_failure and first == n

    def heading(q):
        q = q / np.linalg.norm(q, axis=-1, keepdims=True)
        w, x, y, z = q.T
        return np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))

    heading_delta = heading(actual["body_quat"][:, root]) - heading(target["body_quat"][:, root])
    steps = {"world_root_error_m": np.linalg.norm(error[:, root], axis=1),
             "heading_error_rad": np.abs(np.arctan2(np.sin(heading_delta), np.cos(heading_delta))),
             "root_orientation_error_rad": root_angle,
             "world_body_error_m": np.linalg.norm(error, axis=-1).mean(axis=1),
             "root_local_fk_body_error_m": np.linalg.norm(fk(actual["joint_pos"]) - fk(target["joint_pos"]), axis=-1).mean(axis=1),
             "joint_rmse_rad": np.sqrt(np.mean((actual["joint_pos"] - target["joint_pos"])**2, axis=1)),
             "joint_velocity_rmse_rad_s": np.sqrt(np.mean((actual["joint_vel"] - target["joint_vel"])**2, axis=1)),
             "critical_ee_world_error_m": np.linalg.norm(error[:, critical], axis=-1).mean(axis=1),
             "body_velocity_error_m_s": np.linalg.norm(actual["body_lin_vel"] - target["body_lin_vel"], axis=-1).mean(axis=1),
             "body_angular_velocity_error_rad_s": np.linalg.norm(actual["body_ang_vel"] - target["body_ang_vel"], axis=-1).mean(axis=1),
             "failure_predicate": bad}
    prefix = min(n, int(np.floor(protocol["common_prefix_seconds"] * 50)) + 1)

    def aggregate(count):
        return {k: {"mean": float(v[:count].mean()), "p95": float(np.quantile(v[:count], .95))}
                for k, v in steps.items() if k != "failure_predicate"} if count else None

    result = {"schema": "mimicx.fixed-clock-common-metrics.v1", "frames": n,
              "duration_seconds": (n - 1) / 50, "complete_motion": completed,
              "first_failure_frame": None if completed else first,
              "normalized_failure_horizon": 1. if completed else first / (n - 1),
              "startup_failure": bool(startup_failure), "full_horizon": aggregate(n) if valid_count == n else None,
              "valid_physical_frames": valid_count, "native_failure_frame": native_failure_frame,
              "common_prefix_frames": prefix, "common_prefix": aggregate(prefix) if prefix <= valid_count else None,
              "pre_failure_frames": first, "pre_failure": aggregate(first),
              "p95_definition": "time quantile of per-frame mean body/EE error; no frame censoring",
              "contact_conditioned_slip_m_s": None,
              "unsupported_metrics": ["contact_conditioned_slip_m_s"],
              "native_termination": "separate fall telemetry; no automatic reset", "_steps": steps}
    result.update(scoring_body_names=body_names, scoring_body_count=30,
                  body_representation="canonical_asset_fk_of_root_pose_and_29_named_joints_for_both_controllers")
    return result


def score_recording(path, *, output=None, reason=None):
    path = Path(path).resolve()
    run, folder = verify_run(path), path.parent
    rescoring = output is not None
    destination = Path(output).resolve() if rescoring else folder
    capture_before = None
    if rescoring:
        if not reason or not reason.strip():
            raise ValueError("explicit rescoring requires a provenance reason")
        if destination.is_relative_to(folder):
            raise ValueError("rescoring output must be outside the immutable capture directory")
        if (folder / "attempt.json").exists() and not read(folder / "attempt.json").get("finished_unix"):
            raise ValueError("cannot rescore an active or unfinished physical attempt")
        capture_before = {str(p.relative_to(folder)): sha(p) for p in folder.rglob("*") if p.is_file()}
        destination.mkdir(parents=True, exist_ok=False)
    elif (folder / "result.json").exists():
        raise ValueError("scoring never overwrites an existing result")
    ready, runtime_fk = read(folder / "sim_ready.json"), read(folder / "runtime_fk_audit.json")
    if (ready["reset_on_fall"] or not ready["states_only"]
            or abs(ready["sim_dt"] - 1 / run["runtime"]["sim_frequency"]) > 1e-9
            or not 0 <= runtime_fk["deployment_fk_position_m"] <= .002
            or not 0 <= runtime_fk["deployment_fk_rotation_rad"] <= .002):
        raise ValueError("runtime physics/FK evidence does not meet the frozen protocol")
    target, states = load_npz(folder / "target.npz"), load_npz(folder / "sim_state.npz")
    states, body_audit = canonical_body_states(states, target, run["runtime"]["simulator_model"])
    q = csv_log(folder / "state_logs/q.csv")
    playing = csv_log(folder / "state_logs/motion_playing.csv")
    logged = target_log(folder / "target_motion.csv")
    ack, request = read(folder / "release-band.ack.json"), read(folder / "play_request.json")
    if ack["perturbation"] != run["perturbation"]:
        raise ValueError("release seed/perturbation differs from frozen run")
    startup_audit = verify_startup_evidence(folder, run, states, target, q, playing, ack)
    import mujoco
    canonical = mujoco.MjModel.from_xml_path(run["runtime"]["canonical_model"])
    canonical_fields = kinematic_trajectory(canonical, states, names(states["joint_names"]), names(states["body_names"]))
    for key in ("body_pos", "body_quat", "body_lin_vel", "body_ang_vel"):
        states[key] = canonical_fields[key]
    actual, audit = align_recording(states, target, q, playing, logged, ack, request, run["runtime"]["protocol"])
    audit["body_telemetry"] = body_audit
    metrics = common_metrics(actual, target, model_xml=run["runtime"]["canonical_model"],
                             joint_names=target["joint_names"], body_names=target["body_names"],
                             protocol=run["runtime"]["protocol"], startup_failure=audit["startup_failure"])
    steps = metrics.pop("_steps")
    startup_audit["first_scored_state"] = {k: actual[k][0].tolist() for k in FIELDS}
    startup_audit["reference_frame"] = 0
    dump(destination / "startup_audit.json", startup_audit)
    np.savez_compressed(destination / "paired_states.npz", **actual,
                        **{f"target_{k}": target[k] for k in FIELDS},
                        **{f"metric_{k}": v for k, v in steps.items()},
                        time=np.arange(run["frames"]) / 50,
                        joint_names=target["joint_names"], body_names=target["body_names"])
    dump(destination / "clock_audit.json", audit)
    metrics.update(task_id=run["task"]["task_id"], seed=run["seed"], method="SONIC",
                   reference_sha256=run["task"]["sha256"], adaptation_updates=0,
                   status="evaluated", physical_success=metrics["complete_motion"],
                   perturbation=run["perturbation"], run_sha256=sha(path))
    metrics.update(scoring_body_names=names(target["body_names"]), scoring_body_count=30,
                   control_body_names=list(BODY_INDEX), startup_contract=run.get("startup_contract"),
                   comparative_initialization_compatible=startup_audit["comparative_initialization_compatible"])
    evidence = ["sim_state.npz", "state_logs/q.csv", "state_logs/motion_playing.csv", "target_motion.csv",
                "release-band.ack.json", "play_request.json",
                "sim_ready.json", "runtime_fk_audit.json"]
    evidence.extend(p for p in ("control_request.json", "startup_events.json", "recording_finalization.json",
                                "controller_startup_source.csv", "startup_publications.npz", "startup_publications.json",
                                "state_logs/dq.csv",
                                "state_logs/base_quat.csv", "state_logs/torso_quat.csv") if (folder / p).exists())
    evidence.extend(str(p.relative_to(folder)) for p in (folder / "weights").glob("*.trt"))
    metrics["evidence_sha256"] = {p: sha(folder / p) for p in evidence}
    outputs = ["paired_states.npz", "clock_audit.json", "startup_audit.json"]
    if rescoring:
        evidence.extend(p for p in ("attempt.json", "deploy.log", "sim.log", "run.json", "code/study.py")
                        if (folder / p).exists())
        source_hashes = {p: sha(folder / p) for p in evidence}
        scorer = destination / "scorer.py"
        shutil.copyfile(__file__, scorer)
        provenance = dict(schema="mimicx.sonic-rescore.v1", reason=reason,
                          source_run=str(path), source_run_sha256=sha(path),
                          source_evidence_sha256=source_hashes, scorer_sha256=sha(scorer),
                          original_frozen_scorer_sha256=sha(folder / "code/study.py"),
                          source_attempt=read(folder / "attempt.json") if (folder / "attempt.json").exists() else None,
                          new_physical_execution=False, raw_states_or_behavior_modified=False,
                          common_protocol_sha256=run["protocol_sha256"],
                          source_capture_sha256=capture_before,
                          source_evidence_unchanged=capture_before == {
                              str(p.relative_to(folder)): sha(p) for p in folder.rglob("*") if p.is_file()})
        if not provenance["source_evidence_unchanged"]:
            raise ValueError("source capture changed during rescoring; no result emitted")
        dump(destination / "rescoring.json", provenance)
        metrics["rescoring"] = provenance
        outputs.extend(["scorer.py", "rescoring.json"])
        metrics["generated_evidence_sha256"] = {p: sha(destination / p) for p in outputs}
    else:
        metrics["evidence_sha256"].update({p: sha(destination / p) for p in outputs})
    dump(destination / "result.json", metrics)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("preflight", help="CPU-only dependency/loader inspection")
    check.add_argument("--runtime", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    prep = sub.add_parser("prepare", help="CPU-only audit and immutable per-task preparation")
    prep.add_argument("--manifest", type=Path, required=True)
    prep.add_argument("--runtime", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--task", action="append")
    prep.add_argument("--seed", type=int, required=True)
    prep.add_argument("--perturbation", type=Path, required=True)
    rescore = sub.add_parser("rescore-recording", help="CPU-only explicit new-scorer audit of an immutable capture")
    rescore.add_argument("--run", type=Path, required=True)
    rescore.add_argument("--output", type=Path, required=True)
    rescore.add_argument("--reason", required=True)
    for verb in ("evaluate", "score-recording", "commands", "_simulate"):
        p = sub.add_parser(verb)
        p.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command in ("evaluate", "score-recording", "commands"):
        run_path = args.run.resolve()
        frozen = run_path.parent / "code/study.py"
        if Path(__file__).resolve() != frozen:
            verify_run(run_path)
            # Orchestration and scoring use the prepared code too, not a later checkout.
            return subprocess.run([sys.executable, str(frozen.with_name("entry.py")), args.command,
                                   "--run", str(run_path)], check=False).returncode
    if args.command == "preflight":
        if args.output.exists():
            parser.error("preflight output already exists")
        result = preflight(runtime_config(args.runtime))
        dump(args.output, result)
        print(json.dumps(result))
        return 2 if result["status"] == "blocked" else 0
    if args.command == "prepare":
        print(json.dumps(prepare(args.manifest, args.runtime, args.output, seed=args.seed,
                                 perturb=read(args.perturbation), task_ids=args.task)))
    elif args.command == "commands":
        print(json.dumps(commands(args.run, verify_run(args.run)), indent=2))
    elif args.command == "evaluate":
        print(json.dumps(evaluate(args.run), indent=2))
    elif args.command == "score-recording":
        print(json.dumps(score_recording(args.run), indent=2))
    elif args.command == "rescore-recording":
        print(json.dumps(score_recording(args.run, output=args.output, reason=args.reason), indent=2))
    else:
        simulate(args.run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
