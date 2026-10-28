"""Prospective, CPU-only motion input audit. No policy results are consumed."""
from __future__ import annotations

import csv
import ast
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

SCHEMA = "mimicx.motion-study.v1"
FIELDS = {"joint_pos": (29,), "joint_vel": (29,), "body_pos_w": (30, 3),
          "body_quat_w": (30, 4), "body_lin_vel_w": (30, 3),
          "body_ang_vel_w": (30, 3)}
GROUND_PROTOCOL_BLOCKER = "ground-contact termination protocol must be frozen across all methods before training"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_shortlist(path: Path, subset: str = "all") -> list[dict]:
    config = json.loads(path.read_text())
    if config.get("schema") != "mimicx.motion-shortlist.v1":
        raise ValueError("unsupported shortlist schema")
    tasks = config["tasks"]
    if len(tasks) != 18 or Counter(t["split"] for t in tasks) != {"development": 6, "confirmation": 12}:
        raise ValueError("study must retain six development and twelve confirmation tasks")
    ids = [t["task_id"] for t in tasks]
    if len(set(ids)) != 18 or any(not re.fullmatch(r"[a-z0-9_]+", x) for x in ids):
        raise ValueError("duplicate or unsafe task_id")
    if len({(t['source'], t['stem']) for t in tasks}) != 18:
        raise ValueError("duplicate source identity")
    if subset not in ("all", "development", "confirmation"):
        raise ValueError("unknown subset")
    return [t for t in tasks if subset == "all" or t["split"] == subset]


def read_motion(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in (*FIELDS, "fps")}
        for name in ("joint_names", "body_names"):
            if name in z:
                arrays[name] = z[name]
    fps = arrays["fps"]
    if fps.size != 1 or not np.isfinite(fps).all() or float(fps.reshape(-1)[0]) <= 0:
        raise ValueError("invalid fps")
    n = len(arrays["joint_pos"])
    if n < 2:
        raise ValueError("fewer than two frames")
    for name, shape in FIELDS.items():
        a = arrays[name]
        if a.shape != (n, *shape) or a.dtype.kind not in "fi" or not np.isfinite(a).all():
            raise ValueError(f"invalid {name}: expected finite {(n, *shape)}")
    if np.max(np.abs(np.linalg.norm(arrays["body_quat_w"], axis=-1) - 1)) > 1e-3:
        raise ValueError("malformed/non-unit WXYZ rotations")
    return arrays


def window_motion(arrays: dict, seconds: list | None) -> tuple[dict, dict]:
    fps = float(arrays["fps"].reshape(-1)[0])
    n = len(arrays["joint_pos"])
    start, stop = (0, n - 1) if seconds is None else tuple(int(round(float(s) * fps)) for s in seconds)
    if seconds is not None and any(abs(i / fps - float(s)) > 1e-7 for i, s in zip((start, stop), seconds)):
        raise ValueError("window boundaries must lie on source clock")
    if not 0 <= start < stop < n:
        raise ValueError("window outside source or empty")
    result = {k: v[start:stop + 1].copy() if k in FIELDS else v.copy() for k, v in arrays.items()}
    return result, {"start_frame": start, "end_frame_inclusive": stop,
                    "start_seconds": start / fps, "end_seconds": stop / fps}


def timing_check(frames: int, fps: float, source_frames: int, source_fps: float) -> dict:
    if min(frames, source_frames) < 2 or not np.isfinite([fps, source_fps]).all() or min(fps, source_fps) <= 0:
        raise ValueError("invalid source timing")
    elapsed = (frames - 1) / fps
    source_elapsed = (source_frames - 1) / source_fps
    # One output interval permits endpoint rounding, not silent speed changes.
    tolerance = 1 / fps + 1e-6
    return {"source_frames": source_frames, "source_fps": source_fps,
            "source_duration": source_elapsed, "converted_duration": elapsed,
            "difference_seconds": elapsed - source_elapsed, "tolerance_seconds": tolerance,
            "passed": abs(elapsed - source_elapsed) <= tolerance}


class CanonicalG1:
    def __init__(self, xml: Path):
        import mujoco
        self.mj = mujoco
        self.model = m = mujoco.MjModel.from_xml_path(str(xml))
        if (m.nq, m.nv, m.nbody, m.njnt) != (36, 35, 31, 30):
            raise ValueError("canonical asset must be floating-base G1 robot29/body30")
        self.joint_names = [m.joint(i).name for i in range(1, 30)]
        self.body_names = [m.body(i).name for i in range(1, 31)]
        if self.body_names[0] != "pelvis" or any(not n for n in self.joint_names + self.body_names):
            raise ValueError("unnamed or incompatible canonical asset")
        self.identity = {"xml_origin": str(xml.resolve()), "xml_sha256": sha256(xml),
                         "mujoco_version": mujoco.__version__, "joint_names": self.joint_names,
                         "body_names": self.body_names}

    def audit(self, a: dict) -> dict:
        m, mj = self.model, self.mj
        for name, expected in (("joint_names", self.joint_names), ("body_names", self.body_names)):
            if name in a and a[name].tolist() != expected:
                raise ValueError(f"{name} does not match canonical order")
        n = len(a["joint_pos"])
        indices = np.unique(np.linspace(0, n - 1, min(9, n), dtype=int))
        d = mj.MjData(m)
        pos_error, rotation_error = 0., 0.
        for i in indices:
            d.qpos[:3] = a["body_pos_w"][i, 0]
            d.qpos[3:7] = a["body_quat_w"][i, 0]
            d.qpos[7:] = a["joint_pos"][i]
            mj.mj_forward(m, d)
            pos_error = max(pos_error, float(np.linalg.norm(d.xpos[1:] - a["body_pos_w"][i], axis=-1).max()))
            q = a["body_quat_w"][i].astype(float)
            q /= np.linalg.norm(q, axis=-1, keepdims=True)
            rotation_error = max(rotation_error, float((2 * np.arccos(np.clip(np.abs(np.sum(d.xquat[1:] * q, axis=-1)), 0, 1))).max()))
        limits = m.jnt_range[1:]
        exceed = np.maximum(limits[:, 0] - a["joint_pos"], a["joint_pos"] - limits[:, 1])
        exceed[:, ~m.jnt_limited[1:].astype(bool)] = 0
        maximum = float(np.maximum(exceed, 0).max())
        ankles = [self.body_names.index(x) for x in ("left_ankle_roll_link", "right_ankle_roll_link")]
        feet = a["body_pos_w"][:, ankles, 2]
        return {"fk_sample_frames": indices.tolist(), "fk_max_position_error_m": pos_error,
                "fk_max_rotation_error_rad": rotation_error,
                "fk_passed": pos_error <= .002 and rotation_error <= .002,
                "joint_limit_max_excess_rad": maximum, "joint_limits_passed": maximum <= 1e-4,
                "pelvis_height_range_m": [float(a["body_pos_w"][:, 0, 2].min()), float(a["body_pos_w"][:, 0, 2].max())],
                "ankle_height_range_m": [float(feet.min()), float(feet.max())],
                "boundary_ankle_heights_m": feet[[0, -1]].tolist(),
                "contact_note": "Kinematic link-height inspection only; not collider contact or dynamic feasibility certification"}


def unique_file(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if len(matches) != 1:
        raise ValueError(f"expected one {name}, found {len(matches)} under {root}")
    return matches[0]


def preserve(source: Path, output: Path, relative: str) -> dict:
    dest = output / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    before = sha256(source)
    if dest.exists():
        if sha256(dest) != before:
            raise ValueError(f"refusing to replace changed artifact: {dest}")
    else:
        shutil.copyfile(source, dest)
    if sha256(dest) != before or sha256(source) != before:
        raise ValueError("source changed during copy")
    return {"path": relative, "origin": str(source.resolve()), "sha256": before}


def rebuild_csv(csv_data: np.ndarray, source_times: np.ndarray, robot: CanonicalG1) -> dict:
    """Interpolate at physical timestamps and recompute world link FK/velocities."""
    from scipy.spatial.transform import Rotation, Slerp
    if (csv_data.ndim != 2 or csv_data.shape[1] != 36 or len(csv_data) < 2
            or not np.isfinite(csv_data).all() or source_times.shape != (len(csv_data),)
            or not np.isfinite(source_times).all() or source_times[0] != 0
            or not np.all(np.diff(source_times) > 0)):
        raise ValueError("invalid CSV or source timestamps")
    if np.max(np.abs(np.linalg.norm(csv_data[:, 3:7], axis=1) - 1)) > 1e-3:
        raise ValueError("source CSV has invalid XYZW quaternion")
    t = np.arange(int(np.floor(source_times[-1] * 50 + 1e-9)) + 1) / 50
    if len(t) < 2:
        raise ValueError("source too short")
    qpos = np.empty((len(t), 36))
    for j in [0, 1, 2, *range(7, 36)]:
        qpos[:, j] = np.interp(t, source_times, csv_data[:, j])
    qpos[:, 3:7] = Slerp(source_times, Rotation.from_quat(csv_data[:, 3:7]))(t).as_quat()[:, [3, 0, 1, 2]]
    a = {key: np.empty((len(t), *shape), dtype=np.float32) for key, shape in FIELDS.items()}
    a['fps'] = np.array([50.])
    a['joint_pos'][:] = qpos[:, 7:]
    m, mj = robot.model, robot.mj
    d = mj.MjData(m)
    jp, jr = np.empty((3, m.nv)), np.empty((3, m.nv))
    for i in range(len(t)):
        lo, hi = max(0, i - 1), min(len(t) - 1, i + 1)
        d.qpos[:] = qpos[i]
        mj.mj_differentiatePos(m, d.qvel, t[hi] - t[lo], qpos[lo], qpos[hi])
        mj.mj_forward(m, d)
        a['joint_vel'][i] = d.qvel[6:]
        a['body_pos_w'][i] = d.xpos[1:]
        a['body_quat_w'][i] = d.xquat[1:]
        for body in range(1, 31):
            mj.mj_jacBody(m, d, jp, jr, body)
            a['body_lin_vel_w'][i, body - 1] = jp @ d.qvel
            a['body_ang_vel_w'][i, body - 1] = jr @ d.qvel
    return a


def prepare(config: Path, output: Path, roots: dict[str, Path], xml: Path,
            regenerate_sources: bool = False) -> dict:
    tasks = load_shortlist(config)
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be empty; use a new version rather than overwrite an audit")
    output.mkdir(parents=True, exist_ok=True)
    robot = CanonicalG1(xml)
    producer_evidence = []
    if regenerate_sources:
        for key in ('gmr_timing_source', 'csv_converter_source', 'amass_producer_source'):
            producer_evidence.append(preserve(roots[key], output, 'producer_evidence/' + key + '.py'))
        timing_code = roots['gmr_timing_source'].read_text()
        if 'target_time = np.linspace(0, num_frames-1, new_num_frames)' not in timing_code:
            raise ValueError("unrecognized AMASS endpoint-resampling producer; refuse inferred clock")
        producer = roots['amass_producer_source'].read_text()
        if 'input_fps=args.retarget_fps' not in producer or 'tgt_fps=retarget_fps' not in producer:
            raise ValueError("unrecognized AMASS nominal-FPS conversion chain")
        named_orders = []
        for node in ast.walk(ast.parse(roots['csv_converter_source'].read_text())):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'joint_names' for t in node.targets):
                try:
                    named_orders.append(ast.literal_eval(node.value))
                except (ValueError, TypeError):
                    pass
        if robot.joint_names not in named_orders:
            raise ValueError("CSV producer joint order differs from canonical robot")
    metadata = {}
    if roots.get("bones_metadata") and roots["bones_metadata"].is_file():
        with roots["bones_metadata"].open(newline="") as f:
            for row in csv.DictReader(f):
                key = row["move_g1_path"]
                if key in metadata:
                    raise ValueError("duplicate BONES source metadata identity")
                metadata[key] = row
    records = []
    for task in tasks:
        rec = dict(task, source_path=None, motion_path=None, frames=None, fps=None,
                   duration=None, sha256=None, quality_status="blocked")
        reasons, provenance = [], []
        try:
            source = unique_file(roots[task["source"]], Path(task["stem"]).name + ".npz")
            evidence = preserve(source, output, f"sources/{task['task_id']}/converted.npz")
            provenance.append(evidence)
            rec["source_path"] = evidence["path"]
            a = read_motion(source)
            n, fps = len(a["joint_pos"]), float(a["fps"].reshape(-1)[0])
            rec["original_frames"], rec["original_fps"] = n, fps
            if fps != 50:
                reasons.append("reference is not 50 Hz; no implicit retiming allowed")
            if task["source"] == "BONES":
                row = metadata.get("g1/csv/" + task["stem"] + ".csv")
                if row is None:
                    reasons.append("missing independent BONES source timing metadata")
                else:
                    rel = f"sources/{task['task_id']}/source_metadata.json"
                    (output / rel).write_text(json.dumps(row, indent=2) + "\n")
                    provenance.append({"path": rel, "origin": str(roots['bones_metadata']), "sha256": sha256(output / rel)})
                    rec["timing"] = timing_check(n, fps, int(row["move_duration_frames"]), 120.)
                    rec["source_group"] = row["take_name"] + "__" + row["take_actor"]
                    rec["provenance_note"] = "Original release metadata at 120 Hz; raw G1 archive not re-extracted; transformed arrays independently checked with canonical FK"
            elif task["source"] == "AMASS":
                original = unique_file(roots["amass_original"], task["stem"].removesuffix("_g1") + ".npz")
                provenance.append(preserve(original, output, f"sources/{task['task_id']}/original.npz"))
                with np.load(original, allow_pickle=False) as z:
                    rec["timing"] = timing_check(n, fps, len(z["trans"]), float(z["mocap_frame_rate"]))
                rec["source_group"] = original.parent.name + "/" + original.stem
                if regenerate_sources:
                    original_csv = unique_file(roots['amass_csv'], task['stem'] + '.csv')
                    provenance.append(preserve(original_csv, output, f"sources/{task['task_id']}/retargeted.csv"))
                    csv_data = np.loadtxt(original_csv, delimiter=',')
                    raw_n, raw_fps = rec['timing']['source_frames'], rec['timing']['source_fps']
                    if raw_fps != 120 or len(csv_data) != raw_n // 4:
                        raise ValueError("AMASS source counts do not match audited producer")
                    old_t = np.arange(n) / fps
                    old_joints = np.stack([np.interp(old_t, np.arange(len(csv_data)) / 30, csv_data[:, j]) for j in range(7, 36)], axis=-1)
                    if np.max(np.abs(old_joints - a['joint_pos'])) > 1e-4:
                        raise ValueError("AMASS adjacent CSV does not reproduce inherited joint conversion")
                    rec['inherited_timing'] = rec['timing']
                    rec['inherited_source_audit'] = robot.audit(a)
                    a = rebuild_csv(csv_data, np.linspace(0, (raw_n - 1) / raw_fps, len(csv_data)), robot)
                    n, fps = len(a['joint_pos']), 50.
                    rec['timing'] = timing_check(n, fps, raw_n, raw_fps)
                    rec['rebuild'] = 'AMASS producer spans original endpoints; restore original-time grid, not nominal 30 Hz'
            else:
                original = unique_file(roots["lafan_csv"], task["stem"] + ".csv")
                provenance.append(preserve(original, output, f"sources/{task['task_id']}/original.csv"))
                csv_data = np.loadtxt(original, delimiter=",")
                if csv_data.ndim != 2 or csv_data.shape[1] != 36 or not np.isfinite(csv_data).all():
                    raise ValueError("invalid original LaFAN robot CSV")
                rec["timing"] = timing_check(n, fps, len(csv_data), 30.)
                rec["source_group"] = task["stem"]
                rec["provenance_note"] = "Retargeted LaFAN robot CSV, documented 30 Hz; not a paired human SMPL source"
                t = np.arange(n) / fps
                reconstructed = np.stack([np.interp(t, np.arange(len(csv_data)) / 30., csv_data[:, j]) for j in range(7, 36)], axis=-1)
                error = float(np.abs(reconstructed - a["joint_pos"]).max())
                rec["csv_joint_interpolation_error_rad"] = error
                if regenerate_sources:
                    rec['inherited_source_audit'] = robot.audit(a)
                    rec['inherited_timing'] = rec['timing']
                    a = rebuild_csv(csv_data, np.arange(len(csv_data)) / 30., robot)
                    n, fps = len(a['joint_pos']), 50.
                    rec['timing'] = timing_check(n, fps, len(csv_data), 30.)
                    expected = np.stack([np.interp(np.arange(n)/fps, np.arange(len(csv_data))/30., csv_data[:, j]) for j in range(7, 36)], axis=-1)
                    new_error = float(np.abs(expected - a['joint_pos']).max())
                    rec['rebuilt_csv_joint_interpolation_error_rad'] = new_error
                    if new_error > 1e-4:
                        raise ValueError("rebuilt LaFAN source interpolation mismatch")
                    rec['rebuild'] = 'Float64 source-time interpolation and SLERP; canonical CPU FK/link-origin Jacobian velocities'
                elif error > 1e-4:
                    reasons.append("LaFAN source CSV interpolation does not match transformed joints")
            if "timing" in rec and not rec["timing"]["passed"]:
                reasons.append("source/converted elapsed time mismatch")
            full_audit = robot.audit(a)
            rec["full_source_audit"] = full_audit
            if not full_audit["fk_passed"]:
                reasons.append("canonical G1 FK mismatch")
            if not full_audit["joint_limits_passed"]:
                reasons.append("canonical joint limits exceeded")
            selected, rec["window"] = window_motion(a, task.get("window_seconds"))
            rec["window_reason"] = task.get("window_reason", "Complete provided source take; no outcome-based cropping")
            rec["window_audit"] = robot.audit(selected)
            if not rec["window_audit"]["fk_passed"]:
                reasons.append("selected-window canonical G1 FK mismatch")
            # Names are recovered only from a tested correspondence, never from shape alone.
            if full_audit["fk_passed"] and rec["window_audit"]["fk_passed"]:
                selected["joint_names"] = np.asarray(robot.joint_names)
                selected["body_names"] = np.asarray(robot.body_names)
                selected["quaternion_order"] = np.asarray("wxyz")
                selected["source_start_frame"] = np.asarray(rec["window"]["start_frame"])
                dest = output / "motions" / (task["task_id"] + ".npz")
                dest.parent.mkdir(exist_ok=True)
                np.savez_compressed(dest, **selected)
                rec.update(motion_path=dest.relative_to(output).as_posix(), sha256=sha256(dest))
            rec.update(frames=len(selected["joint_pos"]), fps=fps,
                       duration=(len(selected["joint_pos"]) - 1) / fps)
            if task.get("requires_ground_contact_protocol"):
                reasons.append(GROUND_PROTOCOL_BLOCKER)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            reasons.append(str(exc))
        rec["provenance"] = provenance
        rec["quality_status"] = "blocked" if reasons else "passed"
        rec["quality"] = {"status": rec["quality_status"], "reasons": reasons}
        records.append(rec)
    groups = {}
    for rec in records:
        group = (rec["source"], rec.get("source_group", rec["stem"]))
        if group in groups and groups[group] != rec["split"]:
            raise ValueError("source group leakage across development/confirmation")
        groups[group] = rec["split"]
    manifest = {"schema": SCHEMA, "root": ".", "duration_convention": "(frames-1)/fps",
                "config_sha256": sha256(config), "canonical_g1": robot.identity,
                "producer_evidence": producer_evidence, "regenerated_sources": regenerate_sources,
                "selection": "prospective shortlist; input QA only; no historical policy outcomes",
                "counts": dict(Counter(r["quality_status"] for r in records)), "tasks": records}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    lines = ["# Motion Study Input Eligibility", "", "No training or policy evaluation was performed.", "",
             "| Task | Split | Frames | Seconds | Status | Blocking reasons |", "|---|---|---:|---:|---|---|"]
    for r in records:
        lines.append(f"| {r['task_id']} | {r['split']} | {r['frames']} | {r['duration']} | {r['quality_status']} | {'; '.join(r['quality']['reasons'])} |")
    lines += ["", "FK uses nine spread-out frames per full input and per selected window, with 2 mm / 0.002 rad tolerances.",
              "Joint limits and finite/rotation checks cover every source frame. Contact diagnostics are kinematic, not physics results.",
              "Blocked tasks remain in the fixed 6/12 cohort. Do not screen by training outcome or silently substitute tasks.",
              "Sources and derived licensed motions are local-only, not public release assets."]
    (output / "AUDIT.md").write_text("\n".join(lines) + "\n")
    return manifest


def load_eligible_manifest(path: Path, subset: str = "all", root: Path | None = None) -> list[dict]:
    """Fail the selected cohort closed; never silently filter failed tasks."""
    m = json.loads(path.read_text())
    if m.get("schema") != SCHEMA or subset not in ("all", "development", "confirmation"):
        raise ValueError("invalid manifest/schema/subset")
    if Counter(t["split"] for t in m["tasks"]) != {"development": 6, "confirmation": 12}:
        raise ValueError("incomplete study cohort")
    if len({t['task_id'] for t in m['tasks']}) != 18:
        raise ValueError("duplicate task identities")
    base = (root or path.parent).resolve()
    selected = [t for t in m["tasks"] if subset == "all" or t["split"] == subset]
    for t in selected:
        if t.get("quality_status") != "passed" or t.get("quality", {}).get("status") != "passed":
            raise ValueError(f"ineligible task: {t['task_id']}")
        p = (base / t["motion_path"]).resolve()
        if not p.is_relative_to(base) or sha256(p) != t["sha256"]:
            raise ValueError(f"motion integrity/path failure: {t['task_id']}")
    return selected


def export_isaac_pilot(manifest_path: Path, task_id: str, articulation: Path, output: Path) -> dict:
    """Reorder one eligible input; leave live cross-asset FK to the runtime owner."""
    from mimicx.adapters.isaac_motion import convert_motion, validate_motion
    manifest = json.loads(manifest_path.read_text())
    matches = [t for t in manifest["tasks"] if t["task_id"] == task_id]
    if manifest.get("schema") != SCHEMA or len(matches) != 1:
        raise ValueError("invalid manifest or ambiguous pilot")
    task = matches[0]
    if task.get("quality_status") != "passed" or task["quality"]["status"] != "passed":
        raise ValueError("pilot is blocked")
    source = (manifest_path.parent / task["motion_path"]).resolve()
    if not source.is_relative_to(manifest_path.parent.resolve()) or sha256(source) != task["sha256"]:
        raise ValueError("pilot source identity mismatch")
    live = json.loads(articulation.read_text())
    arrays = read_motion(source)
    robot = CanonicalG1(Path(manifest["canonical_g1"]["xml_origin"]))
    if robot.identity["xml_sha256"] != manifest["canonical_g1"]["xml_sha256"]:
        raise ValueError("canonical asset changed since source audit")
    audit = robot.audit(arrays)
    if not audit["fk_passed"]:
        raise ValueError("source FK failed")
    mapping = dict(source_joints=arrays["joint_names"].tolist(), source_bodies=arrays["body_names"].tolist(),
                   target_joints=live["joint_names"], target_bodies=live["body_names"],
                   quaternion_order="wxyz", units="SI", angular_velocity_frame="world", expected_fps=50.)
    output.mkdir(parents=True, exist_ok=False)
    destination = output / (task_id + "_isaac.npz")
    convert_motion(source, destination, **mapping)
    with np.load(destination, allow_pickle=False) as z:
        converted = {k: z[k] for k in z.files}
    validate_motion(converted, joint_names=live["joint_names"], body_names=live["body_names"], fps=50.)
    permutations = {}
    for family in ("joint", "body"):
        plural = "joints" if family == "joint" else "bodies"
        src, dst = mapping["source_" + plural], mapping["target_" + plural]
        order = [src.index(name) for name in dst]
        permutations[family + "_target_to_source"] = order
        for field in FIELDS:
            if field.startswith(family + "_"):
                if not np.array_equal(converted[field], arrays[field][:, order].astype(np.float32)):
                    raise ValueError("conversion changed values beyond named permutation")
    (output / "mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
    report = dict(task_id=task_id, source_path=str(source), source_sha256=sha256(source),
                  manifest_sha256=sha256(manifest_path), articulation_sha256=sha256(articulation),
                  articulation_origin=str(articulation), motion_path=destination.name,
                  sha256=sha256(destination), frames=task["frames"], fps=task["fps"],
                  duration=task["duration"], source_canonical_fk=audit,
                  permutations=permutations, source_order_evidence=manifest["canonical_g1"],
                  live_isaac_fk_status="pending: runtime owner must verify actual asset before training",
                  velocity_semantics="SI world-frame body-link origin velocities, not COM velocities; reordering only")
    (output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def protocol_digest(protocol: dict) -> str:
    return hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def freeze_ground_contact_protocol(manifest_path: Path, output: Path, protocol: dict,
                                   agreements: dict) -> dict:
    """Issue a new manifest after explicit cross-owner agreement; never edit inputs."""
    required = {'root_height_m': .25, 'relative_ee_height_m': .25,
                'root_orientation_rad': 1., 'persistence_frames': 5,
                'common_prefix_seconds': 1., 'clock_tolerance_s': .03, 'state_gap_s': .045,
                'critical_bodies': ['left_ankle_roll_link', 'right_ankle_roll_link',
                                    'left_wrist_yaw_link', 'right_wrist_yaw_link'],
                'root_body': 'pelvis', 'fps': 50, 'prospective': True}
    if not protocol.get('protocol_id') or any(protocol.get(k) != v for k, v in required.items()):
        raise ValueError('protocol differs from jointly reviewed ground-contact contract')
    digest = protocol_digest(protocol)
    if set(agreements) != {'native_coordinator', 'sonic_scorer'}:
        raise ValueError('both method-owner agreements required')
    for agreement in agreements.values():
        if agreement.get('protocol_sha256') != digest or not agreement.get('reference'):
            raise ValueError('agreement does not cover exact protocol')
    source = manifest_path.resolve()
    destination = output.resolve()
    if destination.is_relative_to(source.parent):
        raise ValueError('new version must be outside source directory')
    m = json.loads(source.read_text())
    if m.get('schema') != SCHEMA or Counter(t['split'] for t in m['tasks']) != {'development': 6, 'confirmation': 12}:
        raise ValueError('invalid source cohort')
    if len({t['task_id'] for t in m['tasks']}) != 18:
        raise ValueError('duplicate task IDs')
    before_hash = sha256(source)
    for t in m['tasks']:
        if t.get('quality_status') != t.get('quality', {}).get('status'):
            raise ValueError('inconsistent quality statuses')
        if t.get('motion_path'):
            path = (source.parent / t['motion_path']).resolve()
            if not path.is_relative_to(source.parent) or sha256(path) != t['sha256']:
                raise ValueError('input identity failed before protocol freeze')
    shutil.copytree(source.parent, destination)
    shutil.copyfile(source, destination / 'input-manifest-before-protocol.json')
    if (destination / 'AUDIT.md').exists():
        shutil.copyfile(destination / 'AUDIT.md', destination / 'INPUT_AUDIT_BEFORE_PROTOCOL.md')
    m['protocol'] = protocol
    m['protocol_sha256'] = digest
    m['protocol_agreements'] = agreements
    m['parent_manifest_sha256'] = before_hash
    for t in m['tasks']:
        t['protocol_id'] = protocol['protocol_id']
        t['protocol_sha256'] = digest
        if t.get('requires_ground_contact_protocol'):
            t['ground_contact_protocol'] = {'status': 'frozen', 'protocol_sha256': digest,
                                            'native_training': 'unchanged', 'common_evaluation': 'reference_relative'}
            t['quality']['reasons'] = [r for r in t['quality']['reasons'] if r != GROUND_PROTOCOL_BLOCKER]
            # An unexplained blocked status is not promoted by agreement alone.
            old = next(x for x in json.loads(source.read_text())['tasks'] if x['task_id'] == t['task_id'])
            if old['quality']['reasons'] == [GROUND_PROTOCOL_BLOCKER]:
                t['quality_status'] = t['quality']['status'] = 'passed'
    m['counts'] = dict(Counter(t['quality_status'] for t in m['tasks']))
    (destination / 'protocol.json').write_text(json.dumps(protocol, indent=2, allow_nan=False) + '\n')
    (destination / 'manifest.json').write_text(json.dumps(m, indent=2, allow_nan=False) + '\n')
    lines = ['# Protocol-Frozen Input Eligibility', '',
             f"Protocol: `{protocol['protocol_id']}`; SHA-256: `{digest}`.",
             'Prospective agreement; no motion arrays or numerical QA thresholds changed.',
             'Native training termination remains unchanged; common evaluation is reference-relative.',
             'Input eligibility does not override independent runtime/asset compatibility gates.', '',
             '| Task | Split | Status | Blocking reasons |', '|---|---|---|---|']
    for t in m['tasks']:
        lines.append(f"| {t['task_id']} | {t['split']} | {t['quality_status']} | {'; '.join(t['quality']['reasons'])} |")
        if t.get('motion_path') and sha256(destination / t['motion_path']) != t['sha256']:
            raise ValueError('copied motion identity mismatch')
    (destination / 'AUDIT.md').write_text('\n'.join(lines) + '\n')
    if sha256(source) != before_hash:
        raise ValueError('source manifest changed during freeze')
    return m
