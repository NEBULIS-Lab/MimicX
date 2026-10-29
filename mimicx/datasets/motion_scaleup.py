"""Offline, prospective motion scale-up. Never reads policy outcomes or uses CUDA."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import subprocess
import tarfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import numpy as np

from mimicx.datasets.motion_study import (
    CanonicalG1, FIELDS, preserve, read_motion, rebuild_csv, sha256,
    timing_check, window_motion,
)

SCHEMA = "mimicx.scaleup.cohort.v1"
SEED = 20261029
ROOT_KEYS = {"bones_archive", "bones_metadata", "bones_npz", "amass_npz",
             "amass_original", "amass_csv", "lafan_npz", "lafan_csv", "g1_xml",
             "old_manifest", "bones_converter"}


def load_roots(path: Path) -> dict[str, Path]:
    data = json.loads(path.read_text())
    if set(data) - ROOT_KEYS:
        raise ValueError("unknown source roots; outcome roots are not accepted")
    return {k: Path(v) for k, v in data.items()}


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def bones_identity(row: dict) -> str:
    stem = PurePosixPath(row["move_g1_path"]).with_suffix("").as_posix()
    return "BONES:" + re.sub(r"_M$", "", stem)


def bones_group(row: dict) -> str:
    # Actors and mirrored releases of the same recorded take stay together.
    take = re.sub(r"_M$", "", row["take_name"])
    return "BONES:" + row.get("take_date", "") + ":" + take


@contextmanager
def archive_stream(path: Path):
    """Stream a local archive; never extract paths supplied by tar headers."""
    import shutil
    if shutil.which("pigz"):
        p = subprocess.Popen(["pigz", "-dc", "-p", "2", str(path)], stdout=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=p.stdout, mode="r|") as tar:
                yield tar
        finally:
            p.stdout.close()
            if p.poll() is None:
                p.terminate()
            code = p.wait()
        if code:
            raise ValueError(f"archive decompressor failed: {code}")
    else:
        with tarfile.open(path, "r|gz") as tar:
            yield tar


def archive_inventory(path: Path, capture: dict[str, Path] | None = None) -> dict:
    before = path.stat()
    names = []
    with archive_stream(path) as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".csv"):
                p = PurePosixPath(member.name)
                if p.is_absolute() or ".." in p.parts:
                    raise ValueError("unsafe archive member")
                names.append(p.as_posix().removeprefix("./"))
                if capture and names[-1] in capture:
                    destination = capture[names[-1]]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    raw = tar.extractfile(member).read()
                    if destination.exists():
                        if destination.read_bytes() != raw:
                            raise ValueError("captured source changed")
                    else:
                        destination.write_bytes(raw)
    if len(set(names)) != len(names):
        raise ValueError("duplicate CSV archive identities")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("archive changed during inventory")
    return dict(path=str(path.resolve()), bytes=before.st_size, mtime_ns=before.st_mtime_ns,
                csv_count=len(names), nonmirror_csv_count=sum(not n.endswith("_M.csv") for n in names),
                members=sorted(names), complete=True)


def read_metadata(path: Path) -> dict[str, dict]:
    result = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = row["move_g1_path"]
            if key in result:
                raise ValueError("duplicate BONES metadata source identity")
            result[key] = row
    return result


def inventory(roots: dict[str, Path], output: Path) -> dict:
    metadata = read_metadata(roots["bones_metadata"])
    capture = {}
    for p in roots["bones_npz"].rglob("*.npz"):
        rel = p.relative_to(roots["bones_npz"]).with_suffix(".csv").as_posix()
        if not rel.startswith("g1/csv/"):
            rel = "g1/csv/" + rel
        capture[rel] = output.parent / "raw-selected" / rel
    archive = archive_inventory(roots["bones_archive"], capture)
    members = set(archive["members"])
    no_props = [r for r in metadata.values() if r["is_mirror"].lower() == "false"
                and r["content_props"] in ("0", "0.0") and r["move_g1_path"] in members]
    counts = {}
    for key in ("bones_npz", "amass_npz", "lafan_npz", "amass_csv", "lafan_csv", "amass_original"):
        if key in roots:
            ext = ".csv" if key.endswith("csv") else ".npz"
            files = sorted(str(p) for p in roots[key].rglob("*" + ext))
            if key.startswith("lafan"):
                files = [p for p in files if re.search(r"_subject[1-5]\.(npz|csv)$", p)]
            counts[key] = dict(count=len(files), files=files)
    result = dict(schema="mimicx.scaleup.inventory.v1", raw_archive=archive,
                  metadata_sha256=sha256(roots["bones_metadata"]), metadata_rows=len(metadata),
                  nonmirror_metadata=sum(r["is_mirror"].lower() == "false" for r in metadata.values()),
                  local_nonmirror_no_props_originals=len({bones_identity(r) for r in no_props}),
                  local_nonmirror_no_props_groups=len({bones_group(r) for r in no_props}),
                  input_qa_target=1000, eligible_input_count=None, eligible_count_note="Requires numeric QA, not archive counts",
                  external_inputs=counts)
    result["captured_raw"] = {k: {"path": str(p), "sha256": sha256(p)}
                              for k, p in capture.items() if p.is_file()}
    write_json(output, result)
    return result


def decode_bones(raw: bytes, joint_names: list[str], expected_frames: int) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation
    f = io.StringIO(raw.decode("utf-8-sig"))
    header = next(csv.reader(f))
    expected = ["Frame", *["root_translate" + x for x in "XYZ"],
                *["root_rotate" + x for x in "XYZ"], *[n + "_dof" for n in joint_names]]
    if header != expected:
        raise ValueError("BONES columns differ from named canonical order")
    values = np.loadtxt(f, delimiter=",", ndmin=2)
    if values.shape != (expected_frames, 36) or expected_frames < 2:
        raise ValueError("BONES frame count/shape differs from metadata")
    if not np.isfinite(values).all():
        raise ValueError("nonfinite BONES source values")
    if not np.array_equal(values[:, 0], np.arange(expected_frames)):
        raise ValueError("non-contiguous BONES Frame clock")
    q = np.empty((len(values), 36))
    q[:, :3] = values[:, 1:4] / 100.
    q[:, 3:7] = Rotation.from_euler("xyz", values[:, 4:7], degrees=True).as_quat()
    q[:, 7:] = np.deg2rad(values[:, 7:])
    return q, np.arange(len(values)) / 120.


def prospective_window(frames: int, fps: float, seconds: float = 12.) -> list[float] | None:
    if frames < 2 or not np.isfinite([fps, seconds]).all() or min(fps, seconds) <= 0:
        raise ValueError("invalid prospective window clock")
    span = int(round(seconds * fps))
    if span < 1:
        raise ValueError("window too short")
    if frames - 1 <= span:
        return None
    start = (frames - 1 - span) // 2
    return [start / fps, (start + span) / fps]


def rank(record: dict, seed: int = SEED) -> tuple[str, str]:
    clip = record["original_clip_id"]
    return hashlib.sha256(f"{seed}:{clip}".encode()).hexdigest(), record["task_id"]


def coverage_plan(metadata: dict, members: set, target: int = 1000) -> dict:
    candidates = []
    for key, row in metadata.items():
        if key not in members or row["is_mirror"].lower() != "false" or row["content_props"] not in ("0", "0.0"):
            continue
        candidates.append(dict(task_id=key, original_clip_id=bones_identity(row),
                               clip_id=bones_identity(row), source_group=bones_group(row),
                               raw_archive_member=key, source_metadata=row,
                               source_full_duration_seconds=(int(row["move_duration_frames"]) - 1) / 120.,
                               qa_status="pending"))
    inputs, groups = [], set()
    for record in sorted(candidates, key=rank):
        if record["source_group"] in groups:
            continue
        groups.add(record["source_group"])
        inputs.append(record)
        if len(inputs) == target:
            break
    return dict(schema="mimicx.scaleup.coverage-input-plan.v1", target=target,
                selection_seed=SEED, inputs=inputs, eligible_inputs=0, evaluated_inputs=0,
                trained_mimicx_policies=0,
                scope="Prospective input QA then released-policy baseline-only coverage; not matched MimicX evidence")


def validate_coverage_plan(plan: dict, expected_count: int = 1000) -> list[dict]:
    if plan.get("schema") != "mimicx.scaleup.coverage-input-plan.v1" or plan.get("target") != expected_count:
        raise ValueError("invalid frozen coverage plan")
    records = []
    for item in plan["inputs"]:
        member = PurePosixPath(item["raw_archive_member"])
        row = item["source_metadata"]
        if (member.is_absolute() or ".." in member.parts or member.as_posix() != row["move_g1_path"]
                or item["clip_id"] != bones_identity(row) or item["source_group"] != bones_group(row)
                or row["is_mirror"].lower() != "false" or row["content_props"] not in ("0", "0.0")):
            raise ValueError("invalid coverage source identity")
        records.append(dict(item, source_task_id=item["task_id"],
                            task_id="coverage_" + hashlib.sha256(item["clip_id"].encode()).hexdigest()[:16],
                            quality_status="pending", extraction_status="pending", reasons=[]))
    if len(records) != expected_count or any(len({r[k] for r in records}) != expected_count
                                            for k in ("clip_id", "source_group", "raw_archive_member", "task_id")):
        raise ValueError("coverage must retain the exact distinct input denominator")
    return records


def budget_write(path: Path, payload: bytes, used: int, limit: int) -> int:
    if used + len(payload) > limit:
        raise ValueError("coverage disk budget exceeded before write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(payload)
    return used + len(payload)


def extract_coverage(archive: Path, records: list[dict], output: Path, limit: int) -> int:
    """One single-threaded gzip pass, capturing only frozen selected members."""
    wanted = {r["raw_archive_member"]: r for r in records}
    used, seen = 0, set()
    before = archive.stat()
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            name = PurePosixPath(member.name).as_posix()
            if name not in wanted:
                continue
            record = wanted[name]
            if name in seen or not member.isfile():
                record.update(quality_status="rejected", extraction_status="rejected")
                record["reasons"].append("duplicate or nonregular archive member")
                continue
            seen.add(name)
            if used + member.size > limit:
                record.update(quality_status="rejected", extraction_status="rejected")
                record["reasons"].append("disk budget prevents extraction")
                continue
            raw = tar.extractfile(member).read()
            relative = "raw/" + record["task_id"] + ".csv"
            used = budget_write(output / relative, raw, used, limit)
            record.update(source_path=relative, raw_sha256=hashlib.sha256(raw).hexdigest(),
                          extraction_status="captured", raw_bytes=len(raw))
            if len(seen) % 25 == 0:
                print(json.dumps(dict(stage="extract", seen=len(seen), denominator=len(records), bytes=used)), flush=True)
    after = archive.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("archive changed during extraction")
    for record in records:
        if record["extraction_status"] == "pending":
            record.update(quality_status="rejected", extraction_status="missing")
            record["reasons"].append("selected raw archive member missing")
    return used


def canonicalize_coverage(raw: bytes, record: dict, robot: CanonicalG1) -> tuple[dict, dict]:
    q, times = decode_bones(raw, robot.joint_names, int(record["source_metadata"]["move_duration_frames"]))
    limits = robot.model.jnt_range[1:]
    excess = np.maximum(limits[:, 0] - q[:, 7:], q[:, 7:] - limits[:, 1])
    excess[:, ~robot.model.jnt_limited[1:].astype(bool)] = 0
    raw_excess = float(np.maximum(excess, 0).max())
    arrays = rebuild_csv(q, times, robot)
    arrays["joint_names"] = np.array(robot.joint_names)
    arrays["body_names"] = np.array(robot.body_names)
    audit = robot.audit(arrays)
    n = len(arrays["joint_pos"])
    timing = timing_check(n, 50., len(q), 120.)
    reasons = check_audit(audit)
    if raw_excess > 1e-4:
        reasons.append("raw 120Hz joint limits exceeded")
    if not timing["passed"]:
        reasons.append("source/converted elapsed time mismatch")
    if not all(np.isfinite(arrays[k]).all() for k in FIELDS):
        reasons.append("nonfinite canonical arrays")
    if np.max(np.abs(np.linalg.norm(arrays["body_quat_w"], axis=-1) - 1)) > 1e-3:
        reasons.append("nonunit canonical rotations")
    return arrays, dict(frames=n, fps=50., duration=(n - 1) / 50.,
                        source_full_duration_seconds=times[-1], scored_window_duration_seconds=(n - 1) / 50.,
                        window={"start_frame": 0, "end_frame_inclusive": n - 1},
                        timing=timing, canonical_audit=audit, raw_joint_limit_max_excess_rad=raw_excess,
                        quality_status="rejected" if reasons else "passed", reasons=reasons)


def prepare_coverage(roots: dict, plan_path: Path, inventory_path: Path, output: Path,
                     disk_budget: int = 5_000_000_000) -> dict:
    import os
    import shutil
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("coverage requires explicit empty CUDA_VISIBLE_DEVICES")
    if not 64_000_000 < disk_budget <= 5_000_000_000:
        raise ValueError("coverage disk budget must not exceed 5 GB")
    plan = json.loads(plan_path.read_text())
    records = validate_coverage_plan(plan)
    inv = json.loads(inventory_path.read_text())
    stat = roots["bones_archive"].stat()
    if (stat.st_size, stat.st_mtime_ns) != (inv["raw_archive"]["bytes"], inv["raw_archive"]["mtime_ns"]):
        raise ValueError("archive identity differs from frozen inventory")
    output.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(output).free < disk_budget:
        raise ValueError("insufficient disk headroom for bounded coverage")
    preserve(plan_path, output, "input-plan.json")
    robot = CanonicalG1(roots["g1_xml"])
    result = dict(schema="mimicx.scaleup.coverage-qa.v1", denominator=len(records), tasks=records,
                  plan_sha256=sha256(plan_path), inventory_sha256=sha256(inventory_path),
                  canonical_g1=robot.identity, complete=False, disk_budget_bytes=disk_budget,
                  evaluated_inputs=0, trained_mimicx_policies=0, replacement_policy="none",
                  scope="Input QA only; future released-policy baseline coverage, not matched MimicX evidence")

    def checkpoint(stage):
        result["stage"] = stage
        result["counts"] = dict(Counter(r["quality_status"] for r in records))
        temporary = output / "manifest.tmp"
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "manifest.json")
        print(json.dumps(dict(stage=stage, counts=result["counts"], denominator=len(records))), flush=True)

    checkpoint("extract")
    used = extract_coverage(roots["bones_archive"], records, output, disk_budget - 64_000_000)
    checkpoint("canonicalize")
    for i, record in enumerate(records):
        if record["quality_status"] == "rejected":
            continue
        try:
            raw = (output / record["source_path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != record["raw_sha256"]:
                raise ValueError("raw source integrity changed")
            arrays, audit = canonicalize_coverage(raw, record, robot)
            record.update(audit)
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **arrays)
            relative = "motions/" + record["task_id"] + ".npz"
            used = budget_write(output / relative, buffer.getvalue(), used, disk_budget - 64_000_000)
            record.update(motion_path=relative, sha256=sha256(output / relative))
            read_motion(output / relative)
        except (ValueError, KeyError, OSError, UnicodeError) as error:
            record["quality_status"] = "rejected"
            record["reasons"].append(str(error))
        if (i + 1) % 10 == 0:
            checkpoint("canonicalize")
    result["complete"] = True
    result["payload_bytes"] = used
    checkpoint("complete")
    return result


def select_tasks(old: list[dict], candidates: list[dict], target: int = 60,
                 development: int = 12, reserve_count: int = 24) -> dict:
    if target < 18 or development < 6 or target - development < 12:
        raise ValueError("invalid cohort counts")
    if len(old) != 18 or Counter(t["split"] for t in old) != {"development": 6, "confirmation": 12}:
        raise ValueError("must preserve old six development and twelve confirmation inputs")
    clips, groups = set(), {}
    selected = []
    for task in old:
        clip, group = task["original_clip_id"], task["split_group"]
        if clip in clips or (group in groups and groups[group] != task["split"]):
            raise ValueError("old duplicate clip or cross-split group")
        clips.add(clip)
        groups[group] = task["split"]
        selected.append(dict(task))
    reserves = []
    for candidate in sorted(candidates, key=rank):
        if candidate.get("quality_status") != "passed":
            continue
        clip, group = candidate["original_clip_id"], candidate["split_group"]
        if clip in clips or group in groups:
            continue
        split = "development" if sum(t["split"] == "development" for t in selected) < development else "confirmation"
        task = dict(candidate, split=split)
        clips.add(clip)
        groups[group] = split
        if len(selected) < target:
            selected.append(task)
        elif len(reserves) < reserve_count:
            task["split"] = "reserve"
            task["replacement_rule"] = "input/protocol defect only; parent logs reason before replacement; never policy outcome"
            reserves.append(task)
    return dict(tasks=selected, reserves=reserves, ready=len(selected) == target,
                missing=max(0, target - len(selected)), target=target,
                counts=dict(Counter(t["split"] for t in selected)))


def check_audit(audit: dict) -> list[str]:
    return (["canonical FK mismatch"] if not audit["fk_passed"] else []) + (
        ["joint limits exceeded"] if not audit["joint_limits_passed"] else [])


def compare_raw(q: np.ndarray, times: np.ndarray, arrays: dict) -> dict:
    """No fitting, offsets, yaw alignment, or time warping are permitted."""
    from scipy.spatial.transform import Rotation, Slerp
    t = np.arange(len(arrays["joint_pos"])) / float(arrays["fps"].reshape(-1)[0])
    if t[-1] > times[-1] + 1e-8:
        raise ValueError("converted clock extends past raw source")
    expected = np.stack([np.interp(t, times, q[:, j]) for j in range(36) if j < 3 or j >= 7], axis=-1)
    joints = float(np.max(np.abs(expected[:, 3:] - arrays["joint_pos"])))
    root = float(np.max(np.linalg.norm(expected[:, :3] - arrays["body_pos_w"][:, 0], axis=-1)))
    rotation = Slerp(times, Rotation.from_quat(q[:, 3:7]))(t)
    actual = Rotation.from_quat(arrays["body_quat_w"][:, 0, [1, 2, 3, 0]])
    angle = float(np.max((rotation.inv() * actual).magnitude()))
    return dict(joint_max_error_rad=joints, root_max_error_m=root, quaternion_max_error_rad=angle,
                joint_tolerance_rad=1e-4, root_tolerance_m=1e-4, quaternion_tolerance_rad=.002,
                passed=joints <= 1e-4 and root <= 1e-4 and angle <= .002)


def protocol_fields(manifest: dict) -> dict:
    from mimicx.datasets.motion_study import protocol_digest
    if protocol_digest(manifest["protocol"]) != manifest["protocol_sha256"]:
        raise ValueError("old protocol digest mismatch")
    return {k: manifest[k] for k in ("protocol", "protocol_sha256", "protocol_agreements")}


def inherit_old(path: Path, metadata: dict, robot: CanonicalG1) -> list[dict]:
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "mimicx.motion-study.v1":
        raise ValueError("invalid old manifest")
    if manifest["canonical_g1"]["xml_sha256"] != robot.identity["xml_sha256"]:
        raise ValueError("canonical asset differs from old cohort")
    result = []
    for old in manifest["tasks"]:
        p = (path.parent / old["motion_path"]).resolve()
        if not p.is_relative_to(path.parent.resolve()) or sha256(p) != old["sha256"]:
            raise ValueError("old motion identity changed")
        arrays = read_motion(p)
        reasons = check_audit(robot.audit(arrays))
        if reasons or old["quality_status"] != "passed":
            raise ValueError(f"old task no longer input-eligible: {old['task_id']}: {reasons}")
        if old["source"] == "BONES":
            row = metadata["g1/csv/" + old["stem"] + ".csv"]
            clip, group = bones_identity(row), bones_group(row)
        else:
            clip = old["source"] + ":" + old["source_group"]
            group = clip
        result.append(dict(old, original_clip_id=clip, clip_id=clip, split_group=group,
                           source_group=group, inherited_source_group=old["source_group"],
                           previously_seen=old["split"] == "development", inherited=True,
                           input_origin=str(p), provenance_ref=str(path.resolve()),
                           old_manifest_sha256=sha256(path),
                           source_full_duration_seconds=old["timing"]["source_duration"],
                           scored_window_duration_seconds=old["duration"]))
    return result


def audit_bones_npz(roots: dict, metadata: dict, members: set, robot: CanonicalG1,
                    captured: dict) -> list[dict]:
    records = []
    for p in sorted(roots["bones_npz"].rglob("*.npz")):
        rel = p.relative_to(roots["bones_npz"]).with_suffix(".csv").as_posix()
        if not rel.startswith("g1/csv/"):
            rel = "g1/csv/" + rel
        row = metadata.get(rel)
        rec = dict(task_id="scale_bones_" + hashlib.sha256(rel.encode()).hexdigest()[:12],
                   source="BONES", stem=rel.removeprefix("g1/csv/").removesuffix(".csv"),
                   input_origin=str(p.resolve()), quality_status="rejected", reasons=[],
                   previously_seen=False, inherited=False)
        try:
            if row is None or rel not in members:
                raise ValueError("missing metadata or raw archive member")
            rec.update(original_clip_id=bones_identity(row), clip_id=bones_identity(row),
                       split_group=bones_group(row), source_group=bones_group(row),
                       raw_archive_member=rel, source_metadata=row,
                       provenance_ref=str(roots["bones_metadata"].resolve()) + "#" + rel)
            if row["is_mirror"].lower() != "false":
                raise ValueError("mirrored release excluded from distinct-original count")
            if row["content_props"] not in ("0", "0.0"):
                raise ValueError("external prop not in flat-ground scene")
            a = read_motion(p)
            evidence = captured.get(rel)
            if not evidence or sha256(Path(evidence["path"])) != evidence["sha256"]:
                raise ValueError("missing or changed captured raw CSV")
            q, times = decode_bones(Path(evidence["path"]).read_bytes(), robot.joint_names,
                                    int(row["move_duration_frames"]))
            rec["raw_source"] = evidence
            rec["raw_value_consistency"] = compare_raw(q, times, a)
            if not rec["raw_value_consistency"]["passed"]:
                rec["reasons"].append("raw CSV / inherited NPZ value mismatch")
            rec["sha256"] = sha256(p)
            n, fps = len(a["joint_pos"]), float(a["fps"].reshape(-1)[0])
            rec["timing"] = timing_check(n, fps, int(row["move_duration_frames"]), 120.)
            rec["full_source_audit"] = robot.audit(a)
            rec["reasons"] += check_audit(rec["full_source_audit"])
            if fps != 50 or not rec["timing"]["passed"]:
                rec["reasons"].append("source/converted timing mismatch or not 50 Hz")
            rec.update(frames=n, fps=fps, duration=(n - 1) / fps,
                       source_full_duration_seconds=(int(row["move_duration_frames"]) - 1) / 120.,
                       scored_window_duration_seconds=(n - 1) / fps,
                       window=dict(start_frame=0, end_frame_inclusive=n - 1, start_seconds=0., end_seconds=(n - 1) / fps),
                       window_reason="Full original short clip; no overlapping windows or retiming")
            rec["quality_status"] = "passed" if not rec["reasons"] else "rejected"
        except (ValueError, KeyError, OSError) as error:
            rec["reasons"].append(str(error))
        records.append(rec)
    return records


def prepare(roots: dict[str, Path], inventory_path: Path, output: Path,
            declared_at: str) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be empty; never overwrite a frozen cohort")
    inv = json.loads(inventory_path.read_text())
    archive = roots["bones_archive"].stat()
    if (archive.st_size, archive.st_mtime_ns) != (inv["raw_archive"]["bytes"], inv["raw_archive"]["mtime_ns"]):
        raise ValueError("archive changed after inventory")
    if sha256(roots["bones_metadata"]) != inv["metadata_sha256"]:
        raise ValueError("metadata changed after inventory")
    output.mkdir(parents=True, exist_ok=True)
    robot = CanonicalG1(roots["g1_xml"])
    metadata = read_metadata(roots["bones_metadata"])
    old = inherit_old(roots["old_manifest"], metadata, robot)
    records = audit_bones_npz(roots, metadata, set(inv["raw_archive"]["members"]), robot,
                              inv.get("captured_raw", {}))
    selected = select_tasks(old, records)
    for task in selected["tasks"] + selected["reserves"]:
        evidence = preserve(Path(task["input_origin"]), output, "motions/" + task["task_id"] + ".npz")
        if task.get("sha256") and task["sha256"] != evidence["sha256"]:
            raise ValueError("input changed after QA")
        task.update(motion_path=evidence["path"], sha256=evidence["sha256"],
                    quality={"status": "passed", "reasons": []})
        task["inherited_source_path"] = task.get("source_path")
        task["source_path"] = evidence["path"]
        task["quality"]["provenance"] = [evidence]
        if task.get("inherited"):
            task["provenance_ref"] = "provenance/old-manifest.json#task=" + task["task_id"]
        if task.get("raw_source"):
            raw = preserve(Path(task["raw_source"]["path"]), output,
                           "sources/" + task["task_id"] + ".csv")
            task["quality"]["provenance"].append(raw)
            task["provenance_ref"] = raw["path"]
    preserve(roots["old_manifest"], output, "provenance/old-manifest.json")
    preserve(roots["bones_converter"], output, "provenance/bones_converter.py")
    manifest = dict(schema="mimicx.scaleup.motion-inputs.v1", declared_at=declared_at,
                    duration_convention="(frames-1)/fps; source full and scored window reported separately",
                    canonical_g1=robot.identity, inventory_sha256=sha256(inventory_path),
                    old_manifest_sha256=sha256(roots["old_manifest"]),
                    selection_seed=SEED, selection_rule="hash(original clip), group-disjoint, input QA only; old18 immutable",
                    protocol_status="Parent owns scaleup protocol agreement and launch; input readiness is not launch authorization",
                    selection=selected, tasks=selected["tasks"], reserves=selected["reserves"],
                    input_audit_counts=dict(Counter(r["quality_status"] for r in records)),
                    candidate_qa=records, readiness=selected["ready"], counts=selected["counts"],
                    coverage=dict(target_inputs=1000, purpose="input QA and released-policy coverage first; not MimicX outputs or policies",
                                  evaluated_inputs=0, mimicx_policy_count=0))
    manifest.update(protocol_fields(json.loads(roots["old_manifest"].read_text())))
    manifest["distinct_original_count"] = len({t["clip_id"] for t in selected["tasks"]})
    manifest["new_relative_to_old_six"] = sum(not t["previously_seen"] for t in selected["tasks"])
    manifest["new_relative_to_old_eighteen"] = sum(not t["inherited"] for t in selected["tasks"])
    write_json(output / "manifest.json", manifest)
    declaration = dict(schema=SCHEMA, declared_at=declared_at,
                       selection_seed=SEED, selection_rule=manifest["selection_rule"],
                       input_manifest_sha256=sha256(output / "manifest.json"),
                       ready=selected["ready"], tasks=selected["tasks"], reserves=selected["reserves"])
    write_json(output / "cohort.json", declaration)
    write_json(output / "development-ready.json", dict(schema="mimicx.scaleup.motion-inputs.v1",
               tasks=[t for t in selected["tasks"] if t["split"] == "development"],
               canonical_g1=robot.identity, parent_manifest_sha256=sha256(output / "manifest.json"),
               **protocol_fields(manifest)))
    write_json(output / "coverage-input-plan.json", coverage_plan(metadata, set(inv["raw_archive"]["members"])))
    return manifest
