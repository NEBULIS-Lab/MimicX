import csv
import io
import json
import tarfile
from collections import Counter

import numpy as np
import pytest

from mimicx.datasets import motion_scaleup as ms


def row(stem="walk_001__A001", take="walk_001", actor="A001"):
    return dict(move_g1_path=f"g1/csv/240101/{stem}.csv", take_date="240101",
                take_name=take, take_actor=actor, is_mirror="False", content_props="0",
                move_duration_frames="1201", category="Walk")


def test_group_mirror_and_source_take():
    a = row()
    b = dict(a, move_g1_path=a["move_g1_path"].replace(".csv", "_M.csv"), is_mirror="True")
    assert ms.bones_identity(a) == ms.bones_identity(b)
    assert ms.bones_group(a) == ms.bones_group(b)
    assert ms.bones_group(a) == ms.bones_group(dict(a, take_actor="A002"))


def test_archive_inventory_counts_actual_members(tmp_path):
    archive = tmp_path / "g1.tar.gz"
    names = [row()["move_g1_path"], row()["move_g1_path"].replace(".csv", "_M.csv")]
    with tarfile.open(archive, "w:gz") as tar:
        for name in names:
            data = b"Frame\n0\n"
            member = tarfile.TarInfo(name)
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
    result = ms.archive_inventory(archive)
    assert result["csv_count"] == 2
    assert result["nonmirror_csv_count"] == 1
    assert result["members"] == sorted(names)


def test_bones_decode_units_names_and_clock():
    names = [f"j{i}" for i in range(29)]
    text = io.StringIO()
    writer = csv.writer(text)
    writer.writerow(["Frame", *["root_translate" + x for x in "XYZ"],
                     *["root_rotate" + x for x in "XYZ"], *[x + "_dof" for x in names]])
    writer.writerows([[i, 100, 0, 80, 0, 0, 90, *([180] * 29)] for i in range(3)])
    raw = text.getvalue().encode()
    q, times = ms.decode_bones(raw, names, expected_frames=3)
    np.testing.assert_allclose(q[:, :3], [[1, 0, .8]] * 3)
    np.testing.assert_allclose(q[:, 7:], np.pi)
    np.testing.assert_allclose(q[:, 3:7], [[0, 0, np.sqrt(.5), np.sqrt(.5)]] * 3, atol=1e-12)
    np.testing.assert_allclose(times, np.arange(3) / 120)
    with pytest.raises(ValueError, match="frame"):
        ms.decode_bones(raw, names, expected_frames=4)
    with pytest.raises(ValueError, match="clock"):
        ms.decode_bones(raw.replace(b"2,100", b"4,100"), names, expected_frames=3)
    with pytest.raises(ValueError, match="columns"):
        ms.decode_bones(raw, list(reversed(names)), expected_frames=3)


def old_tasks():
    return [dict(task_id=f"old_{i}", source="BONES", stem=f"240101/old_{i}",
                 original_clip_id=f"old_clip_{i}", split_group=f"old_group_{i}",
                 split="development" if i < 6 else "confirmation", quality_status="passed")
            for i in range(18)]


def candidates(n=70):
    return [dict(task_id=f"new_{i}", original_clip_id=f"clip_{i}", split_group=f"group_{i}",
                 source="BONES", quality_status="passed") for i in range(n)]


def test_sixty_distinct_with_frozen_old_eighteen_and_reserves():
    old = old_tasks()
    new = candidates()
    new += [dict(new[0], task_id="overlapping_window"),
            dict(new[1], task_id="mirror", original_clip_id="mirror_alias")]
    a = ms.select_tasks(old, new)
    assert a == ms.select_tasks(old, list(reversed(new)))
    assert Counter(t["split"] for t in a["tasks"]) == {"development": 12, "confirmation": 48}
    assert len({t["original_clip_id"] for t in a["tasks"]}) == 60
    assert {t["task_id"] for t in old} <= {t["task_id"] for t in a["tasks"]}
    assert len(a["reserves"]) > 0
    assert not {t["split_group"] for t in a["tasks"]} & {t["split_group"] for t in a["reserves"]}
    assert len({t["split_group"] for t in a["tasks"]}) == 60


def test_old_group_cross_split_fails_closed():
    old = old_tasks()
    old[7]["split_group"] = old[0]["split_group"]
    with pytest.raises(ValueError, match="group"):
        ms.select_tasks(old, candidates())


def test_short_cohort_and_blocked_inputs_never_silently_ready():
    result = ms.select_tasks(old_tasks(), candidates(3))
    assert result["ready"] is False
    assert result["missing"] == 39
    bad = candidates()
    for r in bad:
        r["quality_status"] = "rejected"
    assert ms.select_tasks(old_tasks(), bad)["missing"] == 42


def test_one_prospective_window_and_full_source_timing():
    assert ms.prospective_window(5001, 50, 12) == [44., 56.]
    assert ms.prospective_window(251, 50, 12) is None
    with pytest.raises(ValueError):
        ms.prospective_window(5001, 50, 0)


def test_local_source_config_never_accepts_outcome_roots(tmp_path):
    p = tmp_path / "roots.json"
    p.write_text(json.dumps({"policy_results": "/some/results"}))
    with pytest.raises(ValueError, match="unknown"):
        ms.load_roots(p)


def test_raw_value_comparison_rejects_changed_pose():
    q = np.zeros((121, 36))
    q[:, 6] = 1
    q[:, 0] = np.arange(121) / 120
    a = {"fps": np.array([50.]), "joint_pos": np.zeros((51, 29)),
         "body_pos_w": np.zeros((51, 30, 3)), "body_quat_w": np.zeros((51, 30, 4))}
    a["body_pos_w"][:, 0, 0] = np.arange(51) / 50
    a["body_quat_w"][:, :, 0] = 1
    assert ms.compare_raw(q, np.arange(121) / 120, a)["passed"]
    a["joint_pos"][25, 3] = .1
    assert not ms.compare_raw(q, np.arange(121) / 120, a)["passed"]
    a["joint_pos"][25, 3] = 0
    a["body_pos_w"][20, 0, 0] += .01
    assert not ms.compare_raw(q, np.arange(121) / 120, a)["passed"]


def test_protocol_identity_must_be_preserved():
    from mimicx.datasets.motion_study import protocol_digest
    protocol = {"protocol_id": "test", "fps": 50}
    old = dict(protocol=protocol, protocol_sha256=protocol_digest(protocol), protocol_agreements={"a": 1})
    assert ms.protocol_fields(old) == old
    with pytest.raises(ValueError, match="protocol"):
        ms.protocol_fields(dict(old, protocol_sha256="wrong"))


def test_coverage_does_not_claim_qa_or_policies():
    rows = [row(f"walk_{i}__A001", take=f"walk_{i}") for i in range(10)]
    plan = ms.coverage_plan({r["move_g1_path"]: r for r in rows},
                            {r["move_g1_path"] for r in rows}, target=7)
    assert len(plan["inputs"]) == 7
    assert plan["eligible_inputs"] == 0
    assert plan["trained_mimicx_policies"] == 0
    assert len({r["source_group"] for r in plan["inputs"]}) == 7


def test_coverage_contract_rejects_duplicate_and_unsafe_members():
    rows = [row(f"walk_{i}__A001", take=f"walk_{i}") for i in range(2)]
    plan = ms.coverage_plan({r["move_g1_path"]: r for r in rows},
                            {r["move_g1_path"] for r in rows}, target=2)
    assert len(ms.validate_coverage_plan(plan, expected_count=2)) == 2
    plan["inputs"][1] = dict(plan["inputs"][0])
    with pytest.raises(ValueError, match="distinct"):
        ms.validate_coverage_plan(plan, expected_count=2)
    plan["inputs"][0]["raw_archive_member"] = "../bad.csv"
    with pytest.raises(ValueError):
        ms.validate_coverage_plan(plan, expected_count=2)


def test_coverage_budget_fails_before_writing(tmp_path):
    path = tmp_path / "raw.csv"
    with pytest.raises(ValueError, match="budget"):
        ms.budget_write(path, b"abcd", used=2, limit=5)
    assert not path.exists()
    assert ms.budget_write(path, b"abc", used=2, limit=5) == 5


def test_coverage_one_pass_preserves_missing_denominator(tmp_path):
    rows = [row(f"walk_{i}__A001", take=f"walk_{i}") for i in range(2)]
    plan = ms.coverage_plan({r["move_g1_path"]: r for r in rows},
                            {r["move_g1_path"] for r in rows}, target=2)
    records = ms.validate_coverage_plan(plan, expected_count=2)
    archive = tmp_path / "g1.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo(records[0]["raw_archive_member"])
        member.size = 3
        tar.addfile(member, io.BytesIO(b"bad"))
    output = tmp_path / "out"
    output.mkdir()
    ms.extract_coverage(archive, records, output, limit=1000)
    assert len(records) == 2
    assert records[0]["extraction_status"] == "captured"
    assert records[1]["quality_status"] == "rejected"
    assert "missing" in records[1]["reasons"][0]
    assert (output / records[0]["source_path"]).read_bytes() == b"bad"


def test_coverage_cpu_conversion_and_full_denominator(tmp_path, monkeypatch):
    pytest.importorskip("mujoco")
    bodies = []
    for i in range(29):
        name = {5: "left_ankle_roll_link", 11: "right_ankle_roll_link"}.get(i, f"link_{i}")
        bodies.append(f'<body name="{name}" pos="0 0 .05"><joint name="j{i}" range="-1 1"/><geom type="sphere" size=".01" mass=".1"/>')
    xml = tmp_path / "robot.xml"
    xml.write_text('<mujoco><compiler angle="radian"/><worldbody><body name="pelvis"><freejoint/><geom type="sphere" size=".05" mass="1"/>' + ''.join(bodies) + '</body>' * 30 + '</worldbody></mujoco>')
    robot = ms.CanonicalG1(xml)
    raw = io.StringIO()
    writer = csv.writer(raw)
    writer.writerow(["Frame", *["root_translate" + x for x in "XYZ"],
                     *["root_rotate" + x for x in "XYZ"], *[n + "_dof" for n in robot.joint_names]])
    writer.writerows([[i, 0, 0, 80, 0, 0, 0, *([0] * 29)] for i in range(121)])
    payload = raw.getvalue().encode()
    record = dict(source_metadata={"move_duration_frames": "121"})
    arrays, audit = ms.canonicalize_coverage(payload, record, robot)
    assert audit["quality_status"] == "passed"
    assert audit["frames"] == 51 and audit["duration"] == 1.
    assert audit["canonical_audit"]["fk_passed"]
    assert arrays["body_pos_w"].shape == (51, 30, 3)
    bad = payload.replace(b"0,0,0,80", b"0,nan,0,80", 1)
    with pytest.raises(ValueError, match="nonfinite"):
        ms.canonicalize_coverage(bad, record, robot)
    rows = [dict(row(f"walk_{i}__A001", take=f"walk_{i}"), move_duration_frames="121") for i in range(1000)]
    plan = ms.coverage_plan({r["move_g1_path"]: r for r in rows}, {r["move_g1_path"] for r in rows})
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    archive = tmp_path / "archive.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo(plan["inputs"][0]["raw_archive_member"])
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))
    inv = tmp_path / "inventory.json"
    stat = archive.stat()
    inv.write_text(json.dumps({"raw_archive": {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}}))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    result = ms.prepare_coverage({"bones_archive": archive, "g1_xml": xml}, plan_path, inv,
                                 tmp_path / "out", disk_budget=70_000_000)
    assert result["complete"] and result["denominator"] == 1000
    assert result["counts"] == {"passed": 1, "rejected": 999}
    assert result["evaluated_inputs"] == result["trained_mimicx_policies"] == 0
    assert len(result["tasks"]) == 1000
