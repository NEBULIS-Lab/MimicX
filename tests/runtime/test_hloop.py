from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from mimicx.runtime.hloop import HLoopPlan, compare_hloop_runs, run_hloop_plan, sha256_file


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "scripts/testing/deterministic_hloop_fixture.py"


def _plan(tmp_path: Path) -> HLoopPlan:
    source = tmp_path / "input.json"
    source.write_text('{"fixture":true}\n', encoding="utf-8")
    common = {
        "resource": "cpu",
        "cwd": str(ROOT),
        "inputs": [
            {"role": "source", "path": str(source), "sha256": sha256_file(source)}
        ],
    }
    jobs = []
    for candidate_id, resource, score, device in (
        ("candidate_a", "cpu", "0.6", None),
        ("candidate_b", "gpu", "0.8", "2"),
    ):
        output = f"{{run_dir}}/candidates/{candidate_id}.json"
        job = {
            **common,
            "job_id": candidate_id,
            "resource": resource,
            "candidate_id": candidate_id,
            "seed": 3,
            "command": [
                sys.executable,
                str(FIXTURE),
                "candidate",
                "--candidate-id",
                candidate_id,
                "--seed",
                "3",
                "--score",
                score,
                "--input",
                str(source),
                "--output",
                output,
                "--sleep",
                "0.01",
            ],
            "outputs": [{"role": "candidate", "path": output}],
        }
        if device is not None:
            job["cuda_visible_devices"] = device
        jobs.append(job)
    jobs.append(
        {
            "job_id": "select",
            "resource": "io",
            "depends_on": ["candidate_a", "candidate_b"],
            "cwd": str(ROOT),
            "command": [
                sys.executable,
                str(FIXTURE),
                "select",
                "--candidate",
                "{run_dir}/candidates/candidate_a.json",
                "--candidate",
                "{run_dir}/candidates/candidate_b.json",
                "--output",
                "{run_dir}/selection.json",
            ],
            "inputs": [
                {"role": "candidate", "path": "{run_dir}/candidates/candidate_a.json"},
                {"role": "candidate", "path": "{run_dir}/candidates/candidate_b.json"},
            ],
            "outputs": [{"role": "selection", "path": "{run_dir}/selection.json"}],
        }
    )
    return HLoopPlan.from_mapping(
        {
            "schema_version": "mimicx.hloop-plan.v1",
            "plan_id": "fixture",
            "max_cpu_workers": 2,
            "max_io_workers": 1,
            "selection_artifact": "{run_dir}/selection.json",
            "jobs": jobs,
        }
    )


def test_e0_has_exact_candidate_and_selection_parity(tmp_path: Path) -> None:
    plan = _plan(tmp_path)

    sequential = run_hloop_plan(
        plan,
        run_dir=tmp_path / "sequential",
        repo_root=ROOT,
        mode="sequential",
    )
    e0 = run_hloop_plan(
        plan,
        run_dir=tmp_path / "e0",
        repo_root=ROOT,
        mode="e0",
    )
    parity = compare_hloop_runs(sequential, e0)

    assert parity["passed"] is True
    assert parity["speed"]["claimable"] is False
    assert e0["selection"]["value"]["selected_id"] == "candidate_b"
    gpu_candidate = json.loads(
        (tmp_path / "e0/candidates/candidate_b.json").read_text(encoding="utf-8")
    )
    assert gpu_candidate["cuda_visible_devices"] == "2"


def test_hloop_plan_rejects_dependency_cycle() -> None:
    with pytest.raises(ValueError, match="dependency cycle"):
        HLoopPlan.from_mapping(
            {
                "schema_version": "mimicx.hloop-plan.v1",
                "plan_id": "cycle",
                "jobs": [
                    {
                        "job_id": "a",
                        "resource": "cpu",
                        "command": ["true"],
                        "depends_on": ["b"],
                    },
                    {
                        "job_id": "b",
                        "resource": "cpu",
                        "command": ["true"],
                        "depends_on": ["a"],
                    },
                ],
            }
        )


def test_parity_detects_tampered_output_hash(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    sequential = run_hloop_plan(
        plan,
        run_dir=tmp_path / "sequential",
        repo_root=ROOT,
        mode="sequential",
    )
    e0 = run_hloop_plan(
        plan,
        run_dir=tmp_path / "e0",
        repo_root=ROOT,
        mode="e0",
    )
    e0["jobs"][0]["outputs"][0]["sha256"] = "0" * 64

    parity = compare_hloop_runs(sequential, e0)

    assert parity["passed"] is False
    assert any(
        check["name"] == "candidate_a.output_hashes" and not check["passed"]
        for check in parity["checks"]
    )
