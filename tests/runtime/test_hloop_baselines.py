from __future__ import annotations

from pathlib import Path
import sys

from mimicx.runtime.hloop import HLoopPlan, compare_hloop_runs, run_hloop_plan
from mimicx.runtime.hloop_baselines import run_hloop_plan_bulk_sync


ROOT = Path(__file__).resolve().parents[2]


def _plan() -> HLoopPlan:
    jobs = []
    for job_id, device, dependencies in (
        ("gpu_a0", "3", []),
        ("gpu_a1", "3", []),
        ("gpu_b0", "4", []),
        ("gpu_b1", "4", ["gpu_b0"]),
    ):
        output = f"{{run_dir}}/outputs/{job_id}.txt"
        jobs.append(
            {
                "job_id": job_id,
                "resource": "gpu",
                "cuda_visible_devices": device,
                "depends_on": dependencies,
                "cwd": str(ROOT),
                "command": [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; import sys; "
                        "p=Path(sys.argv[1]); p.parent.mkdir(parents=True, exist_ok=True); "
                        "p.write_text(sys.argv[2] + '\\n', encoding='utf-8')"
                    ),
                    output,
                    job_id,
                ],
                "outputs": [{"role": "fixture", "path": output}],
                "candidate_id": job_id,
            }
        )
    return HLoopPlan.from_mapping(
        {
            "schema_version": "mimicx.hloop-plan.v1",
            "plan_id": "bulk-sync-fixture",
            "max_cpu_workers": 2,
            "max_io_workers": 1,
            "jobs": jobs,
        }
    )


def test_bulk_sync_respects_gpu_exclusivity_and_dependencies(
    tmp_path: Path,
) -> None:
    report = run_hloop_plan_bulk_sync(
        _plan(),
        run_dir=tmp_path / "bulk",
        repo_root=ROOT,
    )

    assert report["waves"] == [
        ["gpu_a0", "gpu_b0"],
        ["gpu_a1", "gpu_b1"],
    ]
    assert all(job["returncode"] == 0 for job in report["jobs"])


def test_bulk_sync_has_exact_e0_output_parity(tmp_path: Path) -> None:
    plan = _plan()
    bulk = run_hloop_plan_bulk_sync(
        plan,
        run_dir=tmp_path / "bulk",
        repo_root=ROOT,
    )
    e0 = run_hloop_plan(
        plan,
        run_dir=tmp_path / "e0",
        repo_root=ROOT,
        mode="e0",
    )

    assert compare_hloop_runs(bulk, e0)["passed"] is True
