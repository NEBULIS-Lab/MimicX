from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mimicx.evaluation.paper_matrix import (
    aggregate_method_task,
    collect_matrix_rows,
    write_matrix_report,
)


def _write_rollout(
    directory: Path,
    *,
    reward: float,
    done_steps: tuple[int, ...],
    body_errors: tuple[float, ...],
) -> Path:
    directory.mkdir(parents=True)
    metrics_path = directory / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "steps": len(body_errors),
                "reward_mean_avg": reward,
                "done_any_count": len(done_steps),
            }
        ),
        encoding="utf-8",
    )
    fieldnames = [
        "step",
        "done_any",
        "termination_anchor_pos",
        "rew_mean",
        "body_pos_error_max",
        "ee_z_error_max",
        "error_anchor_pos",
        "error_body_lin_vel",
        "error_body_ang_vel",
    ]
    with (directory / "metrics_steps.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for step, body_error in enumerate(body_errors, start=1):
            failed = step in done_steps
            writer.writerow(
                {
                    "step": step,
                    "done_any": failed,
                    "termination_anchor_pos": failed,
                    "rew_mean": reward,
                    "body_pos_error_max": body_error,
                    "ee_z_error_max": body_error / 2,
                    "error_anchor_pos": body_error / 4,
                    "error_body_lin_vel": body_error * 2,
                    "error_body_ang_vel": body_error * 3,
                }
            )
    return metrics_path


def _write_job(
    jobs_dir: Path,
    *,
    job_id: str,
    task_id: str,
    method_id: str,
    seed: int,
    complete: bool,
) -> Path:
    job_dir = jobs_dir / job_id
    job_dir.mkdir(parents=True)
    job = {
        "id": job_id,
        "task_id": task_id,
        "method_id": method_id,
        "mode": "static",
        "seed": seed,
        "gpu_id": 3,
        "horizon": 5,
        "verification_seeds": [11, 22, 33],
    }
    (job_dir / "job.json").write_text(json.dumps(job), encoding="utf-8")
    if complete:
        metrics = [
            _write_rollout(
                job_dir / "eval" / "seed_11",
                reward=0.4,
                done_steps=(),
                body_errors=(0.1, 0.2, 0.3, 0.4, 0.5),
            ),
            _write_rollout(
                job_dir / "eval" / "seed_22",
                reward=0.5,
                done_steps=(4,),
                body_errors=(0.2, 0.3, 0.4, 0.5, 0.6),
            ),
            _write_rollout(
                job_dir / "eval" / "seed_33",
                reward=0.6,
                done_steps=(),
                body_errors=(0.3, 0.4, 0.5, 0.6, 0.7),
            ),
        ]
        (job_dir / "job.done.json").write_text(
            json.dumps({"job": job_id, "checkpoint": "/tmp/model.pt", "metrics": [str(path) for path in metrics]}),
            encoding="utf-8",
        )
    return job_dir


def test_collect_matrix_rows_preserves_incomplete_jobs_and_strict_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "matrix"
    jobs_dir = run_dir / "jobs"
    _write_job(
        jobs_dir,
        job_id="tennis__m1__seed101",
        task_id="tennis",
        method_id="m1",
        seed=101,
        complete=True,
    )
    _write_job(
        jobs_dir,
        job_id="tennis__m1__seed202",
        task_id="tennis",
        method_id="m1",
        seed=202,
        complete=False,
    )

    rows = collect_matrix_rows(run_dir)

    assert len(rows) == 2
    completed = next(row for row in rows if row.status == "completed")
    assert completed.repeats == 3
    assert completed.zero_termination_repeats == 2
    assert completed.total_terminations == 1
    assert completed.worst_first_failure_step == 4
    assert completed.reward_mean == pytest.approx(0.5)
    assert completed.body_pos_error_mean == pytest.approx(0.4)
    assert completed.body_pos_error_p95 == pytest.approx(0.7)
    assert completed.body_pos_error_peak == pytest.approx(0.7)
    assert completed.ee_z_error_mean == pytest.approx(0.2)
    assert completed.anchor_pos_error_mean == pytest.approx(0.1)
    assert completed.body_lin_vel_error_mean == pytest.approx(0.8)
    assert completed.body_ang_vel_error_mean == pytest.approx(1.2)

    pending = next(row for row in rows if row.status == "pending")
    assert pending.repeats == 0
    assert pending.reward_mean is None

    summary = aggregate_method_task(rows, expected_seeds=2)
    assert summary[0].completed_seeds == 1
    assert summary[0].expected_seeds == 2
    assert summary[0].strict_success_rate == pytest.approx(2 / 3)


def test_write_matrix_report_emits_machine_and_human_readable_views(tmp_path: Path) -> None:
    run_dir = tmp_path / "matrix"
    _write_job(
        run_dir / "jobs",
        job_id="dance2__m0__seed101",
        task_id="dance2",
        method_id="m0",
        seed=101,
        complete=True,
    )

    output_dir = tmp_path / "report"
    status = write_matrix_report(run_dir, output_dir, expected_seeds=3)

    assert status["total_jobs"] == 1
    assert status["completed_jobs"] == 1
    assert (output_dir / "job_results.csv").is_file()
    assert (output_dir / "method_task_summary.csv").is_file()
    assert json.loads((output_dir / "status.json").read_text())["pending_jobs"] == 0
    markdown = (output_dir / "STATUS.md").read_text(encoding="utf-8")
    assert "dance2" in markdown
    assert "m0" in markdown


def test_completed_job_with_missing_metrics_is_rejected(tmp_path: Path) -> None:
    run_dir = tmp_path / "matrix"
    job_dir = _write_job(
        run_dir / "jobs",
        job_id="football1__m2__seed101",
        task_id="football1",
        method_id="m2",
        seed=101,
        complete=False,
    )
    (job_dir / "job.done.json").write_text(
        json.dumps({"metrics": [str(tmp_path / "missing.json")]}),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="missing.json"):
        collect_matrix_rows(run_dir)


def test_closed_loop_job_uses_final_selected_strict_records(tmp_path: Path) -> None:
    run_dir = tmp_path / "matrix"
    job_dir = _write_job(
        run_dir / "jobs",
        job_id="kongfu1__m3__seed101",
        task_id="kongfu1",
        method_id="m3",
        seed=101,
        complete=False,
    )
    job_path = job_dir / "job.json"
    job = json.loads(job_path.read_text())
    job["mode"] = "closed_loop"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    selected_metrics = _write_rollout(
        job_dir / "closed_loop" / "strict" / "seed_11",
        reward=0.7,
        done_steps=(),
        body_errors=(0.1, 0.1, 0.1, 0.1, 0.1),
    )
    iteration_dir = job_dir / "closed_loop" / "iterations" / "iter_000"
    iteration_dir.mkdir(parents=True)
    (iteration_dir / "selection.json").write_text(
        json.dumps(
            {
                "incumbent": {"candidate_id": "base", "records": []},
                "candidates": [
                    {
                        "candidate_id": "accepted_patch",
                        "records": [{"source": str(selected_metrics)}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "closed_loop" / "state.json").write_text(
        json.dumps(
            {
                "incumbent": {"id": "accepted_patch", "checkpoint": "/tmp/accepted.pt"},
                "history": [{"iteration": 0, "selected_id": "accepted_patch"}],
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "job.done.json").write_text(
        json.dumps({"checkpoint": "/tmp/accepted.pt", "summary": {"status": "completed"}}),
        encoding="utf-8",
    )

    rows = collect_matrix_rows(run_dir)

    assert len(rows) == 1
    assert rows[0].status == "completed"
    assert rows[0].repeats == 1
    assert rows[0].zero_termination_repeats == 1
    assert rows[0].reward_mean == pytest.approx(0.7)
