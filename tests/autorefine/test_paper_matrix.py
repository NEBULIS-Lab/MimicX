from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from scripts.autorefine.launch_paper_matrix import (
    _discover_static_checkpoint,
    _runtime_gpu_id,
    expand_jobs,
    load_matrix,
    preflight_checkpoints,
    prepare_runpack,
    run_job,
)


def write_matrix_fixture(tmp_path: Path) -> Path:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    train = runtime / "train.py"
    rollout = runtime / "rollout.py"
    python_target = tmp_path / "python-real"
    python = tmp_path / "python"
    for path in (train, rollout, python_target):
        path.write_text(path.name, encoding="utf-8")
    python.symlink_to(python_target)

    task_paths = []
    for task_id in ("task-a", "task-b"):
        motion = tmp_path / f"{task_id}.npz"
        refined = tmp_path / f"{task_id}-refined.npz"
        checkpoint = tmp_path / f"{task_id}.pt"
        for path in (motion, refined, checkpoint):
            path.write_text(path.name, encoding="utf-8")
        task_path = tmp_path / f"{task_id}.yaml"
        task_path.write_text(
            yaml.safe_dump(
                {
                    "id": task_id,
                    "motion_file": str(motion),
                    "refined_motion_file": str(refined),
                    "base_checkpoint": str(checkpoint),
                    "policy_observation_dim": 160,
                    "load_run": f"{task_id}-base",
                    "horizon": 100,
                    "failure_window": [40, 60],
                    "priority_bodies": ["pelvis", "left_ankle_roll_link"],
                    "learning_rate": 1.0e-6,
                }
            ),
            encoding="utf-8",
        )
        task_paths.append(str(task_path))

    method_paths = []
    for method_id, mode in (("m0", "static"), ("m3", "closed_loop")):
        method_path = tmp_path / f"{method_id}.yaml"
        method_path.write_text(
            yaml.safe_dump(
                {
                    "id": method_id,
                    "mode": mode,
                    "patch_profile": "none" if method_id == "m0" else "autorefine",
                    "use_refined_reference": method_id == "m3",
                }
            ),
            encoding="utf-8",
        )
        method_paths.append(str(method_path))

    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(
        yaml.safe_dump(
            {
                "schema": "mimicx.paper-matrix.v1",
                "runtime": {
                    "workdir": str(runtime),
                    "python": str(python),
                    "train_script": str(train),
                    "rollout_script": str(rollout),
                    "checkpoint_root": str(tmp_path / "checkpoints"),
                    "static_task": "Fixture-Static",
                    "dynamic_task": "Fixture-Dynamic",
                    "strict_task": "Fixture-Strict",
                },
                "tasks": task_paths,
                "methods": method_paths,
                "seeds": [101, 202, 303],
                "training": {"iterations": 25, "num_envs": 8},
                "verification": {"repeats": 3, "seeds": [11, 22, 33]},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return matrix_path


def test_expand_jobs_forms_full_task_method_seed_cross_product(tmp_path: Path) -> None:
    matrix = load_matrix(write_matrix_fixture(tmp_path))

    jobs = expand_jobs(matrix, gpu_ids=(3, 4))

    assert len(jobs) == 12
    assert matrix.runtime["python"].endswith("/python")
    assert [job.gpu_id for job in jobs] == [3, 4] * 6
    assert len({job.id for job in jobs}) == 12
    assert {job.mode for job in jobs} == {"static", "closed_loop"}
    assert all(len(job.motion_sha256) == 64 for job in jobs)
    assert all(len(job.checkpoint_sha256) == 64 for job in jobs)


def test_prepare_runpack_writes_flat_job_ledger_and_gpu_queues(tmp_path: Path) -> None:
    matrix = load_matrix(write_matrix_fixture(tmp_path))
    run_dir = tmp_path / "paper-matrix-run"

    jobs = prepare_runpack(matrix, run_dir=run_dir, gpu_ids=(3, 4))

    ledger = [json.loads(line) for line in (run_dir / "jobs.jsonl").read_text().splitlines()]
    assert len(ledger) == len(jobs) == 12
    assert (run_dir / "resolved_matrix.json").is_file()
    assert (run_dir / "queue_gpu3.sh").is_file()
    assert (run_dir / "queue_gpu4.sh").is_file()
    assert "MIMICX_DEVICE_ID=3" in (run_dir / "queue_gpu3.sh").read_text()
    assert "MIMICX_DEVICE_ID=4" in (run_dir / "queue_gpu4.sh").read_text()
    queue = (run_dir / "queue_gpu3.sh").read_text()
    assert "set -e" not in queue
    assert "queue_failures.log" in queue
    assert "run_job" in queue
    assert len(list((run_dir / "jobs").glob("*/job.json"))) == 12
    assert len(list((run_dir / "jobs").glob("*/loop_manifest.yaml"))) == 6


def test_preflight_checkpoints_rejects_incompatible_policy_input(tmp_path: Path) -> None:
    matrix = load_matrix(write_matrix_fixture(tmp_path))

    def inspect(_python: Path, checkpoint: Path) -> tuple[int, int]:
        return (154, 29) if checkpoint.name == "task-a.pt" else (160, 29)

    try:
        preflight_checkpoints(matrix, inspector=inspect)
    except ValueError as error:
        message = str(error)
    else:
        raise AssertionError("incompatible checkpoint was accepted")

    assert "task-a" in message
    assert "expected 160" in message
    assert "observed 154" in message


def test_preflight_checkpoints_accepts_matching_policy_inputs(tmp_path: Path) -> None:
    matrix = load_matrix(write_matrix_fixture(tmp_path))

    records = preflight_checkpoints(
        matrix,
        inspector=lambda _python, _checkpoint: (160, 29),
    )

    assert [record["task_id"] for record in records] == ["task-a", "task-b"]
    assert all(record["actor_input_dim"] == 160 for record in records)
    assert all(record["actor_output_dim"] == 29 for record in records)


def test_static_job_resumes_evaluation_from_completed_training_stage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    checkpoint = tmp_path / "model_1248.pt"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    (job_dir / "train.done.json").write_text(
        json.dumps({"checkpoint": str(checkpoint)}),
        encoding="utf-8",
    )
    job = {
        "id": "task__m0__seed101",
        "job_dir": str(job_dir),
        "mode": "static",
        "gpu_id": 3,
        "runtime": {
            "workdir": str(tmp_path),
            "checkpoint_root": str(tmp_path),
            "strict_task": "Fixture-Strict",
            "env": {},
        },
        "verification_seeds": [],
    }
    job_path = job_dir / "job.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")

    def unexpected_training(*_args, **_kwargs):
        raise AssertionError("completed training stage was launched again")

    monkeypatch.setattr(
        "scripts.autorefine.launch_paper_matrix.subprocess.run",
        unexpected_training,
    )

    assert run_job(job_path) == 0
    done = json.loads((job_dir / "job.done.json").read_text())
    assert done["checkpoint"] == str(checkpoint)


def test_static_checkpoint_discovery_prefers_highest_training_step(
    tmp_path: Path,
) -> None:
    run_name = "mimicx_fixture_seed101_m0"
    complete = tmp_path / f"2026-08-15_06-00-00_{run_name}" / "model_1248.pt"
    interrupted = tmp_path / f"2026-08-15_07-00-00_{run_name}" / "model_1000.pt"
    complete.parent.mkdir()
    complete.write_bytes(b"complete")
    interrupted.parent.mkdir()
    interrupted.write_bytes(b"newer but incomplete")
    os.utime(complete, ns=(1_000_000_000, 1_000_000_000))
    os.utime(interrupted, ns=(2_000_000_000, 2_000_000_000))

    selected = _discover_static_checkpoint(
        {
            "id": "fixture",
            "run_name": run_name,
            "runtime": {"checkpoint_root": str(tmp_path)},
        }
    )

    assert selected == complete.resolve()


def test_runtime_gpu_override_changes_placement_only(monkeypatch) -> None:
    job = {"gpu_id": 4}

    assert _runtime_gpu_id(job) == 4

    monkeypatch.setenv("MIMICX_RUNTIME_GPU_ID", "7")
    assert _runtime_gpu_id(job) == 7
