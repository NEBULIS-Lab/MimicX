"""Controlled baseline executors for MimicX-HLoop experiments."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Mapping

from mimicx.runtime.hloop import (
    HLoopJob,
    HLoopPlan,
    _execute_job,
    _selection_record,
    _topological_ids,
)


def run_hloop_plan_bulk_sync(
    plan: HLoopPlan,
    *,
    run_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Run ready jobs in waves, inserting a barrier after every wave.

    This is the naive-parallel control for HLoop. It uses the same resource
    limits and immutable jobs, but it cannot overlap downstream CPU/IO work
    with unfinished GPU jobs from the current wave.
    """

    plan.validate()
    output = run_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"HLoop run directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    context = {
        "repo_root": str(repo_root.expanduser().resolve()),
        "run_dir": str(output),
    }
    started = time.monotonic()
    started_at = _now()
    results, waves = _run_bulk_sync(plan, output=output, context=context)
    report = {
        "schema_version": "mimicx.hloop-run.v1",
        "plan_id": plan.plan_id,
        "plan_semantic_hash": plan.semantic_hash(),
        "mode": "bulk_sync",
        "started_at": started_at,
        "completed_at": _now(),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "status": "completed",
        "candidate_ids": sorted(
            job.candidate_id
            for job in plan.jobs
            if job.candidate_id is not None
        ),
        "jobs": results,
        "waves": waves,
        "selection": _selection_record(plan, context=context),
    }
    (output / "hloop_run.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _run_bulk_sync(
    plan: HLoopPlan,
    *,
    output: Path,
    context: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[list[str]]]:
    jobs = {job.job_id: job for job in plan.jobs}
    order = _topological_ids(plan.jobs)
    pending = set(order)
    completed: set[str] = set()
    results: dict[str, dict[str, Any]] = {}
    waves: list[list[str]] = []
    while pending:
        ready = [
            jobs[job_id]
            for job_id in order
            if job_id in pending
            and set(jobs[job_id].depends_on).issubset(completed)
        ]
        if not ready:
            raise RuntimeError("bulk-synchronous scheduler is deadlocked")
        wave = _select_wave(plan, ready)
        if not wave:
            raise RuntimeError("bulk-synchronous scheduler selected an empty wave")
        waves.append([job.job_id for job in wave])
        with ThreadPoolExecutor(max_workers=len(wave)) as executor:
            futures = {
                job.job_id: executor.submit(
                    _execute_job,
                    job,
                    output=output,
                    context=context,
                )
                for job in wave
            }
            for job in wave:
                results[job.job_id] = futures[job.job_id].result()
        for job in wave:
            pending.remove(job.job_id)
            completed.add(job.job_id)
    return [results[job_id] for job_id in order], waves


def _select_wave(plan: HLoopPlan, ready: list[HLoopJob]) -> list[HLoopJob]:
    selected: list[HLoopJob] = []
    cpu_count = 0
    io_count = 0
    gpus: set[str] = set()
    for job in ready:
        if job.resource == "cpu":
            if cpu_count >= plan.max_cpu_workers:
                continue
            cpu_count += 1
        elif job.resource == "io":
            if io_count >= plan.max_io_workers:
                continue
            io_count += 1
        else:
            assert job.cuda_visible_devices is not None
            if job.cuda_visible_devices in gpus:
                continue
            gpus.add(job.cuda_visible_devices)
        selected.append(job)
    return selected


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
