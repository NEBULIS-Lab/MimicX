"""Aggregate strict paper-matrix rollouts into reproducible result tables."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence


STEP_METRICS = {
    "body_pos_error": "body_pos_error_max",
    "ee_z_error": "ee_z_error_max",
    "anchor_pos_error": "error_anchor_pos",
    "body_lin_vel_error": "error_body_lin_vel",
    "body_ang_vel_error": "error_body_ang_vel",
}


@dataclass(frozen=True)
class MatrixResultRow:
    job_id: str
    task_id: str
    method_id: str
    train_seed: int
    gpu_id: int
    status: str
    checkpoint: str | None = None
    repeats: int = 0
    zero_termination_repeats: int = 0
    total_terminations: int = 0
    worst_first_failure_step: int | None = None
    reward_mean: float | None = None
    body_pos_error_mean: float | None = None
    body_pos_error_p95: float | None = None
    body_pos_error_peak: float | None = None
    ee_z_error_mean: float | None = None
    ee_z_error_p95: float | None = None
    ee_z_error_peak: float | None = None
    anchor_pos_error_mean: float | None = None
    anchor_pos_error_p95: float | None = None
    anchor_pos_error_peak: float | None = None
    body_lin_vel_error_mean: float | None = None
    body_lin_vel_error_p95: float | None = None
    body_lin_vel_error_peak: float | None = None
    body_ang_vel_error_mean: float | None = None
    body_ang_vel_error_p95: float | None = None
    body_ang_vel_error_peak: float | None = None


@dataclass(frozen=True)
class MethodTaskSummary:
    task_id: str
    method_id: str
    completed_seeds: int
    expected_seeds: int
    rollout_repeats: int
    strict_success_rate: float | None
    reward_mean: float | None
    reward_std: float | None
    worst_first_failure_step_mean: float | None
    body_pos_error_mean: float | None
    body_pos_error_p95_mean: float | None
    ee_z_error_mean: float | None
    anchor_pos_error_mean: float | None


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _step_path(metrics_path: Path) -> Path:
    return metrics_path.with_name(f"{metrics_path.stem}_steps.csv")


def _nearest_rank(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def _mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return fmean(present) if present else None


def _population_std(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    center = fmean(present)
    return math.sqrt(fmean((value - center) ** 2 for value in present))


def _selected_closed_loop_metrics(job_dir: Path) -> list[Path]:
    state_path = job_dir / "closed_loop" / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Completed closed-loop job lacks state: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    history = state.get("history", [])
    if not history:
        raise ValueError(f"Completed closed-loop job has no selection history: {state_path}")
    iteration = int(history[-1]["iteration"])
    selected_id = str(state["incumbent"]["id"])
    selection_path = (
        job_dir / "closed_loop" / "iterations" / f"iter_{iteration:03d}" / "selection.json"
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    aggregates = [selection.get("incumbent", {})] + list(selection.get("candidates", []))
    selected = next(
        (item for item in aggregates if item.get("candidate_id") == selected_id),
        None,
    )
    if selected is None:
        raise ValueError(f"Selected policy {selected_id!r} is absent from {selection_path}")
    paths = [Path(record["source"]).expanduser().resolve() for record in selected.get("records", [])]
    if not paths:
        raise ValueError(f"Selected policy {selected_id!r} has no strict rollout records")
    return paths


def _metric_paths(job: Mapping[str, Any], job_dir: Path, done: Mapping[str, Any]) -> list[Path]:
    if job.get("mode") == "closed_loop":
        paths = _selected_closed_loop_metrics(job_dir)
    else:
        paths = [Path(value).expanduser().resolve() for value in done.get("metrics", [])]
    if not paths:
        raise ValueError(f"Completed job has no strict metrics: {job_dir}")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Strict metrics file does not exist: {path}")
        step_path = _step_path(path)
        if not step_path.is_file():
            raise FileNotFoundError(f"Strict step metrics file does not exist: {step_path}")
    return paths


def _completed_row(job: Mapping[str, Any], job_dir: Path, done: Mapping[str, Any]) -> MatrixResultRow:
    horizon = int(job["horizon"])
    reward_values: list[float] = []
    done_counts: list[int] = []
    first_failures: list[int] = []
    step_values: dict[str, list[float]] = {name: [] for name in STEP_METRICS}
    metrics_paths = _metric_paths(job, job_dir, done)
    for metrics_path in metrics_paths:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        done_count = int(metrics.get("done_any_count", metrics.get("done_count", 0)) or 0)
        done_counts.append(done_count)
        reward = _number(metrics.get("reward_mean_avg"))
        if reward is None:
            reward = _number(metrics.get("final_rew_mean"))
        if reward is not None:
            reward_values.append(reward)
        first_failure: int | None = None
        with _step_path(metrics_path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                failed = _truthy(row.get("done_any")) or any(
                    key.startswith("termination_") and _truthy(value)
                    for key, value in row.items()
                )
                if first_failure is None and failed:
                    first_failure = int(row["step"])
                for output_name, column in STEP_METRICS.items():
                    value = _number(row.get(column))
                    if value is not None:
                        step_values[output_name].append(value)
        first_failures.append(horizon + 1 if done_count == 0 else (first_failure or horizon))

    statistics: dict[str, float | None] = {}
    for name, values in step_values.items():
        statistics[f"{name}_mean"] = _mean(values)
        statistics[f"{name}_p95"] = _nearest_rank(values, 0.95)
        statistics[f"{name}_peak"] = max(values) if values else None
    return MatrixResultRow(
        job_id=str(job["id"]),
        task_id=str(job["task_id"]),
        method_id=str(job["method_id"]),
        train_seed=int(job["seed"]),
        gpu_id=int(job["gpu_id"]),
        status="completed",
        checkpoint=str(done.get("checkpoint")) if done.get("checkpoint") else None,
        repeats=len(metrics_paths),
        zero_termination_repeats=sum(count == 0 for count in done_counts),
        total_terminations=sum(done_counts),
        worst_first_failure_step=min(first_failures),
        reward_mean=_mean(reward_values),
        **statistics,
    )


def collect_matrix_rows(run_dir: Path) -> list[MatrixResultRow]:
    root = run_dir.expanduser().resolve()
    job_paths = sorted((root / "jobs").glob("*/job.json"))
    if not job_paths:
        raise FileNotFoundError(f"No matrix jobs found under {root / 'jobs'}")
    rows: list[MatrixResultRow] = []
    for job_path in job_paths:
        job_dir = job_path.parent
        job = json.loads(job_path.read_text(encoding="utf-8"))
        done_path = job_dir / "job.done.json"
        if done_path.is_file():
            done = json.loads(done_path.read_text(encoding="utf-8"))
            rows.append(_completed_row(job, job_dir, done))
        else:
            rows.append(
                MatrixResultRow(
                    job_id=str(job["id"]),
                    task_id=str(job["task_id"]),
                    method_id=str(job["method_id"]),
                    train_seed=int(job["seed"]),
                    gpu_id=int(job["gpu_id"]),
                    status="pending",
                )
            )
    return rows


def aggregate_method_task(
    rows: Sequence[MatrixResultRow],
    *,
    expected_seeds: int,
) -> list[MethodTaskSummary]:
    grouped: dict[tuple[str, str], list[MatrixResultRow]] = {}
    for row in rows:
        grouped.setdefault((row.task_id, row.method_id), []).append(row)
    summaries: list[MethodTaskSummary] = []
    for (task_id, method_id), group in sorted(grouped.items()):
        completed = [row for row in group if row.status == "completed"]
        rollout_repeats = sum(row.repeats for row in completed)
        successful_repeats = sum(row.zero_termination_repeats for row in completed)
        summaries.append(
            MethodTaskSummary(
                task_id=task_id,
                method_id=method_id,
                completed_seeds=len(completed),
                expected_seeds=expected_seeds,
                rollout_repeats=rollout_repeats,
                strict_success_rate=(
                    successful_repeats / rollout_repeats if rollout_repeats else None
                ),
                reward_mean=_mean(row.reward_mean for row in completed),
                reward_std=_population_std(row.reward_mean for row in completed),
                worst_first_failure_step_mean=_mean(
                    float(row.worst_first_failure_step)
                    if row.worst_first_failure_step is not None
                    else None
                    for row in completed
                ),
                body_pos_error_mean=_mean(row.body_pos_error_mean for row in completed),
                body_pos_error_p95_mean=_mean(row.body_pos_error_p95 for row in completed),
                ee_z_error_mean=_mean(row.ee_z_error_mean for row in completed),
                anchor_pos_error_mean=_mean(row.anchor_pos_error_mean for row in completed),
            )
        )
    return summaries


def _write_csv(path: Path, records: Sequence[object]) -> None:
    rows = [asdict(record) for record in records]
    if not rows:
        raise ValueError(f"Cannot write empty result table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def write_matrix_report(
    run_dir: Path,
    output_dir: Path,
    *,
    expected_seeds: int | None = None,
) -> dict[str, Any]:
    rows = collect_matrix_rows(run_dir)
    if expected_seeds is None:
        resolved = Path(run_dir) / "resolved_matrix.json"
        if resolved.is_file():
            expected_seeds = len(json.loads(resolved.read_text(encoding="utf-8")).get("seeds", []))
        if not expected_seeds:
            expected_seeds = max(len({row.train_seed for row in rows}), 1)
    summaries = aggregate_method_task(rows, expected_seeds=expected_seeds)
    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "job_results.csv", rows)
    _write_csv(output / "method_task_summary.csv", summaries)
    completed = sum(row.status == "completed" for row in rows)
    status = {
        "run_dir": str(Path(run_dir).expanduser().resolve()),
        "total_jobs": len(rows),
        "completed_jobs": completed,
        "pending_jobs": len(rows) - completed,
        "expected_train_seeds": expected_seeds,
    }
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Paper Matrix Status",
        "",
        f"- Completed jobs: **{completed}/{len(rows)}**",
        f"- Expected train seeds per task-method pair: **{expected_seeds}**",
        "",
        "| Task | Method | Seeds | Strict success | Reward | Body error | EE-z error |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            "| {task} | {method} | {done}/{expected} | {success} | {reward} | {body} | {ee} |".format(
                task=summary.task_id,
                method=summary.method_id,
                done=summary.completed_seeds,
                expected=summary.expected_seeds,
                success=_format(summary.strict_success_rate),
                reward=_format(summary.reward_mean),
                body=_format(summary.body_pos_error_mean),
                ee=_format(summary.ee_z_error_mean),
            )
        )
    (output / "STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return status
