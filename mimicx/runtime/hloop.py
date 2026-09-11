"""Exact orchestration for independent AutoRefine candidate jobs.

E0 does not split an RL update or change simulator semantics. It executes the
same immutable job DAG either sequentially or concurrently and records enough
hashes to prove whether outputs and selection remained identical.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Literal, Mapping

import yaml


ResourceKind = Literal["cpu", "gpu", "io"]
RunMode = Literal["sequential", "e0"]


@dataclass(frozen=True)
class ArtifactSpec:
    role: str
    path: str
    sha256: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArtifactSpec":
        return cls(
            role=str(value["role"]),
            path=str(value["path"]),
            sha256=None if value.get("sha256") is None else str(value["sha256"]),
        )


@dataclass(frozen=True)
class HLoopJob:
    job_id: str
    resource: ResourceKind
    command: tuple[str, ...]
    depends_on: tuple[str, ...] = ()
    cwd: str = "{repo_root}"
    environment: dict[str, str] = field(default_factory=dict)
    inputs: tuple[ArtifactSpec, ...] = ()
    outputs: tuple[ArtifactSpec, ...] = ()
    candidate_id: str | None = None
    seed: int | None = None
    budget: dict[str, Any] = field(default_factory=dict)
    cuda_visible_devices: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HLoopJob":
        command = value.get("command")
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            raise TypeError("HLoop job command must be a list of strings")
        return cls(
            job_id=str(value["job_id"]),
            resource=str(value["resource"]),  # type: ignore[arg-type]
            command=tuple(command),
            depends_on=tuple(str(item) for item in value.get("depends_on", ())),
            cwd=str(value.get("cwd", "{repo_root}")),
            environment={str(key): str(item) for key, item in dict(value.get("environment", {})).items()},
            inputs=tuple(ArtifactSpec.from_mapping(item) for item in value.get("inputs", ())),
            outputs=tuple(ArtifactSpec.from_mapping(item) for item in value.get("outputs", ())),
            candidate_id=(None if value.get("candidate_id") is None else str(value["candidate_id"])),
            seed=None if value.get("seed") is None else int(value["seed"]),
            budget=dict(value.get("budget", {})),
            cuda_visible_devices=(
                None
                if value.get("cuda_visible_devices") is None
                else str(value["cuda_visible_devices"])
            ),
        )

    def __post_init__(self) -> None:
        if not self.job_id or not self.command:
            raise ValueError("HLoop job id and command are required")
        if self.resource not in {"cpu", "gpu", "io"}:
            raise ValueError(f"unsupported HLoop resource: {self.resource}")
        if self.resource == "gpu" and self.cuda_visible_devices is None:
            raise ValueError(f"GPU job {self.job_id} must pin cuda_visible_devices")


@dataclass(frozen=True)
class HLoopPlan:
    schema_version: str
    plan_id: str
    jobs: tuple[HLoopJob, ...]
    max_cpu_workers: int = 4
    max_io_workers: int = 2
    selection_artifact: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HLoopPlan":
        plan = cls(
            schema_version=str(value["schema_version"]),
            plan_id=str(value["plan_id"]),
            jobs=tuple(HLoopJob.from_mapping(item) for item in value.get("jobs", ())),
            max_cpu_workers=int(value.get("max_cpu_workers", 4)),
            max_io_workers=int(value.get("max_io_workers", 2)),
            selection_artifact=(
                None
                if value.get("selection_artifact") is None
                else str(value["selection_artifact"])
            ),
        )
        plan.validate()
        return plan

    def validate(self) -> None:
        if self.schema_version != "mimicx.hloop-plan.v1":
            raise ValueError(f"unsupported HLoop plan schema: {self.schema_version}")
        if not self.plan_id or not self.jobs:
            raise ValueError("HLoop plan id and jobs are required")
        if self.max_cpu_workers < 1 or self.max_io_workers < 1:
            raise ValueError("HLoop worker limits must be positive")
        ids = [job.job_id for job in self.jobs]
        if len(ids) != len(set(ids)):
            raise ValueError("HLoop job ids must be unique")
        known = set(ids)
        for job in self.jobs:
            unknown = set(job.depends_on) - known
            if unknown:
                raise ValueError(f"job {job.job_id} has unknown dependencies: {sorted(unknown)}")
            if job.job_id in job.depends_on:
                raise ValueError(f"job {job.job_id} depends on itself")
        _topological_ids(self.jobs)

    def semantic_hash(self) -> str:
        return digest_json(
            {
                "schema_version": self.schema_version,
                "plan_id": self.plan_id,
                "max_cpu_workers": self.max_cpu_workers,
                "max_io_workers": self.max_io_workers,
                "selection_artifact": self.selection_artifact,
                "jobs": [asdict(job) for job in self.jobs],
            }
        )


def load_hloop_plan(path: Path) -> HLoopPlan:
    with path.expanduser().resolve().open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError(f"HLoop plan must contain a mapping: {path}")
    return HLoopPlan.from_mapping(payload)


def run_hloop_plan(
    plan: HLoopPlan,
    *,
    run_dir: Path,
    repo_root: Path,
    mode: RunMode,
) -> dict[str, Any]:
    plan.validate()
    output = run_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"HLoop run directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    context = {"repo_root": str(repo_root.expanduser().resolve()), "run_dir": str(output)}
    started = time.monotonic()
    started_at = _now()
    if mode == "sequential":
        results = _run_sequential(plan, output=output, context=context)
    elif mode == "e0":
        results = _run_e0(plan, output=output, context=context)
    else:
        raise ValueError(f"unsupported HLoop mode: {mode}")
    report = {
        "schema_version": "mimicx.hloop-run.v1",
        "plan_id": plan.plan_id,
        "plan_semantic_hash": plan.semantic_hash(),
        "mode": mode,
        "started_at": started_at,
        "completed_at": _now(),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "status": "completed",
        "candidate_ids": sorted(
            job.candidate_id for job in plan.jobs if job.candidate_id is not None
        ),
        "jobs": results,
        "selection": _selection_record(plan, context=context),
    }
    _write_json(output / "hloop_run.json", report)
    return report


def compare_hloop_runs(sequential: Mapping[str, Any], e0: Mapping[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, left: Any, right: Any) -> None:
        checks.append({"name": name, "passed": left == right, "sequential": left, "e0": right})

    check("plan_semantic_hash", sequential.get("plan_semantic_hash"), e0.get("plan_semantic_hash"))
    check("candidate_ids", sequential.get("candidate_ids"), e0.get("candidate_ids"))
    left_jobs = {str(item["job_id"]): item for item in sequential.get("jobs", ())}
    right_jobs = {str(item["job_id"]): item for item in e0.get("jobs", ())}
    check("job_ids", sorted(left_jobs), sorted(right_jobs))
    for job_id in sorted(set(left_jobs) | set(right_jobs)):
        left = left_jobs.get(job_id, {})
        right = right_jobs.get(job_id, {})
        check(f"{job_id}.command_semantic_hash", left.get("command_semantic_hash"), right.get("command_semantic_hash"))
        check(f"{job_id}.input_hashes", left.get("inputs"), right.get("inputs"))
        check(f"{job_id}.output_hashes", left.get("outputs"), right.get("outputs"))
        check(f"{job_id}.returncode", left.get("returncode"), right.get("returncode"))
    check("selection", sequential.get("selection"), e0.get("selection"))
    passed = all(item["passed"] for item in checks)
    return {
        "schema_version": "mimicx.hloop-parity.v1",
        "passed": passed,
        "checks": checks,
        "speed": {
            "sequential_seconds": sequential.get("elapsed_seconds"),
            "e0_seconds": e0.get("elapsed_seconds"),
            "speedup": _speedup(sequential.get("elapsed_seconds"), e0.get("elapsed_seconds")),
            "claimable": False,
            "note": "Parity fixtures do not support a paper speedup claim.",
        },
    }


def _run_sequential(plan: HLoopPlan, *, output: Path, context: Mapping[str, str]) -> list[dict[str, Any]]:
    jobs = {job.job_id: job for job in plan.jobs}
    results: list[dict[str, Any]] = []
    for job_id in _topological_ids(plan.jobs):
        results.append(_execute_job(jobs[job_id], output=output, context=context))
    return results


def _run_e0(plan: HLoopPlan, *, output: Path, context: Mapping[str, str]) -> list[dict[str, Any]]:
    pending = {job.job_id: job for job in plan.jobs}
    completed: set[str] = set()
    active: dict[Future[dict[str, Any]], HLoopJob] = {}
    active_cpu = 0
    active_io = 0
    active_gpus: set[str] = set()
    results: dict[str, dict[str, Any]] = {}
    max_workers = plan.max_cpu_workers + plan.max_io_workers + max(
        1,
        len({job.cuda_visible_devices for job in plan.jobs if job.resource == "gpu"}),
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while pending or active:
            launched = False
            for job_id, job in list(pending.items()):
                if not set(job.depends_on).issubset(completed):
                    continue
                if job.resource == "cpu" and active_cpu >= plan.max_cpu_workers:
                    continue
                if job.resource == "io" and active_io >= plan.max_io_workers:
                    continue
                if job.resource == "gpu" and job.cuda_visible_devices in active_gpus:
                    continue
                future = executor.submit(_execute_job, job, output=output, context=context)
                active[future] = job
                pending.pop(job_id)
                active_cpu += int(job.resource == "cpu")
                active_io += int(job.resource == "io")
                if job.resource == "gpu":
                    assert job.cuda_visible_devices is not None
                    active_gpus.add(job.cuda_visible_devices)
                launched = True
            if not active:
                if pending:
                    raise RuntimeError("HLoop scheduler reached an unschedulable state")
                break
            if launched:
                continue
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                job = active.pop(future)
                active_cpu -= int(job.resource == "cpu")
                active_io -= int(job.resource == "io")
                if job.resource == "gpu":
                    assert job.cuda_visible_devices is not None
                    active_gpus.remove(job.cuda_visible_devices)
                result = future.result()
                results[job.job_id] = result
                completed.add(job.job_id)
        for future, job in list(active.items()):
            results[job.job_id] = future.result()
    order = _topological_ids(plan.jobs)
    return [results[job_id] for job_id in order]


def _execute_job(job: HLoopJob, *, output: Path, context: Mapping[str, str]) -> dict[str, Any]:
    inputs = [_artifact_record(spec, context=context, require_expected_hash=True) for spec in job.inputs]
    command = [_render(item, context) for item in job.command]
    cwd = Path(_render(job.cwd, context)).expanduser().resolve()
    environment = os.environ.copy()
    environment.update({key: _render(value, context) for key, value in job.environment.items()})
    if job.cuda_visible_devices is not None:
        environment["MIMICX_DEVICE_ID"] = job.cuda_visible_devices
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            environment["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    log_path = output / "logs" / f"{job.job_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"HLoop job {job.job_id} failed with rc={completed.returncode}; see {log_path}")
    outputs = [_artifact_record(spec, context=context, require_expected_hash=False) for spec in job.outputs]
    semantic_command = [_normalize_run_dir(item, context["run_dir"]) for item in command]
    return {
        "job_id": job.job_id,
        "candidate_id": job.candidate_id,
        "resource": job.resource,
        "cuda_visible_devices": job.cuda_visible_devices,
        "seed": job.seed,
        "budget": job.budget,
        "command": command,
        "command_semantic_hash": digest_json(semantic_command),
        "inputs": inputs,
        "outputs": outputs,
        "returncode": completed.returncode,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "log": str(log_path),
    }


def _artifact_record(
    spec: ArtifactSpec,
    *,
    context: Mapping[str, str],
    require_expected_hash: bool,
) -> dict[str, Any]:
    path = Path(_render(spec.path, context)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"HLoop artifact is missing: {path}")
    observed = sha256_file(path)
    if spec.sha256 is not None and observed != spec.sha256:
        raise ValueError(f"HLoop artifact hash mismatch: {path}")
    if require_expected_hash and spec.sha256 is None and "{run_dir}" not in spec.path:
        raise ValueError(f"immutable external input lacks expected sha256: {path}")
    return {
        "role": spec.role,
        "path": _normalize_run_dir(str(path), context["run_dir"]),
        "sha256": observed,
        "size_bytes": path.stat().st_size,
    }


def _selection_record(plan: HLoopPlan, *, context: Mapping[str, str]) -> dict[str, Any] | None:
    if plan.selection_artifact is None:
        return None
    path = Path(_render(plan.selection_artifact, context)).expanduser().resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    return {"sha256": sha256_file(path), "value": value}


def _topological_ids(jobs: tuple[HLoopJob, ...]) -> list[str]:
    pending = {job.job_id: set(job.depends_on) for job in jobs}
    order: list[str] = []
    while pending:
        ready = sorted(job_id for job_id, dependencies in pending.items() if not dependencies)
        if not ready:
            raise ValueError("HLoop job graph contains a dependency cycle")
        for job_id in ready:
            order.append(job_id)
            pending.pop(job_id)
        for dependencies in pending.values():
            dependencies.difference_update(ready)
    return order


def _render(value: str, context: Mapping[str, str]) -> str:
    try:
        return value.format_map(dict(context))
    except KeyError as exc:
        raise KeyError(f"unknown HLoop template field: {exc.args[0]}") from exc


def _normalize_run_dir(value: str, run_dir: str) -> str:
    return value.replace(str(Path(run_dir).resolve()), "${RUN_DIR}")


def digest_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _speedup(sequential: Any, e0: Any) -> float | None:
    if not isinstance(sequential, (int, float)) or not isinstance(e0, (int, float)) or e0 <= 0:
        return None
    return float(sequential) / float(e0)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
