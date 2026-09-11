"""Versioned configuration for unattended AutoRefine loops."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


SCHEMA_VERSION = "mimicx.autorefine-loop.v1"


@dataclass(frozen=True)
class TaskSpec:
    id: str
    motion_file: Path
    base_checkpoint: Path
    load_run: str
    strict_task: str
    horizon: int


@dataclass(frozen=True)
class RuntimeSpec:
    workdir: Path
    python: Path
    train_script: Path
    rollout_script: Path
    dynamic_task: str
    checkpoint_root: Path
    checkpoint_glob: str
    env: Mapping[str, str]


@dataclass(frozen=True)
class SearchSpec:
    max_iterations: int
    candidate_budget: int
    train_iterations: int
    num_envs: int
    learning_rate: float
    train_seed: int
    candidate_gpus: tuple[int, ...]
    eval_gpus: tuple[int, ...]


@dataclass(frozen=True)
class GuardSpec:
    direction: str
    relative_tolerance: float = 0.0
    absolute_tolerance: float = 0.0


@dataclass(frozen=True)
class VerificationSpec:
    repeats: int
    seeds: tuple[int, ...]
    metric_keys: tuple[str, ...]
    guards: Mapping[str, GuardSpec]


@dataclass(frozen=True)
class ExecutionSpec:
    max_parallel: int


@dataclass(frozen=True)
class ReplayItemSpec:
    id: str
    checkpoint: Path
    metrics: tuple[Path, ...]


@dataclass(frozen=True)
class ReplaySpec:
    incumbent: ReplayItemSpec
    candidates: tuple[ReplayItemSpec, ...]


@dataclass(frozen=True)
class LoopManifest:
    schema: str
    source_path: Path
    task: TaskSpec
    runtime: RuntimeSpec
    search: SearchSpec
    verification: VerificationSpec
    execution: ExecutionSpec
    replay: ReplaySpec | None = None


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_id(iteration: int, patch: Mapping[str, Any]) -> str:
    canonical = json.dumps(patch, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return f"iter{iteration:03d}_{digest}"


def _mapping(payload: Mapping[str, Any], key: str, prefix: str = "") -> Mapping[str, Any]:
    name = f"{prefix}.{key}" if prefix else key
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Missing or invalid required field: {name}")
    return value


def _required(payload: Mapping[str, Any], key: str, prefix: str) -> Any:
    name = f"{prefix}.{key}"
    if key not in payload or payload[key] is None:
        raise ValueError(f"Missing required field: {name}")
    return payload[key]


def _positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _resolve_existing(value: Any, root: Path, name: str, *, directory: bool = False) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"{name} {kind} does not exist: {path}")
    return path


def _resolve_path(value: Any, root: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _resolve_executable(value: Any, root: Path, name: str) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.absolute()
    if not path.is_file():
        raise FileNotFoundError(f"{name} file does not exist: {path}")
    return path


def _int_tuple(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integers") from exc


def _parse_replay_item(
    payload: Mapping[str, Any], root: Path, name: str
) -> ReplayItemSpec:
    metrics_raw = _required(payload, "metrics", name)
    if not isinstance(metrics_raw, list) or not metrics_raw:
        raise ValueError(f"{name}.metrics must be a non-empty list")
    return ReplayItemSpec(
        id=str(_required(payload, "id", name)),
        checkpoint=_resolve_existing(
            _required(payload, "checkpoint", name), root, f"{name}.checkpoint"
        ),
        metrics=tuple(
            _resolve_existing(value, root, f"{name}.metrics[{index}]")
            for index, value in enumerate(metrics_raw)
        ),
    )


def load_loop_manifest(path: Path) -> LoopManifest:
    source_path = path.expanduser().resolve()
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Manifest root must be a mapping")
    schema = raw.get("schema")
    if schema != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema: {schema!r}; expected {SCHEMA_VERSION!r}")

    root = source_path.parent
    task_raw = _mapping(raw, "task")
    runtime_raw = _mapping(raw, "runtime")
    search_raw = _mapping(raw, "search")
    verification_raw = _mapping(raw, "verification")
    execution_raw = _mapping(raw, "execution")

    task = TaskSpec(
        id=str(_required(task_raw, "id", "task")),
        motion_file=_resolve_existing(
            _required(task_raw, "motion_file", "task"), root, "task.motion_file"
        ),
        base_checkpoint=_resolve_existing(
            _required(task_raw, "base_checkpoint", "task"),
            root,
            "task.base_checkpoint",
        ),
        load_run=str(_required(task_raw, "load_run", "task")),
        strict_task=str(_required(task_raw, "strict_task", "task")),
        horizon=_positive_int(_required(task_raw, "horizon", "task"), "task.horizon"),
    )
    runtime = RuntimeSpec(
        workdir=_resolve_existing(
            _required(runtime_raw, "workdir", "runtime"),
            root,
            "runtime.workdir",
            directory=True,
        ),
        python=_resolve_executable(
            _required(runtime_raw, "python", "runtime"), root, "runtime.python"
        ),
        train_script=_resolve_existing(
            _required(runtime_raw, "train_script", "runtime"),
            root,
            "runtime.train_script",
        ),
        rollout_script=_resolve_existing(
            _required(runtime_raw, "rollout_script", "runtime"),
            root,
            "runtime.rollout_script",
        ),
        dynamic_task=str(_required(runtime_raw, "dynamic_task", "runtime")),
        checkpoint_root=_resolve_path(
            runtime_raw.get("checkpoint_root", Path(runtime_raw["workdir"]) / "logs"),
            root,
        ),
        checkpoint_glob=str(
            runtime_raw.get("checkpoint_glob", "**/{run_name}/model_*.pt")
        ),
        env={
            str(key): str(value)
            for key, value in dict(runtime_raw.get("env", {})).items()
        },
    )
    search = SearchSpec(
        max_iterations=_positive_int(
            _required(search_raw, "max_iterations", "search"), "search.max_iterations"
        ),
        candidate_budget=_positive_int(
            _required(search_raw, "candidate_budget", "search"), "search.candidate_budget"
        ),
        train_iterations=_positive_int(
            _required(search_raw, "train_iterations", "search"), "search.train_iterations"
        ),
        num_envs=_positive_int(_required(search_raw, "num_envs", "search"), "search.num_envs"),
        learning_rate=float(_required(search_raw, "learning_rate", "search")),
        train_seed=int(search_raw.get("train_seed", 42)),
        candidate_gpus=_int_tuple(
            _required(search_raw, "candidate_gpus", "search"), "search.candidate_gpus"
        ),
        eval_gpus=_int_tuple(
            _required(search_raw, "eval_gpus", "search"), "search.eval_gpus"
        ),
    )

    repeats = _positive_int(
        _required(verification_raw, "repeats", "verification"), "verification.repeats"
    )
    seeds = _int_tuple(_required(verification_raw, "seeds", "verification"), "verification.seeds")
    if len(seeds) < repeats:
        raise ValueError("verification.seeds must contain at least verification.repeats values")
    metric_keys_raw = _required(verification_raw, "metric_keys", "verification")
    if not isinstance(metric_keys_raw, list):
        raise ValueError("verification.metric_keys must be a list")
    guards_raw = verification_raw.get("guards", {})
    if not isinstance(guards_raw, Mapping):
        raise ValueError("verification.guards must be a mapping")
    guards: dict[str, GuardSpec] = {}
    for metric, guard_raw in guards_raw.items():
        if not isinstance(guard_raw, Mapping):
            raise ValueError(f"verification.guards.{metric} must be a mapping")
        direction = str(guard_raw.get("direction", "lower"))
        if direction not in {"lower", "higher"}:
            raise ValueError(f"verification.guards.{metric}.direction must be lower or higher")
        guards[str(metric)] = GuardSpec(
            direction=direction,
            relative_tolerance=float(guard_raw.get("relative_tolerance", 0.0)),
            absolute_tolerance=float(guard_raw.get("absolute_tolerance", 0.0)),
        )
    verification = VerificationSpec(
        repeats=repeats,
        seeds=seeds,
        metric_keys=tuple(str(item) for item in metric_keys_raw),
        guards=guards,
    )
    execution = ExecutionSpec(
        max_parallel=_positive_int(
            _required(execution_raw, "max_parallel", "execution"),
            "execution.max_parallel",
        )
    )
    replay: ReplaySpec | None = None
    replay_raw = raw.get("replay")
    if replay_raw is not None:
        if not isinstance(replay_raw, Mapping):
            raise ValueError("replay must be a mapping")
        incumbent_raw = _mapping(replay_raw, "incumbent", "replay")
        candidates_raw = _required(replay_raw, "candidates", "replay")
        if not isinstance(candidates_raw, list):
            raise ValueError("replay.candidates must be a list")
        replay = ReplaySpec(
            incumbent=_parse_replay_item(incumbent_raw, root, "replay.incumbent"),
            candidates=tuple(
                _parse_replay_item(candidate, root, f"replay.candidates[{index}]")
                for index, candidate in enumerate(candidates_raw)
                if isinstance(candidate, Mapping)
            ),
        )
        if len(replay.candidates) != len(candidates_raw):
            raise ValueError("Every replay.candidates entry must be a mapping")
    return LoopManifest(
        schema=SCHEMA_VERSION,
        source_path=source_path,
        task=task,
        runtime=runtime,
        search=search,
        verification=verification,
        execution=execution,
        replay=replay,
    )
