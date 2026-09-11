#!/usr/bin/env python3
"""Freeze, prepare, and launch the native MimicX paper experiment matrix."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mimicx.refinement.checkpoints import select_highest_step_checkpoint
from mimicx.refinement.executor import CommandRecord, run_command
from mimicx.refinement.manifest import hash_file
from mimicx.refinement.state import atomic_write_json


MATRIX_SCHEMA = "mimicx.paper-matrix.v1"


@dataclass(frozen=True)
class MatrixTask:
    id: str
    source_path: Path
    motion_file: Path
    refined_motion_file: Path | None
    base_checkpoint: Path
    policy_observation_dim: int
    load_run: str
    horizon: int
    failure_window: tuple[int, int]
    priority_bodies: tuple[str, ...]
    learning_rate: float


@dataclass(frozen=True)
class MatrixMethod:
    id: str
    source_path: Path
    mode: str
    patch_profile: str
    use_refined_reference: bool
    max_iterations: int = 1
    candidate_budget: int = 3


@dataclass(frozen=True)
class PaperMatrix:
    source_path: Path
    runtime: Mapping[str, Any]
    tasks: tuple[MatrixTask, ...]
    methods: tuple[MatrixMethod, ...]
    seeds: tuple[int, ...]
    train_iterations: int
    num_envs: int
    verification_repeats: int
    verification_seeds: tuple[int, ...]


@dataclass(frozen=True)
class MatrixJob:
    id: str
    task_id: str
    method_id: str
    mode: str
    seed: int
    gpu_id: int
    motion_file: str
    motion_sha256: str
    base_checkpoint: str
    checkpoint_sha256: str
    load_run: str
    horizon: int
    learning_rate: float
    run_name: str
    job_dir: str = ""
    patch_file: str | None = None
    loop_manifest: str | None = None
    train_argv: tuple[str, ...] = ()
    runtime: Mapping[str, Any] | None = None
    verification_seeds: tuple[int, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["train_argv"] = list(self.train_argv)
        payload["verification_seeds"] = list(self.verification_seeds)
        return payload


def _load_yaml(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def _resolve_existing(value: Any, root: Path, name: str, *, directory: bool = False) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def _resolve_output(value: Any, root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _resolve_executable(value: Any, root: Path, name: str) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.absolute()
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def _load_task(path: Path) -> MatrixTask:
    raw = _load_yaml(path)
    root = path.parent
    refined = raw.get("refined_motion_file")
    window = raw.get("failure_window")
    if not isinstance(window, list) or len(window) != 2:
        raise ValueError(f"failure_window must have two entries: {path}")
    return MatrixTask(
        id=str(raw["id"]),
        source_path=path,
        motion_file=_resolve_existing(raw["motion_file"], root, "motion_file"),
        refined_motion_file=(
            _resolve_existing(refined, root, "refined_motion_file") if refined else None
        ),
        base_checkpoint=_resolve_existing(
            raw["base_checkpoint"], root, "base_checkpoint"
        ),
        policy_observation_dim=int(raw["policy_observation_dim"]),
        load_run=str(raw["load_run"]),
        horizon=int(raw["horizon"]),
        failure_window=(int(window[0]), int(window[1])),
        priority_bodies=tuple(str(item) for item in raw.get("priority_bodies", [])),
        learning_rate=float(raw["learning_rate"]),
    )


def inspect_checkpoint_dimensions(
    python: Path,
    checkpoint: Path,
) -> tuple[int, int]:
    script = """
import json
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
actor = payload["actor_state_dict"]
print(json.dumps({
    "actor_input_dim": int(actor["mlp.0.weight"].shape[1]),
    "actor_output_dim": int(actor["mlp.6.weight"].shape[0]),
    "normalizer_dim": int(actor["obs_normalizer._mean"].numel()),
}))
"""
    result = subprocess.run(
        [str(python), "-c", script, str(checkpoint)],
        check=True,
        capture_output=True,
        text=True,
    )
    record = json.loads(result.stdout)
    actor_input = int(record["actor_input_dim"])
    normalizer = int(record["normalizer_dim"])
    if actor_input != normalizer:
        raise ValueError(
            f"Checkpoint network/normalizer mismatch for {checkpoint}: "
            f"actor={actor_input}, normalizer={normalizer}"
        )
    return actor_input, int(record["actor_output_dim"])


def preflight_checkpoints(
    matrix: PaperMatrix,
    *,
    inspector=inspect_checkpoint_dimensions,
) -> list[dict[str, Any]]:
    python = Path(matrix.runtime["python"])
    records: list[dict[str, Any]] = []
    for task in matrix.tasks:
        actor_input, actor_output = inspector(python, task.base_checkpoint)
        if actor_input != task.policy_observation_dim:
            raise ValueError(
                f"Checkpoint observation mismatch for task {task.id}: "
                f"expected {task.policy_observation_dim}, observed {actor_input} "
                f"in {task.base_checkpoint}"
            )
        records.append(
            {
                "task_id": task.id,
                "checkpoint": str(task.base_checkpoint),
                "checkpoint_sha256": hash_file(task.base_checkpoint),
                "actor_input_dim": actor_input,
                "actor_output_dim": actor_output,
            }
        )
    return records


def _load_method(path: Path) -> MatrixMethod:
    raw = _load_yaml(path)
    mode = str(raw["mode"])
    if mode not in {"static", "closed_loop"}:
        raise ValueError(f"Unsupported method mode {mode!r}: {path}")
    return MatrixMethod(
        id=str(raw["id"]),
        source_path=path,
        mode=mode,
        patch_profile=str(raw.get("patch_profile", "none")),
        use_refined_reference=bool(raw.get("use_refined_reference", False)),
        max_iterations=int(raw.get("max_iterations", 1)),
        candidate_budget=int(raw.get("candidate_budget", 3)),
    )


def load_matrix(path: Path) -> PaperMatrix:
    source = path.expanduser().resolve()
    raw = _load_yaml(source)
    if raw.get("schema") != MATRIX_SCHEMA:
        raise ValueError(f"Unsupported paper matrix schema: {raw.get('schema')!r}")
    runtime_raw = raw.get("runtime")
    if not isinstance(runtime_raw, Mapping):
        raise ValueError("runtime must be a mapping")
    root = source.parent
    runtime = {
        "workdir": str(_resolve_existing(runtime_raw["workdir"], root, "runtime.workdir", directory=True)),
        "python": str(_resolve_executable(runtime_raw["python"], root, "runtime.python")),
        "train_script": str(_resolve_existing(runtime_raw["train_script"], root, "runtime.train_script")),
        "rollout_script": str(_resolve_existing(runtime_raw["rollout_script"], root, "runtime.rollout_script")),
        "checkpoint_root": str(_resolve_output(runtime_raw["checkpoint_root"], root)),
        "static_task": str(runtime_raw["static_task"]),
        "dynamic_task": str(runtime_raw["dynamic_task"]),
        "strict_task": str(runtime_raw["strict_task"]),
        "env": {str(key): str(value) for key, value in dict(runtime_raw.get("env", {})).items()},
    }
    tasks = tuple(
        _load_task(_resolve_existing(value, root, "task config"))
        for value in raw["tasks"]
    )
    methods = tuple(
        _load_method(_resolve_existing(value, root, "method config"))
        for value in raw["methods"]
    )
    training = raw["training"]
    verification = raw["verification"]
    return PaperMatrix(
        source_path=source,
        runtime=runtime,
        tasks=tasks,
        methods=methods,
        seeds=tuple(int(seed) for seed in raw["seeds"]),
        train_iterations=int(training["iterations"]),
        num_envs=int(training["num_envs"]),
        verification_repeats=int(verification["repeats"]),
        verification_seeds=tuple(int(seed) for seed in verification["seeds"]),
    )


def expand_jobs(matrix: PaperMatrix, gpu_ids: Sequence[int]) -> list[MatrixJob]:
    if not gpu_ids:
        raise ValueError("At least one GPU is required")
    jobs: list[MatrixJob] = []
    index = 0
    for task in matrix.tasks:
        for method in matrix.methods:
            for seed in matrix.seeds:
                motion = (
                    task.refined_motion_file
                    if method.use_refined_reference and task.refined_motion_file is not None
                    else task.motion_file
                )
                job_id = f"{task.id}__{method.id}__seed{seed}"
                jobs.append(
                    MatrixJob(
                        id=job_id,
                        task_id=task.id,
                        method_id=method.id,
                        mode=method.mode,
                        seed=seed,
                        gpu_id=int(gpu_ids[index % len(gpu_ids)]),
                        motion_file=str(motion),
                        motion_sha256=hash_file(motion),
                        base_checkpoint=str(task.base_checkpoint),
                        checkpoint_sha256=hash_file(task.base_checkpoint),
                        load_run=task.load_run,
                        horizon=task.horizon,
                        learning_rate=task.learning_rate,
                        run_name=f"paper_v1_{task.id}_{method.id}_seed{seed}",
                        runtime=matrix.runtime,
                        verification_seeds=matrix.verification_seeds[
                            : matrix.verification_repeats
                        ],
                    )
                )
                index += 1
    return jobs


def _patch_for(task: MatrixTask, method: MatrixMethod) -> dict[str, Any] | None:
    if method.patch_profile == "none":
        return None
    start, end = task.failure_window
    patch: dict[str, Any] = {
        "id": f"paper_{task.id}_{method.id}",
        "base_profile": "mimicx",
        "sampling": {"mode": "window", "start": start, "end": end, "ratio": 0.50},
        "reward_overrides": {},
        "body_pos_rewards": [],
        "body_z_rewards": [],
        "body_lin_vel_rewards": [],
        "body_ang_vel_rewards": [],
    }
    if method.patch_profile == "window":
        return patch
    if method.patch_profile != "hierarchical":
        raise ValueError(f"Unsupported static patch profile: {method.patch_profile}")
    bodies = list(task.priority_bodies)
    patch["reward_overrides"] = {
        "motion_global_root_pos": {"weight": 5.0, "std": 0.15},
        "motion_global_root_ori": {"weight": 1.2, "std": 0.28},
        "motion_body_pos": {"std": 0.24},
        "motion_body_ori": {"std": 0.32},
        "action_rate_l2": {"weight": -0.07},
    }
    patch["body_pos_rewards"] = [
        {"name": "task_local_pos", "body_names": bodies, "weight": 2.2, "std": 0.12},
        {"name": "global_core_pos", "body_names": ["pelvis", "torso_link"], "weight": 2.5, "std": 0.11},
    ]
    patch["body_z_rewards"] = [
        {"name": "task_local_z", "body_names": bodies, "weight": 1.2, "std": 0.06}
    ]
    patch["body_lin_vel_rewards"] = [
        {"name": "task_local_lin_vel", "body_names": bodies, "weight": 0.7, "std": 0.70}
    ]
    patch["body_ang_vel_rewards"] = [
        {"name": "core_ang_vel", "body_names": ["pelvis", "torso_link"], "weight": 0.35, "std": 2.1}
    ]
    return patch


def _static_train_argv(matrix: PaperMatrix, task: MatrixTask, method: MatrixMethod, job: MatrixJob) -> tuple[str, ...]:
    task_id = matrix.runtime["static_task"] if method.patch_profile == "none" else matrix.runtime["dynamic_task"]
    return (
        matrix.runtime["python"],
        matrix.runtime["train_script"],
        task_id,
        "--motion-file",
        job.motion_file,
        "--env.scene.num-envs",
        str(matrix.num_envs),
        "--agent.max-iterations",
        str(matrix.train_iterations),
        "--agent.save-interval",
        "25",
        "--agent.run-name",
        job.run_name,
        "--agent.resume",
        "True",
        "--agent.load-run",
        task.load_run,
        "--agent.load-checkpoint",
        task.base_checkpoint.name,
        "--agent.algorithm.learning-rate",
        str(task.learning_rate),
        "--agent.seed",
        str(job.seed),
    )


def _write_loop_manifest(
    matrix: PaperMatrix,
    task: MatrixTask,
    method: MatrixMethod,
    job: MatrixJob,
    path: Path,
) -> None:
    payload = {
        "schema": "mimicx.autorefine-loop.v1",
        "task": {
            "id": task.id,
            "motion_file": job.motion_file,
            "base_checkpoint": str(task.base_checkpoint),
            "load_run": task.load_run,
            "strict_task": matrix.runtime["strict_task"],
            "horizon": task.horizon,
        },
        "runtime": {
            "workdir": matrix.runtime["workdir"],
            "python": matrix.runtime["python"],
            "train_script": matrix.runtime["train_script"],
            "rollout_script": matrix.runtime["rollout_script"],
            "dynamic_task": matrix.runtime["dynamic_task"],
            "checkpoint_root": matrix.runtime["checkpoint_root"],
            "checkpoint_glob": "**/{run_name}/model_*.pt",
            "env": matrix.runtime.get("env", {}),
        },
        "search": {
            "max_iterations": method.max_iterations,
            "candidate_budget": method.candidate_budget,
            "train_iterations": matrix.train_iterations,
            "num_envs": matrix.num_envs,
            "learning_rate": task.learning_rate,
            "train_seed": job.seed,
            "candidate_gpus": [job.gpu_id],
            "eval_gpus": [job.gpu_id],
        },
        "verification": {
            "repeats": matrix.verification_repeats,
            "seeds": list(matrix.verification_seeds),
            "metric_keys": ["body_pos_error_max", "ee_z_error_max"],
            "guards": {
                "body_pos_error_max": {"direction": "lower", "relative_tolerance": 0.20}
            },
        },
        "execution": {"max_parallel": 1},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _resolved_matrix(
    matrix: PaperMatrix,
    jobs: Sequence[MatrixJob],
    checkpoint_preflight: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "schema": MATRIX_SCHEMA,
        "source": str(matrix.source_path),
        "source_sha256": hash_file(matrix.source_path),
        "task_configs": [
            {"path": str(task.source_path), "sha256": hash_file(task.source_path)}
            for task in matrix.tasks
        ],
        "method_configs": [
            {"path": str(method.source_path), "sha256": hash_file(method.source_path)}
            for method in matrix.methods
        ],
        "seeds": list(matrix.seeds),
        "training": {"iterations": matrix.train_iterations, "num_envs": matrix.num_envs},
        "verification": {
            "repeats": matrix.verification_repeats,
            "seeds": list(matrix.verification_seeds),
        },
        "checkpoint_preflight": list(checkpoint_preflight or []),
        "jobs": len(jobs),
    }


def prepare_runpack(
    matrix: PaperMatrix,
    *,
    run_dir: Path,
    gpu_ids: Sequence[int],
    checkpoint_preflight: Sequence[Mapping[str, Any]] | None = None,
) -> list[MatrixJob]:
    root = run_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    raw_jobs = expand_jobs(matrix, gpu_ids)
    task_by_id = {task.id: task for task in matrix.tasks}
    method_by_id = {method.id: method for method in matrix.methods}
    prepared: list[MatrixJob] = []
    for raw_job in raw_jobs:
        task = task_by_id[raw_job.task_id]
        method = method_by_id[raw_job.method_id]
        job_dir = root / "jobs" / raw_job.id
        job_dir.mkdir(parents=True, exist_ok=True)
        patch_file: Path | None = None
        loop_manifest: Path | None = None
        train_argv: tuple[str, ...] = ()
        if method.mode == "static":
            patch = _patch_for(task, method)
            if patch is not None:
                patch_file = job_dir / "patch.json"
                atomic_write_json(patch_file, patch)
            train_argv = _static_train_argv(matrix, task, method, raw_job)
        else:
            loop_manifest = job_dir / "loop_manifest.yaml"
            _write_loop_manifest(matrix, task, method, raw_job, loop_manifest)
        job = MatrixJob(
            **{
                **raw_job.as_dict(),
                "job_dir": str(job_dir),
                "patch_file": str(patch_file) if patch_file else None,
                "loop_manifest": str(loop_manifest) if loop_manifest else None,
                "train_argv": train_argv,
                "verification_seeds": raw_job.verification_seeds,
            }
        )
        atomic_write_json(job_dir / "job.json", job.as_dict())
        prepared.append(job)
    with (root / "jobs.jsonl").open("w", encoding="utf-8") as handle:
        for job in prepared:
            handle.write(json.dumps(job.as_dict(), sort_keys=True) + "\n")
    atomic_write_json(
        root / "resolved_matrix.json",
        _resolved_matrix(matrix, prepared, checkpoint_preflight),
    )

    launcher = Path(__file__).resolve()
    for gpu_id in gpu_ids:
        failure_log = root / "queue_failures.log"
        lines = [
            "#!/usr/bin/env bash",
            "set -uo pipefail",
            "",
            "run_job() {",
            "  local job_path=\"$1\"",
            "  local status=0",
            (
                f"  MIMICX_DEVICE_ID={gpu_id} {shlex.quote(sys.executable)} "
                f"{shlex.quote(str(launcher))} --run-job \"$job_path\" || status=$?"
            ),
            "  if (( status != 0 )); then",
            (
                "    printf '%s gpu=%s status=%s job=%s\\n' "
                f"\"$(date --iso-8601=seconds)\" {gpu_id} \"$status\" \"$job_path\" "
                f">> {shlex.quote(str(failure_log))}"
            ),
            "  fi",
            "  return 0",
            "}",
            "",
        ]
        for job in prepared:
            if job.gpu_id != int(gpu_id):
                continue
            lines.append(
                f"run_job {shlex.quote(str(Path(job.job_dir) / 'job.json'))}"
            )
        queue = root / f"queue_gpu{gpu_id}.sh"
        queue.write_text("\n".join(lines) + "\n", encoding="utf-8")
        queue.chmod(0o755)
    return prepared


def _runtime_environment(job: Mapping[str, Any]) -> dict[str, str]:
    runtime = job["runtime"]
    workdir = Path(runtime["workdir"])
    environment = os.environ.copy()
    environment.update(runtime.get("env", {}))
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(workdir), str(workdir / "src"), environment.get("PYTHONPATH", "")]
    )
    environment["WANDB_MODE"] = "offline"
    return environment


def _runtime_gpu_id(job: Mapping[str, Any]) -> int:
    override = os.environ.get("MIMICX_RUNTIME_GPU_ID")
    return int(override) if override is not None else int(job["gpu_id"])


def _discover_static_checkpoint(job: Mapping[str, Any]) -> Path:
    root = Path(job["runtime"]["checkpoint_root"])
    matches = list(root.glob(f"**/*{job['run_name']}*/model_*.pt"))
    if not matches:
        raise FileNotFoundError(f"No checkpoint found for {job['id']} under {root}")
    return select_highest_step_checkpoint(matches).resolve()


def run_job(path: Path) -> int:
    job_path = path.expanduser().resolve()
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job_dir = Path(job["job_dir"])
    done = job_dir / "job.done.json"
    if done.exists():
        return 0
    environment = _runtime_environment(job)
    runtime_gpu_id = _runtime_gpu_id(job)
    environment["MIMICX_DEVICE_ID"] = str(runtime_gpu_id)
    if job["mode"] == "closed_loop":
        argv = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "autorefine" / "run_closed_loop.py"),
            "--manifest",
            job["loop_manifest"],
            "--run-dir",
            str(job_dir / "closed_loop"),
            "--mode",
            "execute",
            "--runtime-gpu",
            str(runtime_gpu_id),
        ]
        log_path = job_dir / "closed_loop.log"
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run(argv, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0:
            return result.returncode
        summary = json.loads((job_dir / "closed_loop" / "loop_summary.json").read_text())
        checkpoint = Path(
            json.loads((job_dir / "closed_loop" / "state.json").read_text())["incumbent"]["checkpoint"]
        )
        atomic_write_json(done, {"job": job["id"], "checkpoint": str(checkpoint), "summary": summary})
        return 0

    if job.get("patch_file"):
        environment["MIMICX_AUTOREFINE_PATCH_FILE"] = job["patch_file"]
    train_done = job_dir / "train.done.json"
    if train_done.exists():
        train_record = json.loads(train_done.read_text(encoding="utf-8"))
        checkpoint = Path(train_record["checkpoint"]).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Recorded training checkpoint does not exist: {checkpoint}"
            )
    else:
        train_log = job_dir / "train.log"
        with train_log.open("a", encoding="utf-8") as log:
            result = subprocess.run(
                job["train_argv"],
                cwd=job["runtime"]["workdir"],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            return result.returncode
        checkpoint = _discover_static_checkpoint(job)
        atomic_write_json(
            train_done,
            {"job": job["id"], "checkpoint": str(checkpoint)},
        )
    metrics_paths = []
    for seed in job["verification_seeds"]:
        metrics = job_dir / "eval" / f"seed_{seed}" / "metrics.json"
        steps = job_dir / "eval" / f"seed_{seed}" / "metrics_steps.csv"
        metrics.parent.mkdir(parents=True, exist_ok=True)
        if metrics.is_file() and steps.is_file():
            json.loads(metrics.read_text(encoding="utf-8"))
            metrics_paths.append(str(metrics))
            continue
        argv = [
            job["runtime"]["python"],
            job["runtime"]["rollout_script"],
            "--task",
            job["runtime"]["strict_task"],
            "--motion-file",
            job["motion_file"],
            "--num-envs",
            "1",
            "--steps",
            str(job["horizon"]),
            "--seed",
            str(seed),
            "--device",
            "cuda:0",
            "--disable-joint-init-noise",
            "--checkpoint-file",
            str(checkpoint),
            "--metrics-output",
            str(metrics),
            "--step-metrics-output",
            str(steps),
        ]
        result = run_command(
            CommandRecord(
                id=f"{job['id']}/eval_seed_{seed}",
                argv=tuple(argv),
                cwd=Path(job["runtime"]["workdir"]),
                log_path=job_dir / "eval" / f"seed_{seed}.log",
                expected_outputs=(metrics, steps),
                env=environment,
                recoverable_exit_codes=(-11,),
                success_log_marker="rollout_ok steps=",
            )
        )
        if result.exit_code != 0:
            return result.exit_code
        metrics_paths.append(str(metrics))
    atomic_write_json(
        done,
        {"job": job["id"], "checkpoint": str(checkpoint), "metrics": metrics_paths},
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--gpus", type=int, nargs="+")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run-job", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.run_job:
        return run_job(args.run_job)
    if not args.matrix or not args.run_dir or not args.gpus:
        raise SystemExit("--matrix, --run-dir, and --gpus are required")
    matrix = load_matrix(args.matrix)
    checkpoint_preflight = preflight_checkpoints(matrix)
    jobs = prepare_runpack(
        matrix,
        run_dir=args.run_dir,
        gpu_ids=args.gpus,
        checkpoint_preflight=checkpoint_preflight,
    )
    print(f"prepared_jobs={len(jobs)} run_dir={args.run_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
