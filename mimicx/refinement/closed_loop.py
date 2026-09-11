"""Unattended policy-in-the-loop AutoRefine coordination."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mimicx.refinement.checkpoints import select_highest_step_checkpoint
from mimicx.refinement.executor import CommandRecord, CommandResult, run_command, run_commands
from mimicx.refinement.gate import (
    RepeatAggregate,
    aggregate_repeats,
    decide_acceptance,
    load_rollout_metrics,
)
from mimicx.refinement.manifest import LoopManifest, ReplayItemSpec, hash_file
from mimicx.refinement.state import atomic_write_json, load_or_create_state, save_state


@dataclass(frozen=True)
class IterationManifest:
    iteration: int
    status: str
    commands: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["commands"] = list(self.commands)
        return payload


@dataclass(frozen=True)
class LoopSummary:
    status: str
    iteration: int
    incumbent_id: str
    accepted: bool
    selected_id: str
    run_dir: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ClosedLoopCoordinator:
    def __init__(
        self,
        manifest: LoopManifest,
        run_dir: Path,
        *,
        runtime_gpu_ids: tuple[int, ...] | None = None,
    ) -> None:
        self.manifest = manifest
        self.candidate_gpus = runtime_gpu_ids or manifest.search.candidate_gpus
        self.eval_gpus = runtime_gpu_ids or manifest.search.eval_gpus
        self.run_dir = run_dir.expanduser().resolve()
        self.project_root = Path(__file__).resolve().parents[2]
        self.manifest_hash = hash_file(manifest.source_path)
        self.state = load_or_create_state(self.run_dir, self.manifest_hash)
        self._write_resolved_manifest()

    def _iteration_dir(self) -> Path:
        return self.run_dir / "iterations" / f"iter_{self.state.current_iteration:03d}"

    def _write_resolved_manifest(self) -> None:
        replay: dict[str, Any] | None = None
        if self.manifest.replay is not None:
            replay = {
                "incumbent": self._replay_item_dict(self.manifest.replay.incumbent),
                "candidates": [
                    self._replay_item_dict(item) for item in self.manifest.replay.candidates
                ],
            }
        payload = {
            "schema": self.manifest.schema,
            "source_path": str(self.manifest.source_path),
            "source_sha256": self.manifest_hash,
            "task": {
                "id": self.manifest.task.id,
                "motion_file": str(self.manifest.task.motion_file),
                "motion_sha256": hash_file(self.manifest.task.motion_file),
                "base_checkpoint": str(self.manifest.task.base_checkpoint),
                "base_checkpoint_sha256": hash_file(self.manifest.task.base_checkpoint),
                "load_run": self.manifest.task.load_run,
                "strict_task": self.manifest.task.strict_task,
                "horizon": self.manifest.task.horizon,
            },
            "runtime": {
                "workdir": str(self.manifest.runtime.workdir),
                "python": str(self.manifest.runtime.python),
                "train_script": str(self.manifest.runtime.train_script),
                "rollout_script": str(self.manifest.runtime.rollout_script),
                "dynamic_task": self.manifest.runtime.dynamic_task,
                "checkpoint_root": str(self.manifest.runtime.checkpoint_root),
                "checkpoint_glob": self.manifest.runtime.checkpoint_glob,
                "env": dict(self.manifest.runtime.env),
            },
            "search": asdict(self.manifest.search),
            "verification": {
                "repeats": self.manifest.verification.repeats,
                "seeds": list(self.manifest.verification.seeds),
                "metric_keys": list(self.manifest.verification.metric_keys),
                "guards": {
                    key: asdict(value)
                    for key, value in self.manifest.verification.guards.items()
                },
            },
            "execution": asdict(self.manifest.execution),
            "replay": replay,
        }
        payload["search"]["candidate_gpus"] = list(self.candidate_gpus)
        payload["search"]["eval_gpus"] = list(self.eval_gpus)
        atomic_write_json(self.run_dir / "loop_manifest.resolved.json", payload)

    @staticmethod
    def _replay_item_dict(item: ReplayItemSpec) -> dict[str, Any]:
        return {
            "id": item.id,
            "checkpoint": str(item.checkpoint),
            "checkpoint_sha256": hash_file(item.checkpoint),
            "metrics": [
                {"path": str(path), "sha256": hash_file(path)} for path in item.metrics
            ],
        }

    def _incumbent_checkpoint(self) -> Path:
        value = self.state.incumbent.get("checkpoint")
        return Path(value) if value else self.manifest.task.base_checkpoint

    def _initialize_incumbent(self) -> None:
        if self.state.incumbent:
            return
        item = self.manifest.replay.incumbent if self.manifest.replay else None
        checkpoint = item.checkpoint if item else self.manifest.task.base_checkpoint
        self.state.incumbent = {
            "id": item.id if item else "incumbent",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hash_file(checkpoint),
        }
        save_state(self.run_dir, self.state)

    def _rollout_command(self, seed: int, checkpoint: Path) -> dict[str, Any]:
        output_dir = self._iteration_dir() / "incumbent" / f"seed_{seed}"
        metrics = output_dir / "metrics.json"
        steps = output_dir / "metrics_steps.csv"
        argv = [
            str(self.manifest.runtime.python),
            str(self.manifest.runtime.rollout_script),
            "--task",
            self.manifest.task.strict_task,
            "--motion-file",
            str(self.manifest.task.motion_file),
            "--num-envs",
            "1",
            "--steps",
            str(self.manifest.task.horizon),
            "--device",
            "cuda:0",
            "--disable-joint-init-noise",
            "--seed",
            str(seed),
            "--checkpoint-file",
            str(checkpoint),
            "--metrics-output",
            str(metrics),
            "--step-metrics-output",
            str(steps),
        ]
        return {
            "id": f"incumbent_rollout_seed{seed}",
            "argv": argv,
            "cwd": str(self.manifest.runtime.workdir),
            "gpu_pool": list(self.eval_gpus),
            "seed": seed,
            "expected_outputs": [str(metrics), str(steps)],
            "recoverable_exit_codes": [-11],
            "success_log_marker": "rollout_ok steps=",
        }

    def prepare(self) -> IterationManifest:
        self._initialize_incumbent()
        iteration_dir = self._iteration_dir()
        commands = [
            self._rollout_command(seed, self._incumbent_checkpoint())
            for seed in self.manifest.verification.seeds[: self.manifest.verification.repeats]
        ]
        first_metrics = Path(commands[0]["expected_outputs"][0])
        first_steps = Path(commands[0]["expected_outputs"][1])
        failure_json = iteration_dir / "failure" / "failure_report.json"
        commands.append(
            {
                "id": "mine_failures",
                "argv": [
                    str(self.manifest.runtime.python),
                    str(self.project_root / "scripts" / "autorefine" / "mine_failures.py"),
                    "--metrics-json",
                    str(first_metrics),
                    "--step-csv",
                    str(first_steps),
                    "--output-json",
                    str(failure_json),
                    "--output-md",
                    str(iteration_dir / "failure" / "failure_report.md"),
                ],
                "cwd": str(self.manifest.runtime.workdir),
                "expected_outputs": [str(failure_json)],
            }
        )
        candidate_dir = iteration_dir / "candidates"
        commands.append(
            {
                "id": "propose_candidates",
                "argv": [
                    str(self.manifest.runtime.python),
                    str(
                        self.project_root
                        / "scripts"
                        / "autorefine"
                        / "propose_candidates.py"
                    ),
                    "--failure-report",
                    str(failure_json),
                    "--runtime-workdir",
                    str(self.manifest.runtime.workdir),
                    "--runtime-python",
                    str(self.manifest.runtime.python),
                    "--output-dir",
                    str(candidate_dir),
                    "--motion-file",
                    str(self.manifest.task.motion_file),
                    "--base-checkpoint",
                    str(self._incumbent_checkpoint()),
                    "--load-run",
                    self.manifest.task.load_run,
                    "--train-iters",
                    str(self.manifest.search.train_iterations),
                    "--num-envs",
                    str(self.manifest.search.num_envs),
                    "--learning-rate",
                    str(self.manifest.search.learning_rate),
                    "--run-prefix",
                    (
                        f"mimicx_{self.manifest.task.id}_seed"
                        f"{self.manifest.search.train_seed}_iter"
                        f"{self.state.current_iteration:03d}"
                    ),
                    "--task-id",
                    self.manifest.task.id,
                    "--candidate-budget",
                    str(self.manifest.search.candidate_budget),
                ],
                "cwd": str(self.manifest.runtime.workdir),
                "expected_outputs": [str(candidate_dir / "candidate_matrix.json")],
            }
        )
        iteration = IterationManifest(
            iteration=self.state.current_iteration,
            status="prepared",
            commands=tuple(commands),
        )
        atomic_write_json(iteration_dir / "iteration_manifest.json", iteration.as_dict())
        return iteration

    def _aggregate_item(self, item: ReplayItemSpec) -> RepeatAggregate:
        records = [
            load_rollout_metrics(
                path,
                horizon=self.manifest.task.horizon,
                metric_keys=self.manifest.verification.metric_keys,
                seed=seed,
            )
            for path, seed in zip(
                item.metrics,
                self.manifest.verification.seeds,
                strict=False,
            )
        ]
        return aggregate_repeats(item.id, records)

    def _load_terminal_summary(self) -> LoopSummary | None:
        summary_path = self.run_dir / "loop_summary.json"
        if self.state.status != "replay_completed" or not summary_path.exists():
            return None
        payload = __import__("json").loads(summary_path.read_text(encoding="utf-8"))
        return LoopSummary(**payload)

    def replay(self) -> LoopSummary:
        terminal = self._load_terminal_summary()
        if terminal is not None:
            return terminal
        if self.manifest.replay is None:
            raise ValueError("Replay mode requires a replay section in the loop manifest")
        self._initialize_incumbent()
        iteration_dir = self._iteration_dir()
        incumbent = self._aggregate_item(self.manifest.replay.incumbent)
        candidates = [self._aggregate_item(item) for item in self.manifest.replay.candidates]
        decision = decide_acceptance(incumbent, candidates, self.manifest.verification)
        previous = dict(self.state.incumbent)
        selected_item = self.manifest.replay.incumbent
        if decision.accepted:
            selected_item = next(
                item
                for item in self.manifest.replay.candidates
                if item.id == decision.selected_id
            )
            self.state.incumbent = {
                "id": selected_item.id,
                "checkpoint": str(selected_item.checkpoint),
                "checkpoint_sha256": hash_file(selected_item.checkpoint),
            }
        history_item = {
            "iteration": self.state.current_iteration,
            "accepted": decision.accepted,
            "previous_id": previous["id"],
            "previous_checkpoint_sha256": previous["checkpoint_sha256"],
            "selected_id": decision.selected_id,
            "selected_checkpoint_sha256": self.state.incumbent["checkpoint_sha256"],
            "reasons": list(decision.reasons),
        }
        self.state.history.append(history_item)
        completed_iteration = self.state.current_iteration
        self.state.current_iteration += 1
        self.state.status = "replay_completed"
        save_state(self.run_dir, self.state)
        atomic_write_json(
            iteration_dir / "selection.json",
            {
                "decision": decision.as_dict(),
                "incumbent": incumbent.as_dict(),
                "candidates": [candidate.as_dict() for candidate in candidates],
            },
        )
        iteration = IterationManifest(
            iteration=completed_iteration,
            status="accepted" if decision.accepted else "rolled_back",
            commands=(),
        )
        atomic_write_json(iteration_dir / "iteration_manifest.json", iteration.as_dict())
        summary = LoopSummary(
            status="replay_completed",
            iteration=completed_iteration,
            incumbent_id=self.state.incumbent["id"],
            accepted=decision.accepted,
            selected_id=decision.selected_id,
            run_dir=str(self.run_dir),
        )
        atomic_write_json(self.run_dir / "loop_summary.json", summary.as_dict())
        return summary

    def execute(self) -> LoopSummary:
        terminal = self._load_execute_summary()
        if terminal is not None:
            return terminal
        self._initialize_incumbent()
        self.state.status = "running"
        save_state(self.run_dir, self.state)
        last_summary: LoopSummary | None = None
        while self.state.current_iteration < self.manifest.search.max_iterations:
            last_summary = self._execute_iteration()
            if last_summary.status in {"strict_success", "no_improvement"}:
                return last_summary
        if last_summary is None:
            raise RuntimeError("AutoRefine loop has no executable iteration")
        summary = LoopSummary(
            status="budget_exhausted",
            iteration=last_summary.iteration,
            incumbent_id=self.state.incumbent["id"],
            accepted=last_summary.accepted,
            selected_id=last_summary.selected_id,
            run_dir=str(self.run_dir),
        )
        self.state.status = summary.status
        save_state(self.run_dir, self.state)
        atomic_write_json(self.run_dir / "loop_summary.json", summary.as_dict())
        return summary

    def _load_execute_summary(self) -> LoopSummary | None:
        summary_path = self.run_dir / "loop_summary.json"
        terminal_states = {"strict_success", "no_improvement", "budget_exhausted"}
        if self.state.status not in terminal_states or not summary_path.exists():
            return None
        return LoopSummary(**json.loads(summary_path.read_text(encoding="utf-8")))

    def _record_from_prepared(
        self, payload: dict[str, Any], *, requires_gpu: bool
    ) -> CommandRecord:
        command_id = f"iter{self.state.current_iteration:03d}/{payload['id']}"
        return CommandRecord(
            id=command_id,
            argv=tuple(str(item) for item in payload["argv"]),
            cwd=Path(payload["cwd"]),
            log_path=self._iteration_dir() / "logs" / f"{payload['id']}.log",
            expected_outputs=tuple(Path(path) for path in payload.get("expected_outputs", [])),
            env=self._runtime_env(),
            requires_gpu=requires_gpu,
            recoverable_exit_codes=tuple(
                int(code) for code in payload.get("recoverable_exit_codes", [])
            ),
            success_log_marker=payload.get("success_log_marker"),
        )

    def _runtime_env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        source_paths = [
            str(self.manifest.runtime.workdir),
            str(self.manifest.runtime.workdir / "src"),
        ]
        if os.environ.get("PYTHONPATH"):
            source_paths.append(os.environ["PYTHONPATH"])
        environment = {
            "PYTHONPATH": os.pathsep.join(source_paths),
            "WANDB_MODE": "offline",
            **dict(self.manifest.runtime.env),
        }
        if extra:
            environment.update(extra)
        return environment

    def _resume_result(self, record: CommandRecord) -> CommandResult | None:
        saved = self.state.completed_commands.get(record.id)
        if not saved or int(saved.get("exit_code", 1)) != 0:
            return None
        outputs = record.expected_outputs
        outputs_valid = bool(outputs) and all(
            path.is_file() and path.stat().st_size > 0 for path in outputs
        )
        checkpoint = saved.get("checkpoint")
        if not outputs_valid and not (checkpoint and Path(checkpoint).is_file()):
            return None
        return CommandResult(
            id=record.id,
            exit_code=0,
            skipped=True,
            gpu_id=saved.get("gpu_id"),
            log_path=record.log_path,
            outputs_valid=outputs_valid,
        )

    def _save_results(self, results: list[CommandResult]) -> None:
        for result in results:
            saved = result.as_dict()
            saved["expected_outputs"] = []
            self.state.completed_commands[result.id] = saved
        save_state(self.run_dir, self.state)

    def _run_batch(
        self,
        records: list[CommandRecord],
        *,
        gpu_ids: tuple[int, ...],
    ) -> list[CommandResult]:
        resolved: dict[str, CommandResult] = {}
        pending: list[CommandRecord] = []
        for record in records:
            resumed = self._resume_result(record)
            if resumed is not None:
                resolved[record.id] = resumed
            else:
                pending.append(record)
        if pending:
            completed = run_commands(
                pending,
                gpu_ids=gpu_ids,
                max_parallel=self.manifest.execution.max_parallel,
            )
            self._save_results(completed)
            resolved.update((result.id, result) for result in completed)
        results = [resolved[record.id] for record in records]
        failures = [result for result in results if result.exit_code != 0]
        if failures:
            failed = failures[0]
            self.state.status = "failed"
            save_state(self.run_dir, self.state)
            raise RuntimeError(f"Command failed ({failed.exit_code}): {failed.id}; log={failed.log_path}")
        return results

    def _run_one(self, record: CommandRecord) -> CommandResult:
        resumed = self._resume_result(record)
        if resumed is not None:
            return resumed
        result = run_command(record)
        self._save_results([result])
        if result.exit_code != 0:
            self.state.status = "failed"
            save_state(self.run_dir, self.state)
            raise RuntimeError(
                f"Command failed ({result.exit_code}): {result.id}; log={result.log_path}"
            )
        return result

    def _train_record(self, candidate: dict[str, Any]) -> CommandRecord:
        run_name = str(candidate["run_name"])
        incumbent_load_run = self._incumbent_load_run()
        argv = (
            str(self.manifest.runtime.python),
            str(self.manifest.runtime.train_script),
            self.manifest.runtime.dynamic_task,
            "--motion-file",
            str(self.manifest.task.motion_file),
            "--env.scene.num-envs",
            str(self.manifest.search.num_envs),
            "--agent.max-iterations",
            str(self.manifest.search.train_iterations),
            "--agent.save-interval",
            "25",
            "--agent.run-name",
            run_name,
            "--agent.resume",
            "True",
            "--agent.load-run",
            incumbent_load_run,
            "--agent.load-checkpoint",
            self._incumbent_checkpoint().name,
            "--agent.algorithm.learning-rate",
            str(self.manifest.search.learning_rate),
            "--agent.seed",
            str(self.manifest.search.train_seed),
        )
        return CommandRecord(
            id=f"iter{self.state.current_iteration:03d}/train_{candidate['id']}",
            argv=argv,
            cwd=self.manifest.runtime.workdir,
            log_path=self._iteration_dir() / "logs" / f"train_{candidate['id']}.log",
            env=self._runtime_env(
                {"MIMICX_AUTOREFINE_PATCH_FILE": str(candidate["patch_file"])}
            ),
            requires_gpu=True,
        )

    def _incumbent_load_run(self) -> str:
        checkpoint = self._incumbent_checkpoint()
        try:
            checkpoint.relative_to(self.manifest.runtime.checkpoint_root)
        except ValueError:
            return str(
                self.state.incumbent.get("load_run", self.manifest.task.load_run)
            )
        return checkpoint.parent.name

    def _discover_checkpoint(self, candidate: dict[str, Any], record: CommandRecord) -> Path:
        saved = self.state.completed_commands.get(record.id, {})
        saved_checkpoint = saved.get("checkpoint")
        template = self.manifest.runtime.checkpoint_glob
        patterns = [
            template.format(
                run_name=candidate["run_name"], candidate_id=candidate["id"]
            )
        ]
        if "{run_name}" in template and "*{run_name}*" not in template:
            patterns.append(
                template.replace("{run_name}", "*{run_name}*").format(
                    run_name=candidate["run_name"], candidate_id=candidate["id"]
                )
            )
        matches = [
            path
            for pattern in patterns
            for path in self.manifest.runtime.checkpoint_root.glob(pattern)
            if path.is_file()
        ]
        if saved_checkpoint and Path(saved_checkpoint).is_file():
            matches.append(Path(saved_checkpoint))
        if not matches:
            raise FileNotFoundError(
                f"No checkpoint for {candidate['id']} under "
                f"{self.manifest.runtime.checkpoint_root} with {patterns}"
            )
        checkpoint = select_highest_step_checkpoint(matches)
        saved["checkpoint"] = str(checkpoint.resolve())
        saved["checkpoint_sha256"] = hash_file(checkpoint)
        self.state.completed_commands[record.id] = saved
        save_state(self.run_dir, self.state)
        return checkpoint.resolve()

    def _recover_completed_training_checkpoints(
        self,
        candidates: list[dict[str, Any]],
        records: list[CommandRecord],
    ) -> None:
        for candidate, record in zip(candidates, records, strict=True):
            saved = self.state.completed_commands.get(record.id, {})
            if int(saved.get("exit_code", 1)) != 0:
                continue
            try:
                self._discover_checkpoint(candidate, record)
            except FileNotFoundError:
                continue

    def _candidate_rollout_record(
        self, candidate_id: str, checkpoint: Path, seed: int
    ) -> CommandRecord:
        output_dir = self._iteration_dir() / "eval" / candidate_id / f"seed_{seed}"
        payload = self._rollout_command(seed, checkpoint)
        payload["id"] = f"eval_{candidate_id}_seed{seed}"
        payload["expected_outputs"] = [
            str(output_dir / "metrics.json"),
            str(output_dir / "metrics_steps.csv"),
        ]
        argv = payload["argv"]
        metrics_index = argv.index("--metrics-output") + 1
        steps_index = argv.index("--step-metrics-output") + 1
        argv[metrics_index] = payload["expected_outputs"][0]
        argv[steps_index] = payload["expected_outputs"][1]
        return self._record_from_prepared(payload, requires_gpu=True)

    def _aggregate_records(
        self, candidate_id: str, records: list[CommandRecord]
    ) -> RepeatAggregate:
        rollout_records = [
            load_rollout_metrics(
                record.expected_outputs[0],
                horizon=self.manifest.task.horizon,
                metric_keys=self.manifest.verification.metric_keys,
                seed=seed,
            )
            for record, seed in zip(
                records,
                self.manifest.verification.seeds,
                strict=False,
            )
        ]
        return aggregate_repeats(candidate_id, rollout_records)

    def _execute_iteration(self) -> LoopSummary:
        iteration_index = self.state.current_iteration
        iteration_dir = self._iteration_dir()
        prepared = self.prepare()
        incumbent_payloads = list(prepared.commands[: self.manifest.verification.repeats])
        incumbent_records = [
            self._record_from_prepared(payload, requires_gpu=True)
            for payload in incumbent_payloads
        ]
        self._run_batch(incumbent_records, gpu_ids=self.eval_gpus)

        mine_payload = next(item for item in prepared.commands if item["id"] == "mine_failures")
        propose_payload = next(
            item for item in prepared.commands if item["id"] == "propose_candidates"
        )
        self._run_one(self._record_from_prepared(mine_payload, requires_gpu=False))
        self._run_one(self._record_from_prepared(propose_payload, requires_gpu=False))
        matrix_path = Path(propose_payload["expected_outputs"][0])
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        candidates = list(matrix["candidates"])[: self.manifest.search.candidate_budget]
        if not candidates:
            raise RuntimeError("Candidate proposal produced no candidates")

        train_records = [self._train_record(candidate) for candidate in candidates]
        self._recover_completed_training_checkpoints(candidates, train_records)
        self._run_batch(train_records, gpu_ids=self.candidate_gpus)
        checkpoints = {
            candidate["id"]: self._discover_checkpoint(candidate, record)
            for candidate, record in zip(candidates, train_records, strict=True)
        }
        eval_records: dict[str, list[CommandRecord]] = {}
        all_evals: list[CommandRecord] = []
        for candidate in candidates:
            records = [
                self._candidate_rollout_record(candidate["id"], checkpoints[candidate["id"]], seed)
                for seed in self.manifest.verification.seeds[: self.manifest.verification.repeats]
            ]
            eval_records[candidate["id"]] = records
            all_evals.extend(records)
        self._run_batch(all_evals, gpu_ids=self.eval_gpus)

        incumbent_id = str(self.state.incumbent["id"])
        incumbent = self._aggregate_records(incumbent_id, incumbent_records)
        aggregates = [
            self._aggregate_records(candidate["id"], eval_records[candidate["id"]])
            for candidate in candidates
        ]
        decision = decide_acceptance(incumbent, aggregates, self.manifest.verification)
        previous = dict(self.state.incumbent)
        selected_aggregate = incumbent
        if decision.accepted:
            selected_candidate = next(
                candidate for candidate in candidates if candidate["id"] == decision.selected_id
            )
            selected_aggregate = next(
                aggregate
                for aggregate in aggregates
                if aggregate.candidate_id == decision.selected_id
            )
            checkpoint = checkpoints[decision.selected_id]
            self.state.incumbent = {
                "id": decision.selected_id,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": hash_file(checkpoint),
                "load_run": checkpoint.parent.name,
            }
        strict_success = (
            selected_aggregate.zero_termination_repeats >= self.manifest.verification.repeats
            and selected_aggregate.worst_first_failure_step > self.manifest.task.horizon
        )
        status = "strict_success" if strict_success else (
            "running" if decision.accepted else "no_improvement"
        )
        history_item = {
            "iteration": iteration_index,
            "accepted": decision.accepted,
            "previous_id": previous["id"],
            "previous_checkpoint_sha256": previous["checkpoint_sha256"],
            "selected_id": decision.selected_id,
            "selected_checkpoint_sha256": self.state.incumbent["checkpoint_sha256"],
            "status": status,
            "reasons": list(decision.reasons),
        }
        self.state.history.append(history_item)
        self.state.current_iteration += 1
        self.state.status = status
        save_state(self.run_dir, self.state)
        atomic_write_json(
            iteration_dir / "selection.json",
            {
                "decision": decision.as_dict(),
                "incumbent": incumbent.as_dict(),
                "candidates": [aggregate.as_dict() for aggregate in aggregates],
            },
        )
        atomic_write_json(
            iteration_dir / "iteration_manifest.json",
            {
                "iteration": iteration_index,
                "status": status,
                "commands": [
                    record.as_dict()
                    for record in incumbent_records + train_records + all_evals
                ],
                "decision": decision.as_dict(),
            },
        )
        summary = LoopSummary(
            status=status,
            iteration=iteration_index,
            incumbent_id=self.state.incumbent["id"],
            accepted=decision.accepted,
            selected_id=decision.selected_id,
            run_dir=str(self.run_dir),
        )
        atomic_write_json(self.run_dir / "loop_summary.json", summary.as_dict())
        return summary
