"""Deterministic local process execution for AutoRefine commands."""

from __future__ import annotations

import os
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class CommandRecord:
    id: str
    argv: tuple[str, ...]
    cwd: Path
    log_path: Path
    expected_outputs: tuple[Path, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    requires_gpu: bool = False
    recoverable_exit_codes: tuple[int, ...] = ()
    success_log_marker: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["argv"] = list(self.argv)
        payload["cwd"] = str(self.cwd)
        payload["log_path"] = str(self.log_path)
        payload["expected_outputs"] = [str(path) for path in self.expected_outputs]
        return payload


@dataclass(frozen=True)
class CommandResult:
    id: str
    exit_code: int
    skipped: bool
    gpu_id: int | None
    log_path: Path
    outputs_valid: bool
    process_exit_code: int | None = None
    recovered_from_process_failure: bool = False

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["log_path"] = str(self.log_path)
        return payload


def _outputs_valid(paths: Sequence[Path]) -> bool:
    return bool(paths) and all(path.is_file() and path.stat().st_size > 0 for path in paths)


def run_command(
    record: CommandRecord,
    *,
    gpu_id: int | None = None,
    dry_run: bool = False,
) -> CommandResult:
    if _outputs_valid(record.expected_outputs):
        return CommandResult(
            id=record.id,
            exit_code=0,
            skipped=True,
            gpu_id=gpu_id,
            log_path=record.log_path,
            outputs_valid=True,
            process_exit_code=None,
        )
    record.log_path.parent.mkdir(parents=True, exist_ok=True)
    for output in record.expected_outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        record.log_path.write_text(
            f"DRY_RUN cwd={record.cwd}\n{shlex.join(record.argv)}\n", encoding="utf-8"
        )
        return CommandResult(
            id=record.id,
            exit_code=0,
            skipped=True,
            gpu_id=gpu_id,
            log_path=record.log_path,
            outputs_valid=_outputs_valid(record.expected_outputs),
            process_exit_code=None,
        )

    environment = os.environ.copy()
    environment.update({str(key): str(value) for key, value in record.env.items()})
    if record.requires_gpu:
        if gpu_id is None:
            raise ValueError(f"GPU command {record.id} has no assigned GPU")
        environment["MIMICX_DEVICE_ID"] = str(gpu_id)
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            environment["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    with record.log_path.open("w", encoding="utf-8") as log:
        log.write(f"cwd={record.cwd}\ncommand={shlex.join(record.argv)}\n")
        log.flush()
        completed = subprocess.run(
            record.argv,
            cwd=record.cwd,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    valid = _outputs_valid(record.expected_outputs)
    process_exit_code = completed.returncode
    exit_code = process_exit_code
    recovered = False
    if (
        process_exit_code in record.recoverable_exit_codes
        and valid
        and record.success_log_marker
        and record.success_log_marker in record.log_path.read_text(encoding="utf-8")
    ):
        exit_code = 0
        recovered = True
        with record.log_path.open("a", encoding="utf-8") as log:
            log.write(
                "recovered process cleanup failure: "
                f"exit_code={process_exit_code}; outputs and success marker are valid\n"
            )
    if exit_code == 0 and record.expected_outputs and not valid:
        exit_code = 66
        with record.log_path.open("a", encoding="utf-8") as log:
            log.write("expected outputs are missing or empty\n")
    return CommandResult(
        id=record.id,
        exit_code=exit_code,
        skipped=False,
        gpu_id=gpu_id,
        log_path=record.log_path,
        outputs_valid=valid,
        process_exit_code=process_exit_code,
        recovered_from_process_failure=recovered,
    )


def run_commands(
    records: Sequence[CommandRecord],
    gpu_ids: Sequence[int],
    max_parallel: int,
) -> list[CommandResult]:
    if max_parallel <= 0:
        raise ValueError("max_parallel must be positive")
    assignments: list[int | None] = []
    gpu_index = 0
    for record in records:
        if record.requires_gpu:
            if not gpu_ids:
                raise ValueError("At least one GPU ID is required for GPU commands")
            assignments.append(int(gpu_ids[gpu_index % len(gpu_ids)]))
            gpu_index += 1
        else:
            assignments.append(None)
    device_locks = {gpu_id: Lock() for gpu_id in assignments if gpu_id is not None}

    def execute(record: CommandRecord, gpu_id: int | None) -> CommandResult:
        if gpu_id is None:
            return run_command(record)
        with device_locks[gpu_id]:
            return run_command(record, gpu_id=gpu_id)

    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = [
            pool.submit(execute, record, gpu_id)
            for record, gpu_id in zip(records, assignments, strict=True)
        ]
        return [future.result() for future in futures]
