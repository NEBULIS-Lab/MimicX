from __future__ import annotations

import json
import signal
import sys
from pathlib import Path

import pytest

from mimicx.refinement.executor import CommandRecord, run_command, run_commands
from mimicx.refinement.state import atomic_write_json, load_or_create_state, save_state


def test_atomic_write_json_replaces_complete_document(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    atomic_write_json(path, {"version": 1, "items": [1, 2]})
    atomic_write_json(path, {"version": 2, "items": [3]})

    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 2, "items": [3]}
    assert list(tmp_path.glob(".state.json.*.tmp")) == []


def test_state_resume_requires_matching_manifest_hash(tmp_path: Path) -> None:
    state = load_or_create_state(tmp_path, "abc123")
    state.current_iteration = 2
    state.status = "running"
    save_state(tmp_path, state)

    resumed = load_or_create_state(tmp_path, "abc123")

    assert resumed.current_iteration == 2
    assert resumed.status == "running"
    with pytest.raises(ValueError, match="manifest hash"):
        load_or_create_state(tmp_path, "different")


def test_completed_command_with_nonempty_output_is_skipped(tmp_path: Path) -> None:
    output = tmp_path / "already_done.txt"
    output.write_text("complete\n", encoding="utf-8")
    marker = tmp_path / "should_not_exist.txt"
    record = CommandRecord(
        id="skip-me",
        argv=(sys.executable, "-c", f"open({str(marker)!r}, 'w').write('ran')"),
        cwd=tmp_path,
        log_path=tmp_path / "skip.log",
        expected_outputs=(output,),
    )

    result = run_command(record)

    assert result.skipped is True
    assert result.exit_code == 0
    assert not marker.exists()


def test_failed_command_retains_log_and_exit_code(tmp_path: Path) -> None:
    record = CommandRecord(
        id="fails",
        argv=(sys.executable, "-c", "import sys; print('failure-detail'); sys.exit(7)"),
        cwd=tmp_path,
        log_path=tmp_path / "fails.log",
    )

    result = run_command(record)

    assert result.exit_code == 7
    assert result.skipped is False
    assert "failure-detail" in record.log_path.read_text(encoding="utf-8")


def test_post_output_cleanup_signal_is_recovered_for_explicit_rollout(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    steps = tmp_path / "metrics_steps.csv"
    script = (
        "import json, os, pathlib, signal; "
        f"pathlib.Path({str(metrics)!r}).write_text(json.dumps({{'steps': 12}})); "
        f"pathlib.Path({str(steps)!r}).write_text('step,reward\\n1,1.0\\n'); "
        "print('rollout_ok steps=12', flush=True); "
        "os.kill(os.getpid(), signal.SIGSEGV)"
    )
    record = CommandRecord(
        id="rollout-cleanup-signal",
        argv=(sys.executable, "-c", script),
        cwd=tmp_path,
        log_path=tmp_path / "rollout.log",
        expected_outputs=(metrics, steps),
        recoverable_exit_codes=(-signal.SIGSEGV,),
        success_log_marker="rollout_ok steps=",
    )

    result = run_command(record)

    assert result.exit_code == 0
    assert result.process_exit_code == -signal.SIGSEGV
    assert result.recovered_from_process_failure is True


def test_cleanup_signal_without_success_marker_remains_failed(tmp_path: Path) -> None:
    output = tmp_path / "partial.txt"
    script = (
        "import os, pathlib, signal; "
        f"pathlib.Path({str(output)!r}).write_text('partial'); "
        "os.kill(os.getpid(), signal.SIGSEGV)"
    )
    record = CommandRecord(
        id="incomplete-rollout",
        argv=(sys.executable, "-c", script),
        cwd=tmp_path,
        log_path=tmp_path / "incomplete.log",
        expected_outputs=(output,),
        recoverable_exit_codes=(-signal.SIGSEGV,),
        success_log_marker="rollout_ok steps=",
    )

    result = run_command(record)

    assert result.exit_code == -signal.SIGSEGV
    assert result.recovered_from_process_failure is False


def test_run_commands_assigns_gpus_deterministically(tmp_path: Path) -> None:
    records = []
    outputs = []
    for index in range(4):
        output = tmp_path / f"gpu_{index}.txt"
        outputs.append(output)
        records.append(
            CommandRecord(
                id=f"job-{index}",
                argv=(
                    sys.executable,
                    "-c",
                    (
                        "import os, pathlib; "
                        f"pathlib.Path({str(output)!r}).write_text(os.environ['MIMICX_DEVICE_ID'])"
                    ),
                ),
                cwd=tmp_path,
                log_path=tmp_path / f"job_{index}.log",
                expected_outputs=(output,),
                requires_gpu=True,
            )
        )

    results = run_commands(records, gpu_ids=(4, 5), max_parallel=2)

    assert [result.gpu_id for result in results] == [4, 5, 4, 5]
    assert [path.read_text(encoding="utf-8") for path in outputs] == ["4", "5", "4", "5"]
