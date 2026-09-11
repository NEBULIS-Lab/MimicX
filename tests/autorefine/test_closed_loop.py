from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import yaml

from mimicx.refinement.closed_loop import ClosedLoopCoordinator
from mimicx.refinement.manifest import hash_file, load_loop_manifest


def write_metrics(
    path: Path,
    *,
    done_count: int,
    first_failure: int,
    body: float,
    reward: float,
) -> None:
    path.write_text(
        json.dumps(
            {
                "steps": 100,
                "done_any_count": done_count,
                "first_failure_step": first_failure,
                "reward_mean_avg": reward,
                "body_pos_error_max": body,
            }
        ),
        encoding="utf-8",
    )


def write_loop_manifest(
    tmp_path: Path,
    *,
    candidate_frontier: int,
    candidate_body: float,
    executable_runtime: bool = False,
    timestamp_checkpoint_dirs: bool = False,
    fail_repeated_training: bool = False,
) -> Path:
    motion = tmp_path / "motion.npz"
    incumbent_checkpoint = tmp_path / "incumbent.pt"
    candidate_checkpoint = tmp_path / "candidate.pt"
    train_script = tmp_path / "train.py"
    rollout_script = tmp_path / "rollout.py"
    for path in (motion, incumbent_checkpoint, candidate_checkpoint, train_script, rollout_script):
        path.write_bytes(path.name.encode("ascii"))
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    if executable_runtime:
        train_script.write_text(
            """\
import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("task")
parser.add_argument("--agent.run-name", required=True)
args, _ = parser.parse_known_args()
root = Path(os.environ["MIMICX_TEST_CHECKPOINT_ROOT"])
run_name = args.__dict__["agent.run_name"]
if os.environ.get("MIMICX_TEST_TIMESTAMP_CHECKPOINT_DIRS") == "1":
    run_name = f"2026-08-15_06-17-00_{run_name}"
checkpoint = root / run_name / "model_20.pt"
checkpoint.parent.mkdir(parents=True, exist_ok=True)
if checkpoint.exists() and os.environ.get("MIMICX_TEST_FAIL_REPEATED_TRAINING") == "1":
    raise SystemExit("training command was repeated")
checkpoint.write_text(args.__dict__["agent.run_name"])
""",
            encoding="utf-8",
        )
        rollout_script.write_text(
            """\
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint-file", type=Path, required=True)
parser.add_argument("--metrics-output", type=Path, required=True)
parser.add_argument("--step-metrics-output", type=Path, required=True)
parser.add_argument("--steps", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
args, _ = parser.parse_known_args()
name = args.checkpoint_file.read_text()
if "auto_01_" in name:
    done, first_failure, body, reward = 0, args.steps + 1, 0.19, 1.4
elif "auto_02_" in name:
    done, first_failure, body, reward = 1, 60, 0.23, 1.2
elif "auto_03_" in name:
    done, first_failure, body, reward = 1, 50, 0.24, 1.1
else:
    done, first_failure, body, reward = 1, 70, 0.20, 1.0
args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
args.metrics_output.write_text(json.dumps({
    "steps": args.steps,
    "done_any_count": done,
    "first_failure_step": first_failure,
    "reward_mean_avg": reward,
    "body_pos_error_max": body,
}))
rows = ["step,done_any,body_pos_error_max,ee_z_error_max"]
for step in range(1, args.steps + 1):
    rows.append(f"{step},{str(done > 0 and step == first_failure).lower()},{body},0.1")
args.step_metrics_output.write_text("\\n".join(rows) + "\\n")
""",
            encoding="utf-8",
        )

    incumbent_metrics = []
    candidate_metrics = []
    for index, seed in enumerate((11, 22, 33)):
        incumbent_path = tmp_path / f"incumbent_seed{seed}.json"
        candidate_path = tmp_path / f"candidate_seed{seed}.json"
        write_metrics(
            incumbent_path,
            done_count=1,
            first_failure=70 + index,
            body=0.20,
            reward=1.0,
        )
        write_metrics(
            candidate_path,
            done_count=1,
            first_failure=candidate_frontier + index,
            body=candidate_body,
            reward=1.2,
        )
        incumbent_metrics.append(str(incumbent_path))
        candidate_metrics.append(str(candidate_path))

    payload = {
        "schema": "mimicx.autorefine-loop.v1",
        "task": {
            "id": "fixture-task",
            "motion_file": str(motion),
            "base_checkpoint": str(incumbent_checkpoint),
            "load_run": "fixture-run",
            "strict_task": "Fixture-Strict-Task",
            "horizon": 100,
        },
        "runtime": {
            "workdir": str(tmp_path),
            "python": sys.executable,
            "train_script": str(train_script),
            "rollout_script": str(rollout_script),
            "dynamic_task": "Fixture-Dynamic-Task",
            "checkpoint_root": str(checkpoint_root),
            "checkpoint_glob": "**/{run_name}/model_*.pt",
            "env": {
                "MIMICX_TEST_CHECKPOINT_ROOT": str(checkpoint_root),
                "MIMICX_TEST_TIMESTAMP_CHECKPOINT_DIRS": (
                    "1" if timestamp_checkpoint_dirs else "0"
                ),
                "MIMICX_TEST_FAIL_REPEATED_TRAINING": (
                    "1" if fail_repeated_training else "0"
                ),
            },
        },
        "search": {
            "max_iterations": 2,
            "candidate_budget": 3,
            "train_iterations": 20,
            "num_envs": 4,
            "learning_rate": 2.0e-6,
            "train_seed": 202,
            "candidate_gpus": [4, 5],
            "eval_gpus": [4, 5],
        },
        "verification": {
            "repeats": 3,
            "seeds": [11, 22, 33],
            "metric_keys": ["body_pos_error_max"],
            "guards": {
                "body_pos_error_max": {
                    "direction": "lower",
                    "relative_tolerance": 0.1,
                }
            },
        },
        "execution": {"max_parallel": 2},
        "replay": {
            "incumbent": {
                "id": "incumbent",
                "checkpoint": str(incumbent_checkpoint),
                "metrics": incumbent_metrics,
            },
            "candidates": [
                {
                    "id": "candidate-a",
                    "checkpoint": str(candidate_checkpoint),
                    "metrics": candidate_metrics,
                }
            ],
        },
    }
    manifest_path = tmp_path / "loop.yaml"
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return manifest_path


def test_prepare_writes_commands_without_executing_them(tmp_path: Path) -> None:
    manifest = load_loop_manifest(
        write_loop_manifest(tmp_path, candidate_frontier=90, candidate_body=0.21)
    )
    run_dir = tmp_path / "prepared-run"

    iteration = ClosedLoopCoordinator(manifest, run_dir).prepare()

    ids = [command["id"] for command in iteration.commands]
    assert ids[:3] == [
        "incumbent_rollout_seed11",
        "incumbent_rollout_seed22",
        "incumbent_rollout_seed33",
    ]
    assert "mine_failures" in ids
    assert "propose_candidates" in ids
    proposal = next(command for command in iteration.commands if command["id"] == "propose_candidates")
    prefix_index = proposal["argv"].index("--run-prefix") + 1
    assert proposal["argv"][prefix_index] == "mimicx_fixture-task_seed202_iter000"
    assert (run_dir / "loop_manifest.resolved.json").is_file()
    assert (run_dir / "iterations" / "iter_000" / "iteration_manifest.json").is_file()
    assert not list(run_dir.rglob("*.log"))


def test_replay_rejection_keeps_incumbent_checkpoint_hash(tmp_path: Path) -> None:
    manifest = load_loop_manifest(
        write_loop_manifest(tmp_path, candidate_frontier=90, candidate_body=0.40)
    )
    run_dir = tmp_path / "rejected-run"
    original_hash = hash_file(manifest.task.base_checkpoint)

    summary = ClosedLoopCoordinator(manifest, run_dir).replay()

    assert summary.accepted is False
    assert summary.selected_id == "incumbent"
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert state["incumbent"]["checkpoint_sha256"] == original_hash
    assert state["incumbent"]["id"] == "incumbent"


def test_replay_acceptance_advances_immutable_incumbent_record(tmp_path: Path) -> None:
    manifest = load_loop_manifest(
        write_loop_manifest(tmp_path, candidate_frontier=90, candidate_body=0.21)
    )
    run_dir = tmp_path / "accepted-run"

    summary = ClosedLoopCoordinator(manifest, run_dir).replay()

    assert summary.accepted is True
    assert summary.selected_id == "candidate-a"
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert state["incumbent"]["id"] == "candidate-a"
    assert state["incumbent"]["checkpoint_sha256"] == hash_file(tmp_path / "candidate.pt")
    assert state["history"][0]["previous_checkpoint_sha256"] == hash_file(
        tmp_path / "incumbent.pt"
    )


def test_replay_second_invocation_resumes_terminal_iteration(tmp_path: Path) -> None:
    manifest = load_loop_manifest(
        write_loop_manifest(tmp_path, candidate_frontier=90, candidate_body=0.21)
    )
    run_dir = tmp_path / "resume-run"
    first = ClosedLoopCoordinator(manifest, run_dir).replay()
    state_before = (run_dir / "state.json").read_bytes()

    second = ClosedLoopCoordinator(manifest, run_dir).replay()

    assert second == first
    assert (run_dir / "state.json").read_bytes() == state_before
    assert len(json.loads(state_before)["history"]) == 1


def test_execute_runs_full_candidate_cycle_and_stops_on_strict_success(tmp_path: Path) -> None:
    manifest = load_loop_manifest(
        write_loop_manifest(
            tmp_path,
            candidate_frontier=90,
            candidate_body=0.21,
            executable_runtime=True,
        )
    )
    run_dir = tmp_path / "execute-run"

    summary = ClosedLoopCoordinator(manifest, run_dir).execute()

    assert summary.accepted is True
    assert summary.selected_id.startswith("auto_01_")
    assert summary.status == "strict_success"
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert state["incumbent"]["id"] == summary.selected_id
    assert len(state["completed_commands"]) == 17
    selection = json.loads(
        (run_dir / "iterations" / "iter_000" / "selection.json").read_text(
            encoding="utf-8"
        )
    )
    assert selection["decision"]["accepted"] is True
    assert len(selection["candidates"]) == 3
    train_logs = sorted((run_dir / "iterations" / "iter_000" / "logs").glob("train_*.log"))
    assert train_logs
    assert all("--agent.seed 202" in path.read_text(encoding="utf-8") for path in train_logs)


def test_execute_discovers_mjlab_timestamp_prefixed_checkpoint_dirs(
    tmp_path: Path,
) -> None:
    manifest = load_loop_manifest(
        write_loop_manifest(
            tmp_path,
            candidate_frontier=90,
            candidate_body=0.21,
            executable_runtime=True,
            timestamp_checkpoint_dirs=True,
        )
    )

    summary = ClosedLoopCoordinator(manifest, tmp_path / "timestamped-run").execute()

    assert summary.accepted is True
    assert summary.selected_id.startswith("auto_01_")
    assert summary.status == "strict_success"


def test_runtime_gpu_override_is_used_for_closed_loop_commands(tmp_path: Path) -> None:
    manifest = load_loop_manifest(
        write_loop_manifest(tmp_path, candidate_frontier=90, candidate_body=0.21)
    )
    run_dir = tmp_path / "runtime-gpu-run"

    iteration = ClosedLoopCoordinator(
        manifest, run_dir, runtime_gpu_ids=(7,)
    ).prepare()

    rollout = next(command for command in iteration.commands if command["id"].startswith("incumbent_rollout"))
    assert rollout["gpu_pool"] == [7]
    resolved = json.loads((run_dir / "loop_manifest.resolved.json").read_text())
    assert resolved["search"]["candidate_gpus"] == [7]
    assert resolved["search"]["eval_gpus"] == [7]


def test_execute_resumes_successful_training_after_checkpoint_discovery_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = load_loop_manifest(
        write_loop_manifest(
            tmp_path,
            candidate_frontier=90,
            candidate_body=0.21,
            executable_runtime=True,
            timestamp_checkpoint_dirs=True,
            fail_repeated_training=True,
        )
    )
    run_dir = tmp_path / "interrupted-run"
    interrupted = ClosedLoopCoordinator(manifest, run_dir)

    def fail_discovery(*_args: object, **_kwargs: object) -> Path:
        raise FileNotFoundError("simulated interruption after successful training")

    monkeypatch.setattr(interrupted, "_discover_checkpoint", fail_discovery)
    with pytest.raises(FileNotFoundError, match="simulated interruption"):
        interrupted.execute()

    resumed = ClosedLoopCoordinator(manifest, run_dir).execute()

    assert resumed.accepted is True
    assert resumed.status == "strict_success"


def test_checkpoint_discovery_prefers_highest_training_step_over_newer_mtime(
    tmp_path: Path,
) -> None:
    manifest = load_loop_manifest(
        write_loop_manifest(tmp_path, candidate_frontier=90, candidate_body=0.21)
    )
    coordinator = ClosedLoopCoordinator(manifest, tmp_path / "step-priority-run")
    candidate = {
        "id": "auto_01_window_local_global",
        "run_name": "mimicx_fixture_seed202_iter000_auto_01_window_local_global",
        "patch_file": str(tmp_path / "patch.json"),
    }
    record = coordinator._train_record(candidate)
    complete = (
        manifest.runtime.checkpoint_root
        / f"2026-08-15_06-17-00_{candidate['run_name']}"
        / "model_1248.pt"
    )
    interrupted = (
        manifest.runtime.checkpoint_root
        / f"2026-08-15_07-08-00_{candidate['run_name']}"
        / "model_1000.pt"
    )
    complete.parent.mkdir(parents=True)
    complete.write_bytes(b"complete")
    interrupted.parent.mkdir(parents=True)
    interrupted.write_bytes(b"newer but incomplete")
    os.utime(complete, ns=(1_000_000_000, 1_000_000_000))
    os.utime(interrupted, ns=(2_000_000_000, 2_000_000_000))

    discovered = coordinator._discover_checkpoint(candidate, record)

    assert discovered == complete.resolve()


def test_next_iteration_loads_timestamped_incumbent_run_directory(tmp_path: Path) -> None:
    manifest = load_loop_manifest(
        write_loop_manifest(tmp_path, candidate_frontier=90, candidate_body=0.21)
    )
    coordinator = ClosedLoopCoordinator(manifest, tmp_path / "continuation-run")
    run_dir = manifest.runtime.checkpoint_root / "2026-08-23_20-57-24_candidate"
    checkpoint = run_dir / "model_2318.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    coordinator.state.incumbent = {
        "id": "accepted-candidate",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hash_file(checkpoint),
        "load_run": "candidate",
    }
    candidate = {
        "id": "auto_01_window_local_global",
        "run_name": "mimicx_fixture_seed202_iter001_auto_01_window_local_global",
        "patch_file": str(tmp_path / "patch.json"),
    }

    record = coordinator._train_record(candidate)
    load_run_index = record.argv.index("--agent.load-run") + 1

    assert record.argv[load_run_index] == run_dir.name
