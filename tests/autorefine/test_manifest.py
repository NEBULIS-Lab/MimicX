from __future__ import annotations

from pathlib import Path
import sys

import pytest
import yaml

from mimicx.refinement.manifest import candidate_id, hash_file, load_loop_manifest


def write_manifest(tmp_path: Path, **overrides: object) -> Path:
    motion = tmp_path / "motion.npz"
    checkpoint = tmp_path / "model.pt"
    train_script = tmp_path / "train.py"
    rollout_script = tmp_path / "rollout.py"
    for path in (motion, checkpoint, train_script, rollout_script):
        path.write_bytes(path.name.encode("ascii"))

    payload: dict[str, object] = {
        "schema": "mimicx.autorefine-loop.v1",
        "task": {
            "id": "fixture-task",
            "motion_file": str(motion),
            "base_checkpoint": str(checkpoint),
            "load_run": "fixture-run",
            "strict_task": "Fixture-Strict-Task",
            "horizon": 12,
        },
        "runtime": {
            "workdir": str(tmp_path),
            "python": sys.executable,
            "train_script": str(train_script),
            "rollout_script": str(rollout_script),
            "dynamic_task": "Fixture-Dynamic-Task",
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
            "metric_keys": ["body_pos_error_max", "ee_z_error_max"],
            "guards": {
                "body_pos_error_max": {
                    "direction": "lower",
                    "relative_tolerance": 0.1,
                }
            },
        },
        "execution": {"max_parallel": 2},
    }
    payload.update(overrides)
    path = tmp_path / "loop.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_load_loop_manifest_resolves_inputs_and_freezes_values(tmp_path: Path) -> None:
    path = write_manifest(tmp_path)

    manifest = load_loop_manifest(path)

    assert manifest.schema == "mimicx.autorefine-loop.v1"
    assert manifest.task.id == "fixture-task"
    assert manifest.task.motion_file == (tmp_path / "motion.npz").resolve()
    assert manifest.task.horizon == 12
    assert manifest.search.candidate_gpus == (4, 5)
    assert manifest.search.train_seed == 202
    assert manifest.runtime.python == Path(sys.executable).absolute()
    assert manifest.verification.seeds == (11, 22, 33)
    with pytest.raises(AttributeError):
        manifest.task.id = "changed"  # type: ignore[misc]


def test_load_loop_manifest_rejects_unknown_schema(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, schema="mimicx.autorefine-loop.v0")

    with pytest.raises(ValueError, match="schema"):
        load_loop_manifest(path)


def test_load_loop_manifest_names_missing_required_field(tmp_path: Path) -> None:
    path = write_manifest(tmp_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    del payload["task"]["horizon"]
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="task.horizon"):
        load_loop_manifest(path)


def test_load_loop_manifest_rejects_missing_input(tmp_path: Path) -> None:
    path = write_manifest(tmp_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["task"]["motion_file"] = str(tmp_path / "missing-motion.npz")
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="task.motion_file"):
        load_loop_manifest(path)


def test_hash_file_is_sha256_and_changes_with_content(tmp_path: Path) -> None:
    path = tmp_path / "input.bin"
    path.write_bytes(b"first")
    first = hash_file(path)
    path.write_bytes(b"second")

    assert len(first) == 64
    assert first != hash_file(path)


def test_candidate_id_is_independent_of_mapping_order() -> None:
    patch_a = {"sampling": {"end": 20, "start": 10}, "weight": 1.25}
    patch_b = {"weight": 1.25, "sampling": {"start": 10, "end": 20}}

    assert candidate_id(3, patch_a) == candidate_id(3, patch_b)
    assert candidate_id(3, patch_a).startswith("iter003_")
    assert candidate_id(4, patch_a) != candidate_id(3, patch_a)
