"""Planning and persistent state for the native video completion campaign."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from mimicx.refinement.state import atomic_write_json


TERMINAL = {
    "completed",
    "protected_rollback",
    "reference_qa_rejected",
    "infrastructure_failed",
}


@dataclass(frozen=True)
class PipelineStage:
    id: str
    kind: str
    task_id: str | None
    gpu_id: int | None
    depends_on: tuple[str, ...]
    admission_inputs: tuple[str, ...]
    max_retries: int = 2


def _sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_stage_plan(
    resolved_sources: Path,
    run_dir: Path,
    gpu_ids: Sequence[int],
) -> tuple[PipelineStage, ...]:
    if not gpu_ids:
        raise ValueError("at least one video GPU is required")
    payload = json.loads(resolved_sources.read_text(encoding="utf-8"))
    if payload.get("schema") != "mimicx.resolved-video-sources.v1":
        raise ValueError("unsupported resolved video source schema")
    stages: list[PipelineStage] = []
    qa_ids: list[str] = []
    new_index = 0
    for record in payload["records"]:
        role = str(record.get("role", ""))
        if not role.startswith("new_confirmation"):
            continue
        task_id = str(record["id"])
        gpu = int(gpu_ids[new_index % len(gpu_ids)])
        new_index += 1
        admission = (
            str(record["materialized_path"]),
            str(record["source_sha256"]),
        )
        gvhmr = f"{task_id}__gvhmr"
        retarget = f"{task_id}__retarget"
        qa = f"{task_id}__reference_qa"
        stages.extend(
            [
                PipelineStage(gvhmr, "gvhmr", task_id, gpu, (), admission),
                PipelineStage(retarget, "retarget", task_id, gpu, (gvhmr,), admission),
                PipelineStage(qa, "reference_qa", task_id, gpu, (retarget,), admission),
            ]
        )
        qa_ids.append(qa)
    if not qa_ids:
        raise ValueError("resolved source catalog has no new confirmation tasks")
    stages.extend(
        [
            PipelineStage("prepare_matrix", "prepare_matrix", None, None, tuple(qa_ids), ()),
            PipelineStage("launch_matrix", "launch_matrix", None, None, ("prepare_matrix",), ()),
            PipelineStage("summarize_matrix", "summarize_matrix", None, None, ("launch_matrix",), ()),
        ]
    )
    return tuple(stages)


def write_initial_state(
    plan: Sequence[PipelineStage],
    state_path: Path,
) -> dict[str, Any]:
    payload = {
        "schema": "mimicx.video-pipeline-state.v1",
        "plan_sha256": _sha256_json([asdict(stage) for stage in plan]),
        "stages": {
            stage.id: {
                "status": "pending",
                "attempts": 0,
                "max_retries": stage.max_retries,
                "gpu_id": stage.gpu_id,
                "kind": stage.kind,
                "task_id": stage.task_id,
                "exit_code": None,
            }
            for stage in plan
        },
    }
    atomic_write_json(state_path, payload)
    return payload


def runnable_stages(
    plan: Sequence[PipelineStage],
    state: Mapping[str, Any],
) -> tuple[PipelineStage, ...]:
    stage_state = state["stages"]
    ready: list[PipelineStage] = []
    for stage in plan:
        status = stage_state[stage.id]["status"]
        if status != "pending":
            continue
        dependency_states = [stage_state[item]["status"] for item in stage.depends_on]
        if any(value in {"infrastructure_failed", "reference_qa_rejected"} for value in dependency_states):
            continue
        if all(value in TERMINAL for value in dependency_states):
            ready.append(stage)
    return tuple(ready)


def select_nonconflicting_stages(
    ready: Sequence[PipelineStage],
) -> tuple[PipelineStage, ...]:
    """Select one ready stage per pinned GPU while retaining CPU-only stages."""
    selected: list[PipelineStage] = []
    active_gpus: set[int] = set()
    for stage in ready:
        if stage.gpu_id is not None:
            if stage.gpu_id in active_gpus:
                continue
            active_gpus.add(stage.gpu_id)
        selected.append(stage)
    return tuple(selected)


def load_or_initialize_state(
    plan: Sequence[PipelineStage],
    state_path: Path,
) -> dict[str, Any]:
    expected = _sha256_json([asdict(stage) for stage in plan])
    if not state_path.is_file():
        return write_initial_state(plan, state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("plan_sha256") != expected:
        raise ValueError("existing video pipeline state belongs to a different plan")
    return state


def write_plan(plan: Sequence[PipelineStage], path: Path) -> None:
    atomic_write_json(
        path,
        {
            "schema": "mimicx.video-pipeline-plan.v1",
            "plan_sha256": _sha256_json([asdict(stage) for stage in plan]),
            "stages": [asdict(stage) for stage in plan],
        },
    )
