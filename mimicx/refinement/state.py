"""Crash-safe persistent state for AutoRefine runs."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


STATE_SCHEMA = "mimicx.autorefine-state.v1"


@dataclass
class LoopState:
    manifest_hash: str
    current_iteration: int = 0
    status: str = "initialized"
    incumbent: dict[str, Any] = field(default_factory=dict)
    completed_commands: dict[str, dict[str, Any]] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    schema: str = STATE_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _state_from_payload(payload: Mapping[str, Any]) -> LoopState:
    if payload.get("schema") != STATE_SCHEMA:
        raise ValueError(f"Unsupported state schema: {payload.get('schema')!r}")
    return LoopState(
        manifest_hash=str(payload["manifest_hash"]),
        current_iteration=int(payload.get("current_iteration", 0)),
        status=str(payload.get("status", "initialized")),
        incumbent=dict(payload.get("incumbent", {})),
        completed_commands=dict(payload.get("completed_commands", {})),
        history=list(payload.get("history", [])),
    )


def save_state(run_dir: Path, state: LoopState) -> None:
    atomic_write_json(run_dir / "state.json", state.as_dict())


def load_or_create_state(run_dir: Path, manifest_hash: str) -> LoopState:
    root = run_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.json"
    if state_path.exists():
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        state = _state_from_payload(payload)
        if state.manifest_hash != manifest_hash:
            raise ValueError(
                "Cannot resume: manifest hash does not match existing state "
                f"({state.manifest_hash} != {manifest_hash})"
            )
        return state
    state = LoopState(manifest_hash=manifest_hash)
    save_state(root, state)
    return state
