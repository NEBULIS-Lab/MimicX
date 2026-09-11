"""Durable, deduplicated progress records for the running paper matrix."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class ProgressRecordResult:
    appended: bool
    completed_jobs: int
    pending_jobs: int


def _fingerprint(payload: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in payload.items() if key != "timestamp"}


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def record_matrix_progress(
    status_json: Path,
    latest_path: Path,
    history_path: Path,
    *,
    active_sessions: Sequence[str],
    timestamp: str,
) -> ProgressRecordResult:
    status = json.loads(status_json.read_text(encoding="utf-8"))
    payload: dict[str, object] = {
        "timestamp": timestamp,
        "total_jobs": int(status["total_jobs"]),
        "completed_jobs": int(status["completed_jobs"]),
        "pending_jobs": int(status["pending_jobs"]),
        "active_sessions": sorted(str(item) for item in active_sessions),
    }
    previous = None
    if latest_path.is_file():
        previous = json.loads(latest_path.read_text(encoding="utf-8"))
    appended = previous is None or _fingerprint(previous) != _fingerprint(payload)
    _atomic_json(latest_path, payload)
    if appended:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return ProgressRecordResult(
        appended=appended,
        completed_jobs=int(payload["completed_jobs"]),
        pending_jobs=int(payload["pending_jobs"]),
    )
