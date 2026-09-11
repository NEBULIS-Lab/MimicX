"""Terminal-state checks for the paper evidence completion campaign."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def campaign_status(paths: Mapping[str, Path]) -> dict[str, object]:
    states: dict[str, str] = {}
    payloads: dict[str, object] = {}
    for name, path in paths.items():
        if not path.is_file():
            states[name] = "pending"
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            states[name] = "invalid"
            continue
        payloads[name] = payload
        if name == "video8":
            statuses = list(payload.get("stages", {}).values())
            states[name] = "completed" if statuses and all(item in {"completed", "protected_rollback"} for item in statuses) else "failed"
        elif name == "sonic":
            states[name] = "completed" if payload.get("completed") == payload.get("expected") else "failed"
        elif name == "hloop":
            states[name] = "completed" if payload.get("decision_parity") and int(payload.get("repetitions", 0)) >= 5 else "failed"
        elif name == "convergence":
            states[name] = "completed" if payload.get("completed_points") == payload.get("expected_points") else "failed"
        else:
            states[name] = "completed"
    return {
        "schema": "mimicx.evidence-campaign-status.v1",
        "settled": bool(states) and all(value not in {"pending", "invalid"} for value in states.values()),
        "terminal": bool(states) and all(value == "completed" for value in states.values()),
        "workstreams": states,
        "artifacts": {name: str(path) for name, path in paths.items()},
    }
