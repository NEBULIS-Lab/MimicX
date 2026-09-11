#!/usr/bin/env python3
"""Create a path-independent deterministic selection from native failure reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_selection(candidates: list[tuple[str, Path]]) -> dict[str, object]:
    rows = []
    for candidate_id, path in candidates:
        report = json.loads(path.read_text())
        summary = report["summary"]
        steps = int(summary["steps"])
        frontier = int(summary["first_done_step"])
        rows.append({
            "candidate_id": candidate_id,
            "steps": steps,
            "first_done_step": frontier,
            "accepted": frontier >= steps,
        })
    rows.sort(key=lambda row: row["candidate_id"])
    selected = sorted(
        (row for row in rows if row["accepted"]),
        key=lambda row: row["candidate_id"],
    )
    return {
        "schema": "mimicx.native-hloop-selection.v1",
        "candidates": rows,
        "selected_candidate_ids": [row["candidate_id"] for row in selected],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", nargs=2, action="append", metavar=("ID", "REPORT"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_selection([(item[0], Path(item[1])) for item in args.candidate])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
