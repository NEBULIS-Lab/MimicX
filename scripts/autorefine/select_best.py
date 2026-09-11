#!/usr/bin/env python3
"""Rank AutoRefine checkpoints by strict rollout metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def as_bool(value: str | None) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def first_done_from_csv(metrics_path: Path) -> int | None:
    candidates = [
        metrics_path.with_name(metrics_path.stem + "_steps.csv"),
        metrics_path.with_name(metrics_path.name.replace(".json", "_steps.csv")),
    ]
    for path in candidates:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if as_bool(row.get("done_any")):
                    return int(row["step"])
                for key, value in row.items():
                    if key.startswith("termination_") and as_bool(value):
                        return int(row["step"])
    return None


def final_step_errors(metrics_path: Path) -> dict[str, float | None]:
    path = metrics_path.with_name(metrics_path.stem + "_steps.csv")
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    last = rows[-1]
    return {
        "final_body_pos": as_float(last.get("body_pos_error_max")),
        "final_ee_z": as_float(last.get("ee_z_error_max")),
        "final_anchor_pos": as_float(last.get("anchor_pos"))
        or as_float(last.get("motion_anchor_pos")),
    }


def row_for_metrics(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    first_done = first_done_from_csv(path)
    done_count = int(data.get("done_any_count", 0))
    if first_done is None:
        first_done = int(data.get("steps", 0)) + 1
    errors = final_step_errors(path)
    row = {
        "metrics": str(path.resolve()),
        "checkpoint_file": data.get("checkpoint_file"),
        "done_any_count": done_count,
        "first_done_step": first_done,
        "reward_mean_avg": data.get("reward_mean_avg"),
        "final_rew_mean": data.get("final_rew_mean"),
        "final_body_pos": errors.get("final_body_pos"),
        "final_ee_z": errors.get("final_ee_z"),
        "final_anchor_pos": errors.get("final_anchor_pos"),
    }
    body = row["final_body_pos"] if row["final_body_pos"] is not None else 999.0
    ee_z = row["final_ee_z"] if row["final_ee_z"] is not None else 999.0
    anchor = row["final_anchor_pos"] if row["final_anchor_pos"] is not None else 999.0
    row["selection_score"] = (
        1000.0 * done_count - float(first_done) + 5.0 * body + 8.0 * ee_z + 10.0 * anchor
    )
    return row


def main() -> int:
    args = parse_args()
    rows = [row_for_metrics(path) for path in args.metrics]
    rows.sort(key=lambda row: row["selection_score"])
    result = {"best": rows[0] if rows else None, "ranked": rows}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# AutoRefine Strict Selection",
            "",
            "| rank | metrics | done | first done | avg reward | final reward | score |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
        for index, row in enumerate(rows, start=1):
            lines.append(
                f"| {index} | `{Path(row['metrics']).name}` | {row['done_any_count']} | "
                f"{row['first_done_step']} | {row['reward_mean_avg']} | "
                f"{row['final_rew_mean']} | {row['selection_score']:.4f} |"
            )
        args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"selection={args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
