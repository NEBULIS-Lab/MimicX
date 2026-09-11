"""Deterministic aggregation for the frozen MimicX paper matrix."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


HIGHER_IS_BETTER = ("reward_mean", "worst_first_failure_step")
LOWER_IS_BETTER = (
    "body_pos_error_mean", "body_pos_error_p95", "ee_z_error_mean",
    "anchor_pos_error_mean", "body_lin_vel_error_mean", "body_ang_vel_error_mean",
)


def _pct_change(before: float, after: float, higher: bool) -> float:
    if before == 0:
        raise ValueError("cannot compute relative change from zero")
    return 100.0 * ((after - before) / before if higher else (before - after) / before)


def build_final_evidence(matrix_root: Path, runs_root: Path) -> dict[str, object]:
    rows = list(csv.DictReader((matrix_root / "job_results.csv").open(encoding="utf-8")))
    if len(rows) != 48 or any(row["status"] != "completed" for row in rows):
        raise ValueError(f"expected 48 completed rows, found {len(rows)}")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["task_id"], row["method_id"])].append(row)
    tasks = sorted({row["task_id"] for row in rows})
    methods = ["m0_open_loop", "m1_policy_window", "m2_task_hierarchy", "m3_full_mimicx"]
    if set(grouped) != {(task, method) for task in tasks for method in methods} or any(len(values) != 3 for values in grouped.values()):
        raise ValueError("matrix is not a complete 4-task x 4-method x 3-seed design")
    directions = {metric: True for metric in HIGHER_IS_BETTER}
    directions.update({metric: False for metric in LOWER_IS_BETTER})
    task_effects: dict[str, dict[str, float]] = {}
    paired_wins = {metric: 0 for metric in directions}
    for task in tasks:
        m0 = {int(row["train_seed"]): row for row in grouped[(task, "m0_open_loop")]}
        m3 = {int(row["train_seed"]): row for row in grouped[(task, "m3_full_mimicx")]}
        effects: dict[str, float] = {}
        for metric, higher in directions.items():
            before = mean(float(row[metric]) for row in m0.values())
            after = mean(float(row[metric]) for row in m3.values())
            effects[metric] = _pct_change(before, after, higher)
            for seed in sorted(m0):
                left, right = float(m0[seed][metric]), float(m3[seed][metric])
                paired_wins[metric] += int(right > left if higher else right < left)
        task_effects[task] = effects
    selections = sorted(runs_root.glob("jobs/*__m3_full_mimicx__seed*/closed_loop/iterations/iter_000/selection.json"))
    if len(selections) != 12:
        raise ValueError(f"expected 12 M3 selections, found {len(selections)}")
    decisions = []
    for path in selections:
        payload = json.loads(path.read_text(encoding="utf-8"))
        job_id = path.parents[3].name
        task, _, seed_text = job_id.partition("__m3_full_mimicx__seed")
        decision = payload["decision"]
        decisions.append({"job_id": job_id, "task_id": task, "train_seed": int(seed_text), "accepted": bool(decision["accepted"]), "selected_id": decision["selected_id"], "reasons": decision.get("reasons", [])})
    accepted = sum(item["accepted"] for item in decisions)
    return {
        "schema": "mimicx.final-evidence.v1", "completed_jobs": len(rows),
        "tasks": tasks, "methods": methods,
        "task_effects_m3_vs_m0_percent": task_effects,
        "macro_effects_m3_vs_m0_percent": {metric: mean(task_effects[task][metric] for task in tasks) for metric in directions},
        "paired_seed_wins_m3_vs_m0": {metric: {"wins": wins, "total": 12} for metric, wins in paired_wins.items()},
        "m3_gate": {"accepted": accepted, "rollback": 12 - accepted, "total": 12, "decisions": decisions},
    }


def write_final_evidence(payload: dict[str, object], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    effects = payload["task_effects_m3_vs_m0_percent"]
    metrics = list(payload["macro_effects_m3_vs_m0_percent"])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task_id", *metrics])
        for task, values in effects.items():
            writer.writerow([task, *(values[metric] for metric in metrics)])
        writer.writerow(["macro", *(payload["macro_effects_m3_vs_m0_percent"][metric] for metric in metrics)])
