#!/usr/bin/env python3
"""Mine rollout failures from MimicX step metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--step-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--pre-window", type=int, default=40)
    parser.add_argument("--post-window", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def as_bool(value: str | None) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def body_from_metric(name: str) -> str:
    for prefix in ("body_pos_error_", "body_z_error_", "ee_z_error_"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def channel_from_metric(name: str) -> str:
    if name.startswith("ee_z_error_"):
        return "end_effector_z"
    if name.startswith("body_z_error_"):
        return "body_z"
    if name.startswith("body_pos_error_"):
        return "body_pos"
    if "anchor" in name or "root" in name:
        return "root_anchor"
    if "joint" in name:
        return "joint"
    if "lin_vel" in name or "ang_vel" in name or "vel" in name:
        return "velocity"
    return "other"


def family_from_bodies_and_channels(bodies: list[str], channels: list[str]) -> list[str]:
    families: list[str] = []
    body_text = " ".join(bodies)
    if any(word in body_text for word in ("wrist", "elbow", "shoulder")):
        families.append("arm_wrist_precision")
    if any(word in body_text for word in ("ankle", "knee", "hip")):
        families.append("lower_body_contact_z")
    if any(word in body_text for word in ("pelvis", "torso")):
        families.append("root_core_guard")
    if any(channel in channels for channel in ("root_anchor",)):
        families.append("root_core_guard")
    if any(channel in channels for channel in ("velocity",)):
        families.append("dynamics_smoothness")
    if any(channel in channels for channel in ("body_z", "end_effector_z")):
        families.append("z_clearance_contact")
    if not families:
        families.append("general_body_tracking")
    return sorted(set(families))


def main() -> int:
    args = parse_args()
    metrics = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    with args.step_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {args.step_csv}")

    done_step = None
    termination_counts: dict[str, int] = {}
    for row in rows:
        step = int(row["step"])
        if as_bool(row.get("done_any")) and done_step is None:
            done_step = step
        for key, value in row.items():
            if key.startswith("termination_") and as_bool(value):
                termination_counts[key.replace("termination_", "")] = (
                    termination_counts.get(key.replace("termination_", ""), 0) + 1
                )
                if done_step is None:
                    done_step = step

    numeric_columns = [
        key
        for key in rows[0]
        if key != "step" and any(as_float(row.get(key)) is not None for row in rows)
    ]
    error_columns = [
        key
        for key in numeric_columns
        if key.startswith(("body_pos_error_", "body_z_error_", "ee_z_error_"))
        or key
        in {
            "body_pos_error_max",
            "body_z_error_max",
            "ee_z_error_max",
            "rew_mean",
        }
        or "anchor" in key
        or "joint" in key
        or "vel" in key
    ]

    if done_step is None:
        peak_column = "body_pos_error_max" if "body_pos_error_max" in error_columns else error_columns[0]
        done_step = max(
            rows,
            key=lambda row: as_float(row.get(peak_column)) or float("-inf"),
        )["step"]
        done_step = int(done_step)

    start = max(1, done_step - args.pre_window)
    end = min(int(rows[-1]["step"]), done_step + args.post_window)
    window_rows = [row for row in rows if start <= int(row["step"]) <= end]

    ranked: list[dict] = []
    for key in error_columns:
        values = [as_float(row.get(key)) for row in window_rows]
        values = [value for value in values if value is not None]
        if not values:
            continue
        ranked.append(
            {
                "metric": key,
                "avg": mean(values),
                "max": max(values),
                "channel": channel_from_metric(key),
                "body": body_from_metric(key),
            }
        )
    ranked.sort(key=lambda item: (item["max"], item["avg"]), reverse=True)
    top_metrics = ranked[: args.top_k]
    dominant_bodies = [
        item["body"]
        for item in top_metrics
        if item["metric"].startswith(("body_pos_error_", "body_z_error_", "ee_z_error_"))
    ]
    dominant_channels = sorted({item["channel"] for item in top_metrics})

    report = {
        "source": {
            "metrics_json": str(args.metrics_json.resolve()),
            "step_csv": str(args.step_csv.resolve()),
        },
        "summary": {
            "steps": int(rows[-1]["step"]),
            "done_any_count": metrics.get("done_any_count"),
            "first_done_step": done_step,
            "termination_counts": termination_counts,
            "reward_mean_avg": metrics.get("reward_mean_avg"),
            "final_rew_mean": metrics.get("final_rew_mean"),
        },
        "failure_windows": [
            {
                "window": [start, end],
                "center_step": done_step,
                "dominant_bodies": dominant_bodies[: args.top_k],
                "dominant_channels": dominant_channels,
                "recommended_families": family_from_bodies_and_channels(
                    dominant_bodies, dominant_channels
                ),
                "top_metrics": top_metrics,
            }
        ],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        window = report["failure_windows"][0]
        lines = [
            "# AutoRefine Failure Report",
            "",
            f"- Metrics: `{args.metrics_json}`",
            f"- Step CSV: `{args.step_csv}`",
            f"- Done count: `{report['summary']['done_any_count']}`",
            f"- First/peak failure step: `{done_step}`",
            f"- Window: `{window['window'][0]}-{window['window'][1]}`",
            f"- Recommended families: `{', '.join(window['recommended_families'])}`",
            "",
            "## Top Metrics",
            "",
            "| metric | channel | body | avg | max |",
            "|---|---|---|---:|---:|",
        ]
        for item in top_metrics:
            lines.append(
                f"| `{item['metric']}` | `{item['channel']}` | `{item['body']}` | "
                f"{item['avg']:.4f} | {item['max']:.4f} |"
            )
        args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"failure_report={args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
