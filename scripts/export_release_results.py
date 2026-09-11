#!/usr/bin/env python3
"""Publish numerical evidence, omitting machine placement and private artifact paths."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

FILES = (
    "core_method_task_summary.csv", "core_trial_results.csv", "accept_protect_decisions.csv",
    "hloop_timings.csv", "sonic_comparison.csv", "additional_video_tasks.csv",
    "motion_breadth.csv", "reward_higher_regression_rejection.csv",
    "tennis_training_dynamics_3000_points.csv",
)
OMIT = {"gpu_id", "checkpoint", "source", "source_path", "source_log", "run_dir", "path"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name in FILES:
        source = args.source_dir / name
        with source.open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fields = [key for key in reader.fieldnames if key not in OMIT]
        # Preserve numeric strings exactly; path-bearing metadata is not public data.
        fields = [key for key in fields if not any(
            str(row.get(key, "")).startswith(("/", "file://")) for row in rows)]
        target = args.output_dir / name
        with target.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        records.append({"file": name, "rows": len(rows),
                        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "published_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                        "omitted_columns": [key for key in reader.fieldnames if key not in fields]})
    (args.output_dir / "provenance.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
