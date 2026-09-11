#!/usr/bin/env python3
"""Create a four-method controlled comparison from an existing loop manifest."""

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mimicx.refinement.manifest import load_loop_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--failure-window", type=int, nargs=2, required=True)
    parser.add_argument("--priority-bodies", nargs="+", required=True)
    parser.add_argument("--refined-motion", type=Path)
    parser.add_argument("--observation-dim", type=int, default=160)
    args = parser.parse_args()
    manifest = load_loop_manifest(args.loop_manifest)
    start, end = args.failure_window
    if not 1 <= start < end <= manifest.task.horizon:
        parser.error("failure window must lie inside [1, horizon]")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    runtime = asdict(manifest.runtime)
    runtime = {key: str(value) if isinstance(value, Path) else value for key, value in runtime.items()}
    runtime.update(static_task=manifest.task.strict_task, strict_task=manifest.task.strict_task)
    task = {"id": manifest.task.id, "motion_file": str(manifest.task.motion_file),
            "base_checkpoint": str(manifest.task.base_checkpoint),
            "load_run": manifest.task.load_run, "horizon": manifest.task.horizon,
            "policy_observation_dim": args.observation_dim,
            "failure_window": args.failure_window, "priority_bodies": args.priority_bodies,
            "learning_rate": manifest.search.learning_rate}
    if args.refined_motion:
        if not args.refined_motion.is_file():
            raise FileNotFoundError(args.refined_motion)
        task["refined_motion_file"] = str(args.refined_motion.resolve())
    methods = [
        {"id": "m0_open_loop", "mode": "static", "patch_profile": "none"},
        {"id": "m1_policy_window", "mode": "static", "patch_profile": "window"},
        {"id": "m2_task_hierarchy", "mode": "static", "patch_profile": "hierarchical"},
        {"id": "m3_full_mimicx", "mode": "closed_loop", "candidate_budget": manifest.search.candidate_budget,
         "max_iterations": manifest.search.max_iterations, "use_refined_reference": bool(args.refined_motion)},
    ]
    files = {"task.yaml": task}
    files.update({method["id"] + ".yaml": method for method in methods})
    files["matrix.yaml"] = {
        "schema": "mimicx.paper-matrix.v1", "runtime": runtime,
        "tasks": ["task.yaml"], "methods": [method["id"] + ".yaml" for method in methods],
        "seeds": [101, 202, 303],
        "training": {"iterations": manifest.search.train_iterations, "num_envs": manifest.search.num_envs},
        "verification": {"repeats": manifest.verification.repeats, "seeds": list(manifest.verification.seeds)},
    }
    for name, value in files.items():
        (args.output_dir / name).write_text(yaml.safe_dump(value, sort_keys=False))
    print(args.output_dir / "matrix.yaml")


if __name__ == "__main__":
    main()
