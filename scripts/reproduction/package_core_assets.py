#!/usr/bin/env python3
"""Prepare exact core-paper inputs and path-free task specifications locally."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import yaml


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--trial-results", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    if args.bundle.exists() or args.spec.exists():
        raise FileExistsError("Choose new bundle/spec paths; existing artifacts are never overwritten")
    args.bundle.mkdir(parents=True)
    payload = {"schema": "mimicx.paper-inputs.v1", "tasks": [], "assets": [],
               "seeds": [101, 202, 303], "verification_seeds": [1001, 2002, 3003],
               "train_iterations": 250, "num_envs": 1024}

    def copy(source, relative, role):
        source, relative = Path(source), Path(relative)
        if not source.is_file():
            raise FileNotFoundError(source)
        target = args.bundle / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        item = {"path": str(relative), "role": role, "bytes": target.stat().st_size,
                "sha256": digest(target)}
        payload["assets"].append(item)
        return str(relative)

    for task_id in ("tennis", "football1", "dance2", "kongfu1"):
        task = yaml.safe_load((args.task_dir / f"{task_id}.yaml").read_text())
        checkpoint = Path(task["base_checkpoint"])
        task["base_checkpoint"] = copy(checkpoint, f"warmstarts/{task_id}/{checkpoint.name}", "warmstart")
        task["motion_file"] = copy(task["motion_file"], f"motions/{task_id}/original.npz", "original reference")
        if task.get("refined_motion_file"):
            task["refined_motion_file"] = copy(task["refined_motion_file"], f"motions/{task_id}/registered.npz", "registered repaired reference")
        for source in sorted((checkpoint.parent / "params").glob("*.yaml")):
            # Parameter dumps can contain private paths; retain them in the local bundle only.
            copy(source, f"warmstarts/{task_id}/params/{source.name}", "historical parameter dump (local)")
        payload["tasks"].append(task)
    with args.trial_results.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["method_id"] == "m0_open_loop" and int(row["train_seed"]) == 101:
            source = Path(row["checkpoint"])
            relative = copy(source, f"hloop/{row['task_id']}/{source.name}", "fixed HLoop baseline policy")
            next(task for task in payload["tasks"] if task["id"] == row["task_id"])["hloop_checkpoint"] = relative
    args.spec.parent.mkdir(parents=True, exist_ok=True)
    args.spec.write_text(json.dumps(payload, indent=2) + "\n")
    (args.bundle / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"assets": len(payload["assets"]), "bytes": sum(a["bytes"] for a in payload["assets"]),
                      "spec": str(args.spec), "bundle": str(args.bundle)}, indent=2))


if __name__ == "__main__":
    main()
