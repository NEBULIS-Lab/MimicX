#!/usr/bin/env python3
"""Validate paper inputs and generate a relocatable four-task matrix."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.autorefine.launch_paper_matrix import load_matrix, expand_jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--backend", type=Path, default=ROOT / "third_party/unitree_rl_mjlab")
    parser.add_argument("--runtime-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=ROOT / "mimicx/configs/paper/core_inputs.json")
    args = parser.parse_args()
    bundle, backend = args.assets.resolve(), args.backend.resolve()
    spec = json.loads(args.spec.read_text())
    for asset in spec["assets"]:
        path = bundle / asset["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing paper input: {asset['path']}")
        with path.open("rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        if digest != asset["sha256"]:
            raise ValueError(f"Hash mismatch: {asset['path']}")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    task_paths = []
    for task in spec["tasks"]:
        task = dict(task)
        checkpoint = bundle / task["base_checkpoint"]
        target = backend / "logs/rsl_rl/g1_tracking" / task["load_run"] / checkpoint.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != checkpoint.read_bytes():
            raise FileExistsError(f"Different checkpoint already exists: {target}")
        if not target.exists():
            shutil.copy2(checkpoint, target)
        task["base_checkpoint"] = str(target)
        for key in ("motion_file", "refined_motion_file", "hloop_checkpoint"):
            if key in task:
                task[key] = str(bundle / task[key])
        name = f"{task['id']}.yaml"
        (args.output / name).write_text(yaml.safe_dump(task, sort_keys=False))
        task_paths.append(name)
    methods = []
    for source in sorted((ROOT / "mimicx/configs/paper/methods").glob("*.yaml")):
        shutil.copy2(source, args.output / source.name)
        methods.append(source.name)
    runtime = {"workdir": str(backend), "python": str(args.runtime_python.absolute()),
               "train_script": str(backend / "scripts/train.py"),
               "rollout_script": str(ROOT / "scripts/rollout_mjlab_motion_smoke.py"),
               "checkpoint_root": str(backend / "logs/rsl_rl/g1_tracking"),
               "static_task": "Unitree-G1-Tracking-MimicX-Curriculum",
               "strict_task": "Unitree-G1-Tracking-MimicX-Curriculum",
               "dynamic_task": "Unitree-G1-Tracking-MimicX-AutoRefine-Dynamic-Curriculum",
               "env": {"WANDB_MODE": "offline", "MIMICX_ROOT": str(ROOT)}}
    matrix = {"schema": "mimicx.paper-matrix.v1", "runtime": runtime, "tasks": task_paths,
              "methods": methods, "seeds": spec["seeds"],
              "training": {"iterations": spec["train_iterations"], "num_envs": spec["num_envs"]},
              "verification": {"repeats": 3, "seeds": spec["verification_seeds"]}}
    output = args.output / "matrix.yaml"
    output.write_text(yaml.safe_dump(matrix, sort_keys=False))
    hloop = {"schema": "mimicx.native-hloop-benchmark.v1", "task_dir": str(args.output.resolve()),
             "runtime_python": str(args.runtime_python.absolute()), "backend": str(backend),
             "tasks": [task["id"] for task in spec["tasks"]], "gpus": [0, 1],
             "evaluation_seeds": [1001, 2002], "repetitions": 5, "warmup": True}
    (args.output / "hloop.yaml").write_text(yaml.safe_dump(hloop, sort_keys=False))
    jobs = expand_jobs(load_matrix(output), gpu_ids=(0,))
    print(json.dumps({"validated_assets": len(spec["assets"]), "prepared_jobs": len(jobs),
                      "matrix": str(output)}, indent=2))


if __name__ == "__main__":
    main()
