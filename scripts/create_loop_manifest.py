#!/usr/bin/env python3
"""Create and validate a task's portable AutoRefine manifest."""

import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mimicx.refinement.manifest import load_loop_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--horizon", type=int, required=True, help="Full evaluation length at 50 Hz, including any intended padding")
    parser.add_argument("--backend", type=Path, default=ROOT / "third_party/unitree_rl_mjlab")
    parser.add_argument("--runtime-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--train-iterations", type=int, default=250)
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--devices", nargs="+", type=int, default=[0], help="Indices inside the visible GPU set")
    args = parser.parse_args()
    backend, checkpoint = args.backend.resolve(), args.checkpoint.resolve()
    checkpoint_root = (args.checkpoint_root or backend / "logs/rsl_rl/g1_tracking").resolve()
    checkpoint.relative_to(checkpoint_root)
    payload = {
        "schema": "mimicx.autorefine-loop.v1",
        "task": {"id": args.task_id, "motion_file": str(args.motion_file.resolve()),
                 "base_checkpoint": str(checkpoint), "load_run": checkpoint.parent.name,
                 "strict_task": "Unitree-G1-Tracking-MimicX-Curriculum", "horizon": args.horizon},
        "runtime": {"workdir": str(backend), "python": str(args.runtime_python.absolute()),
                    "train_script": str(backend / "scripts/train.py"),
                    "rollout_script": str(ROOT / "scripts/rollout_mjlab_motion_smoke.py"),
                    "checkpoint_root": str(checkpoint_root),
                    "dynamic_task": "Unitree-G1-Tracking-MimicX-AutoRefine-Dynamic-Curriculum",
                    "env": {"WANDB_MODE": "offline", "MIMICX_ROOT": str(ROOT)}},
        "search": {"max_iterations": args.max_iterations, "candidate_budget": 3,
                   "train_iterations": args.train_iterations, "num_envs": args.num_envs,
                   "learning_rate": args.learning_rate, "train_seed": args.seed,
                   "candidate_gpus": args.devices, "eval_gpus": args.devices},
        "verification": {"repeats": 3, "seeds": [1001, 2002, 3003],
                         "metric_keys": ["body_pos_error_max", "ee_z_error_max"],
                         "guards": {"body_pos_error_max": {"direction": "lower", "relative_tolerance": 0.2}}},
        "execution": {"max_parallel": len(args.devices)},
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False))
    load_loop_manifest(output)
    print(output)


if __name__ == "__main__":
    main()
