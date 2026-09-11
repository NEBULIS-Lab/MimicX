#!/usr/bin/env python3
"""Generate AutoRefine candidate patches from a failure report."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mimicx.refinement.proposals import ProposalContext, generate_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--load-run", required=True)
    parser.add_argument("--train-iters", type=int, default=200)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--run-prefix", default="mimicx_autorefine_auto")
    parser.add_argument("--task-id", default="unknown")
    parser.add_argument("--candidate-budget", type=int, default=3)
    parser.add_argument("--runtime-workdir", type=Path, default=PROJECT_ROOT / "third_party/unitree_rl_mjlab")
    parser.add_argument("--runtime-python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--backend",
        choices=("heuristic", "llm", "hybrid"),
        default="heuristic",
    )
    parser.add_argument(
        "--llm-response-file",
        type=Path,
        help="Pre-generated JSON response for a constrained llm/hybrid proposal.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = json.loads(args.failure_report.read_text(encoding="utf-8"))
    context = ProposalContext.from_failure_report(report, task_id=args.task_id)
    llm_generate = None
    if args.backend != "heuristic":
        if args.llm_response_file is None:
            raise SystemExit("--llm-response-file is required for llm/hybrid backend")
        response = args.llm_response_file.read_text(encoding="utf-8")
        llm_generate = lambda _prompt: response
    patches = generate_candidates(
        context,
        backend=args.backend,
        llm_generate=llm_generate,
        candidate_budget=args.candidate_budget,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    patch_dir = args.output_dir / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for index, patch in enumerate(patches, start=1):
        candidate_id = f"auto_{index:02d}_{patch['id']}"
        patch["id"] = candidate_id
        patch_path = patch_dir / f"{candidate_id}.json"
        patch_path.write_text(json.dumps(patch, indent=2) + "\n", encoding="utf-8")
        run_name = f"{args.run_prefix}_{candidate_id}"
        command = (
            f"cd {shlex.quote(str(args.runtime_workdir.resolve()))} && "
            "export PYTHONPATH=$PWD:$PWD/src && "
            "export WANDB_MODE=offline && "
            f"export MIMICX_AUTOREFINE_PATCH_FILE={shlex.quote(str(patch_path.resolve()))} && "
            f"{shlex.quote(str(args.runtime_python.absolute()))} "
            "scripts/train.py Unitree-G1-Tracking-MimicX-AutoRefine-Dynamic-Curriculum "
            f"--motion-file {shlex.quote(str(args.motion_file.resolve()))} "
            f"--env.scene.num-envs {args.num_envs} "
            f"--agent.max-iterations {args.train_iters} "
            "--agent.save-interval 25 "
            f"--agent.run-name {run_name} "
            "--agent.resume True "
            f"--agent.load-run {shlex.quote(args.load_run)} "
            f"--agent.load-checkpoint {shlex.quote(args.base_checkpoint.name)} "
            f"--agent.algorithm.learning-rate {args.learning_rate}"
        )
        candidates.append(
            {
                "id": candidate_id,
                "patch_file": str(patch_path),
                "task": "Unitree-G1-Tracking-MimicX-AutoRefine-Dynamic-Curriculum",
                "run_name": run_name,
                "hypothesis": patch["id"].replace("_", " "),
                "train_command": command,
            }
        )

    matrix = {
        "source_failure_report": str(args.failure_report.resolve()),
        "motion_file": str(args.motion_file.resolve()),
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "load_run": args.load_run,
        "proposal_backend": args.backend,
        "candidates": candidates,
    }
    matrix_json = args.output_dir / "candidate_matrix.json"
    matrix_json.write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# AutoRefine Candidate Matrix",
        "",
        f"- Failure report: `{args.failure_report}`",
        f"- Motion file: `{args.motion_file}`",
        f"- Base checkpoint: `{args.base_checkpoint}`",
        "",
        "| id | patch | task |",
        "|---|---|---|",
    ]
    for candidate in candidates:
        lines.append(
            f"| `{candidate['id']}` | `{candidate['patch_file']}` | `{candidate['task']}` |"
        )
    (args.output_dir / "candidate_matrix.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"candidate_matrix={matrix_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
