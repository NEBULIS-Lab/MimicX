#!/usr/bin/env python3
"""Execute a fixed HLoop plan or compare two completed scheduler reports."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mimicx.runtime.hloop import load_hloop_plan, run_hloop_plan, compare_hloop_runs
from mimicx.runtime.hloop_baselines import run_hloop_plan_bulk_sync


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--mode", choices=("sequential", "hloop", "bulk-sync"), default="hloop")
    compare = sub.add_parser("compare")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "compare":
        report = compare_hloop_runs(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()))
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1
    plan = load_hloop_plan(args.plan)
    if args.mode == "bulk-sync":
        result = run_hloop_plan_bulk_sync(plan, run_dir=args.run_dir, repo_root=ROOT)
    else:
        result = run_hloop_plan(plan, run_dir=args.run_dir, repo_root=ROOT,
                                mode="e0" if args.mode == "hloop" else args.mode)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
