#!/usr/bin/env python3
"""Inventory/audit local external-source inputs without networks, policy results, or GPU."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mimicx.datasets.motion_scaleup import inventory, load_roots, prepare, prepare_coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inventory", "prepare", "coverage"])
    parser.add_argument("--roots", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--declared-at", help="Prospective UTC ISO timestamp, explicit for reproducibility")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("Set CUDA_VISIBLE_DEVICES='' for this CPU-only tool")
    roots = load_roots(args.roots)
    if args.command == "inventory":
        result = inventory(roots, args.output)
        print(json.dumps({k: v for k, v in result.items() if k not in ("raw_archive", "external_inputs")}, indent=2))
        print("archive_csv_count", result["raw_archive"]["csv_count"])
    elif args.command == "coverage":
        if args.inventory is None or args.plan is None:
            parser.error("coverage requires --inventory and --plan")
        result = prepare_coverage(roots, args.plan, args.inventory, args.output)
        print(json.dumps(result["counts"]))
    else:
        if args.inventory is None or args.declared_at is None:
            parser.error("prepare requires --inventory and --declared-at")
        result = prepare(roots, args.inventory, args.output, args.declared_at)
        print(json.dumps(dict(ready=result["readiness"], counts=result["counts"], qa=result["input_audit_counts"])))
        return 0 if result["readiness"] else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
