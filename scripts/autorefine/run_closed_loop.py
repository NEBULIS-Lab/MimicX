#!/usr/bin/env python3
"""Run or replay an unattended MimicX AutoRefine loop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mimicx.refinement.closed_loop import ClosedLoopCoordinator
from mimicx.refinement.manifest import load_loop_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("prepare", "replay", "execute"), required=True)
    parser.add_argument("--runtime-gpu", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime_gpu_ids = (args.runtime_gpu,) if args.runtime_gpu is not None else None
    coordinator = ClosedLoopCoordinator(
        load_loop_manifest(args.manifest),
        args.run_dir,
        runtime_gpu_ids=runtime_gpu_ids,
    )
    result = getattr(coordinator, args.mode)()
    print(json.dumps(result.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
