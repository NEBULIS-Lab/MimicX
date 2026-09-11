#!/usr/bin/env python3
"""Build live or frozen paper tables from a native MimicX matrix run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mimicx.evaluation.paper_matrix import write_matrix_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int)
    args = parser.parse_args()
    status = write_matrix_report(
        args.run_dir,
        args.output_dir,
        expected_seeds=args.expected_seeds,
    )
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
