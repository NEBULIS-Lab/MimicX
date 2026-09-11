#!/usr/bin/env python3
"""Normalize native AutoRefine trials into an auditable history dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mimicx.refinement.history import write_history_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_history_dataset(args.config, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
