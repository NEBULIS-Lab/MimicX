#!/usr/bin/env python
"""Convert GMR robot motion pickle files to Unitree RL Mjlab CSV files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mimicx.adapters.gmr_mjlab import convert_gmr_pkl_to_mjlab_csv, load_gmr_motion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="Input GMR .pkl file.")
    parser.add_argument("--output", required=True, type=Path, help="Output .csv file.")
    parser.add_argument("--print-summary", action="store_true", help="Print motion shape and fps before converting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.print_summary:
        motion = load_gmr_motion(args.input)
        print(f"input={args.input}")
        print(f"fps={motion.fps}")
        print(f"frames={motion.num_frames}")
        print(f"dofs={motion.num_dofs}")
    out = convert_gmr_pkl_to_mjlab_csv(args.input, args.output)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
