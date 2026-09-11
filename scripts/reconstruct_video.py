#!/usr/bin/env python3
"""Run the pinned GVHMR demo without changing inherited device visibility."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gvhmr", type=Path, default=ROOT / "third_party/GVHMR")
    parser.add_argument("--python", type=Path, required=True, help="Python in the separate GVHMR environment")
    parser.add_argument("--static-camera", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Export GVHMR intermediate visualizations")
    args = parser.parse_args()
    video = args.video.resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    argv = [str(args.python.absolute()), "tools/demo/demo.py", "--video", str(video),
            "--output_root", str(args.output.resolve())]
    if args.static_camera:
        argv.append("--static_cam")
    if args.verbose:
        argv.append("--verbose")
    return subprocess.run(argv, cwd=args.gvhmr.resolve(), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
