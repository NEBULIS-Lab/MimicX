#!/usr/bin/env python3
"""Convert a named NPZ with an explicit JSON articulation mapping."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mimicx.adapters.isaac_motion import convert_motion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--mapping", type=Path, required=True)
    args = parser.parse_args()
    mapping = json.loads(args.mapping.read_text())
    print(convert_motion(args.source, args.destination, **mapping))


if __name__ == "__main__":
    main()
