"""Audit baseline state/support telemetry against measured playback timestamps."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mimicx.visualization.baseline_recording import validate_playback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--playback-wall", type=Path, required=True, help="NPY monotonic timestamps in seconds")
    parser.add_argument("--clock-audit", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    clock = json.loads(args.clock_audit.read_text()) if args.clock_audit else None
    with np.load(args.states, allow_pickle=False) as states:
        result = validate_playback(states, np.load(args.playback_wall, allow_pickle=False), clock_audit=clock)
    result = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()}
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
