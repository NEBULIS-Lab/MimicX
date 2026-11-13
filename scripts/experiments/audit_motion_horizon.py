"""Portable CPU-only horizon audit; paths are caller-supplied, never embedded.

Examples (from the repository root)::

    python3 scripts/experiments/audit_motion_horizon.py artifacts/reference.npz
    python3 scripts/experiments/audit_motion_horizon.py artifacts/reference.npz \
        --rollout artifacts/rollout.npz --timestamp-kind simulation \
        --minimum-duration-s 60 --stages-json artifacts/stages.json

stages.json is a list of {"name": "approach", "start_frame": 0,
"end_frame_inclusive": 500} objects partitioning the scoring window with shared
endpoints. No pose-based phase inference. Output is JSON to stdout by default,
contains hashes but no input paths, and never equates horizon with success.
Horizon verification accepts either declared actual simulation or wall seconds;
simulation does not need to run at wall speed. real_time_playback_verified is
unknown unless the selected timestamps are measured wall seconds.
Exit status: 0 = report produced (including unknown/failed audits), 2 = invalid
input; --require-verified additionally exits 1 unless the horizon is verified.
"""

from pathlib import Path
import argparse
import json
import sys
from zipfile import BadZipFile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mimicx.evaluation.horizon_audit import Stage, audit_motion_horizon


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", type=Path, help="Reference NPZ with explicit fps/frame_rate")
    parser.add_argument("--rollout", type=Path, help="Actual row-synchronized rollout metadata NPZ")
    parser.add_argument("--scoring-window", nargs=2, type=int, metavar=("START_FRAME", "END_FRAME"),
                        help="Inclusive source-frame endpoints (default: entire source)")
    parser.add_argument("--stages-json", type=Path, help="Ordered stage list with shared inclusive boundaries")
    parser.add_argument("--minimum-duration-s", type=float, help="Explicit long-horizon requirement; no inferred threshold")
    parser.add_argument("--timestamp-kind", choices=("wall", "simulation"),
                        help="Actual simulation or wall clock; required unless declared in rollout NPZ")
    parser.add_argument("--timestamp-key", default="timestamps", help="Actual simulation/wall seconds vector; never synthesized")
    parser.add_argument("--reference-frame-key", default="reference_frame")
    parser.add_argument("--reset-key", default="reset", help="Incoming sample is reset/invalid, not initial setup")
    parser.add_argument("--failure-key", default="failure", help="Incoming sample fails; tail is censored")
    parser.add_argument("--clock-tolerance-s", type=float, help="Override common-protocol absolute clock tolerance")
    parser.add_argument("--state-gap-s", type=float, help="Override common-protocol maximum sample gap")
    parser.add_argument("--duration-ratio-bounds", nargs=2, type=float, metavar=("LOW", "HIGH"), default=(.97, 1.03))
    parser.add_argument("--output", type=Path, help="New JSON file; default stdout; refuses to overwrite")
    parser.add_argument("--require-verified", action="store_true",
                        help="Require continuous horizon on the declared clock, not necessarily wall-speed playback")
    args = parser.parse_args(argv)
    try:
        stages = None
        if args.stages_json is not None:
            items = json.loads(args.stages_json.read_text(encoding="utf-8"))
            fields = {"name", "start_frame", "end_frame_inclusive"}
            if not isinstance(items, list) or any(not isinstance(item, dict) or set(item) != fields for item in items):
                raise ValueError("stages JSON must be a list of name/start_frame/end_frame_inclusive objects")
            stages = [Stage(**item) for item in items]
        options = dict(scoring_window=args.scoring_window, stages=stages,
                       minimum_duration_s=args.minimum_duration_s, timestamp_kind=args.timestamp_kind,
                       timestamp_key=args.timestamp_key, reference_frame_key=args.reference_frame_key,
                       reset_key=args.reset_key, failure_key=args.failure_key,
                       duration_ratio_bounds=args.duration_ratio_bounds)
        for key in ("clock_tolerance_s", "state_gap_s"):
            if getattr(args, key) is not None:
                options[key] = getattr(args, key)
        report = audit_motion_horizon(args.reference, args.rollout, **options)
        payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is None:
            print(payload, end="")
        else:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(payload)
    except (ValueError, OSError, EOFError, BadZipFile) as exc:
        parser.exit(2, f"audit input error: {exc}\n")
    return int(args.require_verified and report["horizon_verified"] is not True)


if __name__ == "__main__":
    raise SystemExit(main())
