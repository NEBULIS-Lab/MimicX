"""Prepare source-clock poses, stabilize supplied cameras, or explicitly render replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mimicx.visualization.source_camera import export_source_video, prepare_source_replay, stabilize


def load_replay(folder):
    keys = ("timestamps", "robot_root_pose", "robot_joint_positions", "reference_root_pose", "reference_joint_positions")
    arrays = {key: np.load(folder / f"{key}.npy", allow_pickle=False) for key in keys}
    if (folder / "reference_timestamps.npy").is_file():
        arrays["reference_timestamps"] = np.load(folder / "reference_timestamps.npy", allow_pickle=False)
    names = json.loads((folder / "joint_names.json").read_text())
    return arrays, names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("stabilize")
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--background", type=Path, required=True, help="Source-background motion audit JSON")
    p.add_argument("--sigma-frames", type=float, default=10.)
    p.add_argument("--output", type=Path, required=True)
    for command in ("prepare", "render"):
        p = sub.add_parser(command)
        p.add_argument("--replay", type=Path, required=True)
        p.add_argument("--frames", type=int, required=True)
        p.add_argument("--fps", type=float, required=True)
        p.add_argument("--max-endpoint-hold", type=float, default=.04)
        p.add_argument("--output", type=Path, required=True)
        if command == "render":
            p.add_argument("--calibration", type=Path, required=True)
            p.add_argument("--model", type=Path, required=True)
            p.add_argument("--source-size", type=int, nargs=2, required=True, metavar=("WIDTH", "HEIGHT"))
            p.add_argument("--output-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"))
            p.add_argument("--backend", choices=("egl", "osmesa"), required=True)
            p.add_argument("--ghost-rgba", type=float, nargs=4, default=(.23, .47, .66, .23))
    args = parser.parse_args()
    if args.command == "prepare" and args.output.suffix != ".npz":
        args.output = Path(str(args.output) + ".npz")
    destinations = [args.output]
    if args.command == "render":
        destinations.append(args.output.with_suffix(".json"))
    for path in destinations:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite output: {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "stabilize":
        result = stabilize(json.loads(args.calibration.read_text()), json.loads(args.background.read_text()),
                           sigma_frames=args.sigma_frames)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    else:
        arrays, names = load_replay(args.replay)
        prepared = prepare_source_replay(arrays, frames=args.frames, fps=args.fps, max_endpoint_hold=args.max_endpoint_hold)
        if args.command == "prepare":
            with args.output.open("xb") as stream:
                np.savez_compressed(stream, time=prepared["time"], actual=prepared["actual"],
                                    reference=prepared["reference"], joint_names=np.asarray(names, dtype=str))
            result = {key: prepared[key] for key in ("coordinate_transform", "endpoint_hold_seconds")}
            result["frames"] = args.frames
        else:
            os.environ["MUJOCO_GL"] = args.backend
            if args.backend == "osmesa":
                os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
            result = export_source_video(prepared, names, json.loads(args.calibration.read_text()),
                                         model_path=args.model, output=args.output, fps=args.fps,
                                         source_size=args.source_size, output_size=args.output_size,
                                         ghost_rgba=args.ghost_rgba)
            result["endpoint_hold_seconds"] = prepared["endpoint_hold_seconds"]
            result["input_sha256"] = {key: hashlib.sha256((args.replay / f"{key}.npy").read_bytes()).hexdigest()
                                      for key in arrays}
            result["calibration_sha256"] = hashlib.sha256(args.calibration.read_bytes()).hexdigest()
            result["model_sha256"] = hashlib.sha256(args.model.read_bytes()).hexdigest()
            with args.output.with_suffix(".json").open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
