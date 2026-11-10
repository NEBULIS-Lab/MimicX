"""Build one CPU-only, full-clock 2x2 comparison from explicit local inputs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mimicx.visualization.comparison import export_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New MP4; sibling JSON receipt must also be new")
    parser.add_argument("--font", type=Path, help="TrueType font (default: DejaVuSans.ttf)")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".mp4":
        raise ValueError("Comparison output must be an MP4")
    receipt = args.output.with_suffix(".json")
    for path in (args.output, receipt):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite output: {path}")
    config = json.loads(args.config.read_text())
    base = args.config.resolve().parent
    rows = config["clips"]
    result = export_comparison(
        [base / row["video"] for row in rows], labels=[row["label"] for row in rows],
        physical_complete=[row["physical_complete"] for row in rows],
        calibrations=[json.loads((base / row["camera"]).read_text()) for row in rows],
        frames=config["frames"], fps=config["fps"], source_size=config["source_size"],
        tile_size=config.get("tile_size", (640, 360)), output=args.output, font_path=args.font)
    with receipt.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
