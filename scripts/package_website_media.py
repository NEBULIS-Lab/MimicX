#!/usr/bin/env python3
"""Copy approved MimicX media into the website with a public provenance index."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MEDIA = {
    "tennis-court.mp4": "VIDEO_SCENE__TENNIS__MIMICX__COURT_WIDE.mp4",
    "tennis-fixed.mp4": "VIDEO__TENNIS__FIXED_REFERENCE__WITH_REFERENCE.mp4",
    "tennis-ours.mp4": "VIDEO__TENNIS__MIMICX__WITH_REFERENCE.mp4",
    "football-fixed.mp4": "VIDEO__FOOTBALL__FIXED_REFERENCE__WITH_REFERENCE.mp4",
    "football-ours.mp4": "VIDEO__FOOTBALL__MIMICX__WITH_REFERENCE.mp4",
    "dance-fixed.mp4": "VIDEO__DANCE__FIXED_REFERENCE__WITH_REFERENCE.mp4",
    "dance-ours.mp4": "VIDEO__DANCE__MIMICX__WITH_REFERENCE.mp4",
    "kungfu-fixed.mp4": "VIDEO__KONGFU__FIXED_REFERENCE__WITH_REFERENCE.mp4",
    "kungfu-ours.mp4": "VIDEO__KONGFU__MIMICX__WITH_REFERENCE.mp4",
    "stage-video.png": "PIPELINE__V9__TENNIS__01_ORIGINAL_RGB.png",
    "stage-human.png": "PIPELINE__V9__TENNIS__03_SMPL_CAMERA.png",
    "stage-world.png": "PIPELINE__V9__TENNIS__04_SMPL_WORLD.png",
    "stage-reference.png": "PIPELINE__V9__TENNIS__05_RETARGETED_G1_REFERENCE.png",
    "stage-fixed.png": "PIPELINE__V9__TENNIS__06_FIXED_REFERENCE_POLICY.png",
    "stage-ours.png": "PIPELINE__V9__TENNIS__07_MIMICX_POLICY.png",
    "tracking-channels.png": "RESULT__V9__01_PAIRED_SEED_IMPROVEMENTS.png",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    destination = ROOT / "docs/assets/media"
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for name, original in MEDIA.items():
        source = args.source / original
        target = destination / name
        shutil.copy2(source, target)
        target.chmod(0o644)
        records.append({"file": name, "source_filename": original,
                        "bytes": target.stat().st_size,
                        "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    poster = destination / "tennis-court-poster.jpg"
    subprocess.run([args.ffmpeg, "-y", "-loglevel", "error", "-hwaccel", "none",
                    "-ss", "1", "-i", str(destination / "tennis-court.mp4"),
                    "-frames:v", "1", "-q:v", "2", "-update", "1", str(poster)], check=True)
    records.append({"file": poster.name, "source_filename": MEDIA["tennis-court.mp4"],
                    "operation": "CPU-decoded video frame at 1.0 seconds",
                    "bytes": poster.stat().st_size,
                    "sha256": hashlib.sha256(poster.read_bytes()).hexdigest()})
    (destination / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"Copied {len(MEDIA)} original media files and extracted one CPU video poster")


if __name__ == "__main__":
    main()
