#!/usr/bin/env python3
"""Copy approved MimicX media into the website with a public provenance index."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

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
    "hero-tennis.png": "MOTION__TENNIS__MIMICX__COURT_WIDE__7POSE.png",
    "readme-tennis-motion.png": "MOTION__TENNIS__MIMICX__ACTION_PANORAMA__7POSE.png",
    "hero-tennis-mobile.png": "FRAME__TENNIS__MIMICX__PIPELINE_CAMERA__STEP000201.png",
    "stage-human.png": "REFERENCE_ASSET__TENNIS__SMPLX_CAMERA__OBLIQUE__POSE05__FRAME000131__ALPHA.png",
    "stage-world.png": "REFERENCE_ASSET__TENNIS__SMPLX_GLOBAL__OBLIQUE__POSE05__FRAME000131__ALPHA.png",
    "stage-reference.png": "REFERENCE_ASSET__TENNIS__G1__OBLIQUE__POSE05__FRAME000131__ALPHA.png",
    "stage-fixed.png": "FRAME__TENNIS__FIXED_REFERENCE__SOURCE_REGISTERED__STEP000131.png",
    "stage-ours.png": "FRAME__TENNIS__MIMICX__SOURCE_REGISTERED__STEP000131.png",
    "tracking-channels.svg": "RESULT__V9__01_PAIRED_SEED_IMPROVEMENTS.svg",
    "execution-horizon.svg": "RESULT__V9__04_PAIRED_EXECUTION_HORIZON.svg",
    "training-body-error.svg": "RESULT__V9__07_DENSE_LEARNING_BODY_ERROR.svg",
    "hloop-wall-time.svg": "RESULT__V9__09_HLOOP_PAIRED_WALL_TIME.svg",
}
PROMPT_MEDIA = {"stage-video.png": "01_OVERVIEW_16x11/REF01_HUMAN_VIDEO.png"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prompt-source", type=Path, required=True)
    args = parser.parse_args()
    destination = ROOT / "docs/assets/media"
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for name, original in (MEDIA | PROMPT_MEDIA).items():
        source = (args.prompt_source if name in PROMPT_MEDIA else args.source) / original
        target = destination / name
        operation = {}
        if original.endswith('__ALPHA.png'):
            from PIL import Image

            with Image.open(source) as raw:
                rgba = raw.convert('RGBA')
                bounds = rgba.getchannel('A').getbbox()
                if bounds is None:
                    raise ValueError(f'Empty transparent source: {source.name}')
                pose = rgba.crop(bounds)
                pose.thumbnail((1152, 852), Image.Resampling.LANCZOS)
                canvas = Image.new('RGBA', (1200, 900))
                canvas.alpha_composite(pose, ((1200 - pose.width) // 2, (900 - pose.height) // 2))
                canvas.save(target, optimize=True)
                operation = {'operation': 'Trim alpha-only padding, fit pose within 1152x852, center on transparent 1200x900 web canvas',
                             'alpha_bounds': list(bounds),
                             'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
        else:
            shutil.copy2(source, target)
        target.chmod(0o644)
        records.append({"file": name, "source_filename": Path(original).name,
                        "bytes": target.stat().st_size,
                        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), **operation})
    (destination / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"Packaged {len(records)} media files with source provenance")


if __name__ == "__main__":
    main()
