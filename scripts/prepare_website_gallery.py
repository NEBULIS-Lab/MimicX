"""Make uncropped web stills from the reviewed continuous-evolution figures."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image


def prepare(project, destination, ffmpeg):
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for folder, task in [('05_CONTINUOUS_EVOLUTION_TRACK', 'TRACK'),
                         ('06_CONTINUOUS_EVOLUTION_STAIRS', 'STAIRS'),
                         ('07_CONTINUOUS_EVOLUTION_BENCH', 'BENCH'),
                         ('08_CONTINUOUS_EVOLUTION_FOREST', 'FOREST')]:
        source = project / '0.prompt' / folder / f'EVOLUTION__{task}__CLEAN.png'
        target = destination / f'{task.lower()}-evolution.webp'
        with Image.open(source) as original:
            im = original.convert('RGB')
            im.thumbnail((3300, 1600), Image.Resampling.LANCZOS)
            im.save(target, quality=93, method=6)
        records.append(dict(file=target.name, source=source.name,
                            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                            sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                            stage='Source, reconstruction and retargeted reference',
                            width=im.width, height=im.height, crop=None))
    source = project / 'paper-review-media-flat/PAPER_COLLISION__DIAGNOSTIC__POLICY__PARKOUR__SELECTED__NO_GHOST.mp4'
    target = destination / 'parkour-policy.mp4'
    subprocess.run([ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                    '-hwaccel', 'none', '-i', str(source), '-map_metadata', '-1', '-an',
                    '-vf', 'scale=-2:720', '-c:v', 'libx264', '-crf', '24', '-threads', '2',
                    '-movflags', '+faststart', str(target)], check=True)
    poster = destination / 'parkour-policy.webp'
    subprocess.run([ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                    '-hwaccel', 'none', '-i', str(source), '-vf', 'select=eq(n\\,80),scale=-2:720',
                    '-frames:v', '1', '-c:v', 'libwebp', '-quality', '92', '-threads', '2',
                    str(poster)], check=True)
    for file in (target, poster):
        records.append(dict(file=file.name, source=source.name,
                            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                            sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                            stage='Recorded diagnostic policy rollout',
                            full_horizon_success=False, frame=80 if file == poster else None))
    (destination / 'manifest.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True, type=Path)
    parser.add_argument('--ffmpeg', default='ffmpeg')
    args = parser.parse_args()
    prepare(args.project, Path(__file__).resolve().parents[1] / 'docs/assets/media/gallery', args.ffmpeg)
