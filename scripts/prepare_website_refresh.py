"""Export CPU-only website derivatives from approved paper art and recordings."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def prepare(artwork, ffmpeg='ffmpeg'):
    import fitz
    from PIL import Image

    media = ROOT / 'docs/assets/media'
    output = media / 'research'
    output.mkdir(exist_ok=True)
    records = []
    for name, filename in [('refinement', 'mimicx_refinement.pdf'),
                           ('verification', 'mimicx_verification_hloop.pdf')]:
        source = artwork / filename
        with fitz.open(source) as doc:
            pix = doc[0].get_pixmap(matrix=fitz.Matrix(2400/doc[0].rect.width, 2400/doc[0].rect.width), alpha=False)
            image = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
            target = output / f'{name}.webp'
            image.save(target, quality=94, method=6)
        records.append({'file': target.name, 'source': source.name,
                        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                        'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    for task in ('tennis', 'football', 'dance', 'kungfu'):
        for method in ('fixed', 'ours'):
            source = media / f'{task}-{method}.mp4'
            target = output / f'{task}-{method}.jpg'
            subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-hwaccel', 'none',
                            '-y', '-ss', '0.1', '-i', str(source), '-frames:v', '1',
                            '-vf', 'scale=960:-2', '-q:v', '3', str(target)], check=True)
            records.append({'file': target.name, 'source': source.name,
                            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                            'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    (output / 'manifest.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artwork', required=True, type=Path)
    parser.add_argument('--ffmpeg', default='ffmpeg')
    args = parser.parse_args()
    prepare(args.artwork, args.ffmpeg)
