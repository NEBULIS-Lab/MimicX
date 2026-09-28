"""Export CPU-only web derivatives of existing paper figures and recordings."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(project, ffmpeg, destination):
    import fitz
    from PIL import Image
    destination.mkdir(parents=True, exist_ok=True)
    records = []

    def record(target, source, **extra):
        records.append(dict(file=target.name, source=source.name, source_sha256=digest(source),
                            sha256=digest(target), **extra))

    def still(name, source, stage, width=2400):
        with Image.open(source) as original:
            im = original.convert('RGB')
            im.thumbnail((width, 1600), Image.Resampling.LANCZOS)
            target = destination / f'{name}.webp'
            im.save(target, quality=92, method=6)
            record(target, source, stage=stage, width=im.width, height=im.height)

    def video(name, source, frame=0):
        target = destination / f'{name}.mp4'
        subprocess.run([ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                        '-hwaccel', 'none', '-i', str(source), '-map_metadata', '-1',
                        '-an', '-vf', 'scale=-2:720', '-c:v', 'libx264', '-crf', '25',
                        '-preset', 'medium', '-threads', '4', '-movflags', '+faststart', str(target)], check=True)
        record(target, source, stage='Policy rollout', transform='720p; full duration; original timebase; no audio')
        poster = destination / f'{name}.webp'
        subprocess.run([ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                        '-hwaccel', 'none', '-i', str(source), '-vf', f'select=eq(n\\,{frame}),scale=-2:720',
                        '-frames:v', '1', '-c:v', 'libwebp', '-quality', '90', '-threads', '2', str(poster)], check=True)
        record(poster, source, stage='Policy rollout', frame=frame)

    flat = project / 'paper-review-media-flat'
    still('parkour-sequence', project / '0.prompt/04_CONTINUOUS_EVOLUTION_WIDE/EVOLUTION__PARKOUR__SOURCE_COORDINATES__QA.png', 'Motion sequence', 3300)
    still('parkour-reference', project / 'code-single-human-paper-native/runs/parkour_park1_20260914/review-media-flat/PARKOUR__SCENE_REPAIRED_REFERENCE__SCENE_FK__FRAME000025__1080P.png', 'Robot reference')
    for name, filename, frame in [
        ('forest', 'PAPER_COLLISION__POLICY__FOREST__REFINED_CONTINUATION__SEED2002.mp4', 202),
        ('stairs', 'PAPER_COLLISION__POLICY__STAIRS__SELECTED__NO_GHOST.mp4', 0),
        ('track', 'PAPER_COLLISION__POLICY__TRACK__SELECTED__NO_GHOST.mp4', 0),
        ('platform', 'PAPER_COLLISION__POLICY__BENCH__SELECTED__NO_GHOST.mp4', 28),
        ('tennis-scene', 'VIDEO_SCENE__TENNIS__MIMICX__SOURCE_REGISTERED.mp4', 196),
        ('football-scene', 'VIDEO_SCENE__FOOTBALL_JUGGLING__MIMICX__SOURCE_REGISTERED.mp4', 0),
    ]:
        video(name, flat / filename, frame)
    source = project / 'paper-repo-iclr/figures/assets/registered_scene_stages.pdf'
    with fitz.open(source) as document:
        embedded = document.extract_image(document[0].get_images(full=True)[0][0])
        im = Image.open(io.BytesIO(embedded['image'])).convert('RGB')
        w, h = im.size
        for i, stage in enumerate(('input', 'human', 'simulation', 'scene')):
            box = (round(w*(.006+i*.249)), 125, min(round(w*(.249+i*.249)), w), h-40)
            panel = im.crop(box)
            panel.thumbnail((1200, 1000), Image.Resampling.LANCZOS)
            target = destination / f'tennis-stage-{stage}.webp'
            panel.save(target, quality=94, method=6)
            record(target, source, stage=stage, crop=list(box), width=panel.width, height=panel.height,
                   selection='Unchanged paper-selected four-stage correspondence')
    (destination / 'manifest.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--destination', type=Path, default=Path(__file__).resolve().parents[1] / 'docs/assets/media/showcase')
    args = parser.parse_args()
    prepare(args.project, args.ffmpeg, args.destination)
