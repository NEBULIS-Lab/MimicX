"""Create checked low-bandwidth derivatives without altering original recordings."""
import argparse
import hashlib
import html
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import struct
import subprocess
from urllib.parse import urlsplit

import imageio_ffmpeg


class Videos(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.tags = []
        self.div_depth = 0
        self.recording_depth = None
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        if tag == 'div':
            self.div_depth += 1
            if 'recording-frame' in dict(attrs).get('class', '').split():
                self.recording_depth = self.div_depth
        if tag == 'video' and self.recording_depth is not None:
            self.tags.append((self.get_starttag_text(), dict(attrs)))

    def handle_endtag(self, tag):
        if tag == 'div':
            if self.div_depth == self.recording_depth:
                self.recording_depth = None
            self.div_depth -= 1


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata(path):
    reader = imageio_ffmpeg.read_frames(str(path))
    try:
        return next(reader)
    finally:
        reader.close()


def fast_start(path):
    atoms = []
    with path.open('rb') as stream:
        while header := stream.read(8):
            if len(header) != 8:
                return False
            size, kind = struct.unpack('>I4s', header)
            head_size = 8
            if size == 1:
                size = struct.unpack('>Q', stream.read(8))[0]
                head_size = 16
            atoms.append(kind)
            if size == 0:
                break
            if size < head_size:
                return False
            stream.seek(size - head_size, 1)
    return b'moov' in atoms and b'mdat' in atoms and atoms.index(b'moov') < atoms.index(b'mdat')


def prepare(site, ffmpeg, rewrite=True):
    site = site.resolve()
    os.environ['IMAGEIO_FFMPEG_EXE'] = ffmpeg
    page = site / 'index.html'
    text = page.read_text()
    destination = site / 'assets/media/previews'
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / 'manifest.json'
    previous = json.loads(manifest.read_text()) if manifest.exists() else {}
    old = {r['source']: r for r in previous.get('videos', [])} if previous.get('encoding', {}).get('fps_filter') == 'round=near' else {}
    records = {}
    for tag, attrs in Videos(text).tags:
        original_url = attrs.get('data-original-src', attrs.get('src'))
        source_rel = urlsplit(original_url).path
        source = (site / source_rel).resolve()
        if not source.is_relative_to(site) or not source.is_file():
            raise ValueError(f'Invalid local source: {source_rel}')
        preview = destination / ('--'.join(Path(source_rel).parts[2:]))
        relative = preview.relative_to(site).as_posix()
        source_sha = sha(source)
        record = old.get(source_rel)
        if source_rel in records:
            record = records[source_rel]
        elif not (record and record['source_sha256'] == source_sha and preview.is_file()
                  and sha(preview) == record['preview_sha256']):
            meta = metadata(source)
            fps = min(30., meta['fps'])
            tmp = preview.with_suffix('.tmp.mp4')
            subprocess.run([ffmpeg, '-nostdin', '-hide_banner', '-v', 'error', '-y',
                            '-hwaccel', 'none', '-threads', '2', '-i', str(source),
                            '-map', '0:v:0', '-map_metadata', '-1', '-an',
                            '-vf', "scale='min(854,iw)':'min(480,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2:flags=lanczos," + f'fps={fps}:round=near',
                            '-c:v', 'libx264', '-threads', '2', '-preset', 'slow',
                            '-crf', '27', '-maxrate', '600k', '-bufsize', '1200k',
                            '-g', str(round(fps)), '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(tmp)], check=True)
            out = metadata(tmp)
            frames, duration = imageio_ffmpeg.count_frames_and_secs(str(tmp))
            if abs(duration - meta['duration']) > 1 / fps + .012:
                raise ValueError(f'Timing changed: {source_rel}: {duration} vs {meta["duration"]}')
            if not fast_start(tmp):
                raise ValueError(f'Fast Start missing: {source_rel}')
            tmp.replace(preview)
            record = dict(source=source_rel, preview=relative, source_sha256=source_sha,
                          preview_sha256=sha(preview), source_bytes=source.stat().st_size,
                          bytes=preview.stat().st_size, source_duration=meta['duration'],
                          duration=duration, width=out['size'][0], height=out['size'][1],
                          fps=out['fps'], frames=frames, full_decode_passed=True, fast_start=True)
        records[source_rel] = record
        print(json.dumps(record), flush=True)
        # The full-resolution target remains on the existing viewer link.
        attrs.pop('src', None)
        if 'poster' in attrs:
            attrs['data-poster'] = attrs.pop('poster')
        attrs.update({'data-src': relative, 'data-original-src': original_url, 'preload': 'none'})
        replacement = '<video ' + ' '.join(k if v is None else f'{k}="{html.escape(v, quote=True)}"'
                                           for k, v in attrs.items()) + '>'
        text = text.replace(tag, replacement, 1)
        manifest.write_text(json.dumps(dict(schema='mimicx.web-previews.v1',
            encoding=dict(codec='libx264', crf=27, maxrate_kbps=600, max_height=480,
                          max_width=854, max_fps=30, fps_filter='round=near', audio=False, original_media_unchanged=True),
            videos=list(records.values())), indent=2) + '\n')
    if rewrite:
        page.write_text(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, required=True)
    parser.add_argument('--ffmpeg', default=imageio_ffmpeg.get_ffmpeg_exe())
    args = parser.parse_args()
    prepare(args.site, args.ffmpeg)
