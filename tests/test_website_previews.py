import hashlib
import json
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

SITE = Path(__file__).resolve().parents[1] / 'docs'


class Media(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.videos, self.links, self.inputs = [], [], []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        if tag in ('video', 'a', 'input'):
            getattr(self, {'video': 'videos', 'a': 'links', 'input': 'inputs'}[tag]).append(dict(attrs))


def test_inline_videos_only_load_previews_on_demand_and_keep_hd_links():
    media = Media((SITE / 'index.html').read_text())
    assert len(media.videos) == 34
    hd_links = {a['href'] for a in media.links if a.get('data-viewer') == 'video'}
    for video in media.videos:
        assert 'src' not in video, 'Do not let the parser load offscreen originals'
        assert video['preload'] == 'none'
        assert video['data-src'].startswith('assets/media/previews/')
        assert (SITE / video['data-src']).is_file()
        assert video['data-original-src'] in hd_links
        assert video.get('data-poster') and 'poster' not in video
    assert any(a.get('id') == 'low-data-mode' and a.get('type') == 'checkbox' for a in media.inputs)


def test_preview_manifest_preserves_sources_timing_and_faststart():
    path = SITE / 'assets/media/previews/manifest.json'
    assert path.exists(), 'Each derivative needs a checked source and encoding record'
    manifest = json.loads(path.read_text())
    assert len(manifest['videos']) == 34
    for row in manifest['videos']:
        original, preview = SITE / row['source'], SITE / row['preview']
        assert hashlib.sha256(original.read_bytes()).hexdigest() == row['source_sha256']
        assert hashlib.sha256(preview.read_bytes()).hexdigest() == row['preview_sha256']
        assert row['width'] <= 854 and row['height'] <= 480
        assert 0 < row['fps'] <= 30.01
        assert abs(row['duration'] - row['source_duration']) <= 1 / row['fps'] + .002
        assert row['full_decode_passed'] and row['fast_start']
        assert row['bytes'] < row['source_bytes']
        assert row['frames'] > 0


def test_preview_loader_preserves_interrupted_resume_and_allows_restart():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required for the media lifecycle regression')
    script = r'''
const fs = require('fs'), vm = require('vm'), assert = require('assert/strict');
class Video extends EventTarget {
  constructor() { super(); this.dataset = {src:'preview.mp4'}; this.attrs = new Set(); this.currentTime = 0; this.duration = 10; this.readyState = 0; }
  hasAttribute(name) { return this.attrs.has(name); }
  removeAttribute(name) { this.attrs.delete(name); }
  set src(value) { this.attrs.add('src'); }
  pause() {}
  load() { this.readyState = 0; this.currentTime = 0; }
  metadata() { this.readyState = 1; this.dispatchEvent(new Event('loadedmetadata')); }
}
const toggle = new EventTarget();
const document = new EventTarget();
document.getElementById = () => toggle;
document.querySelectorAll = () => [];
document.documentElement = {dataset:{}};
const context = {document, navigator:{}, localStorage:{getItem:()=>null}, window:{}, Event,
  IntersectionObserver:class {observe() {}}};
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), context);
const media = context.window.MimicXMedia, video = new Video();
media.prepare(video); video.metadata(); video.currentTime = 5; media.release(video);
media.prepare(video); media.release(video); // Leave before metadata restores the position.
media.prepare(video); video.metadata();
assert.equal(video.currentTime, 5, 'Interrupted reload must not overwrite the saved position');
media.release(video); media.prepare(video); media.reset(video); video.metadata();
assert.equal(video.currentTime, 0, 'Carousel reselection must cancel pending resume');
media.release(video); media.prepare(video); video.metadata();
assert.equal(video.currentTime, 0, 'Reset must also replace the stored position');
'''
    subprocess.run([node, '-e', script, str(SITE / 'assets/js/video-loading.js')], check=True)
