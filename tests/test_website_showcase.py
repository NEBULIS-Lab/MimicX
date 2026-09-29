import hashlib
import json
import re
from pathlib import Path

SITE = Path(__file__).resolve().parents[1] / 'docs'


def test_showcase_order_and_media_inventory():
    html = (SITE / 'index.html').read_text()
    assert html.index('</header>') < html.index('class="research-nav"') < html.index('id="showcase"')
    if 'id="authors"' in html:
        assert html.index('id="authors"') < html.index('id="showcase"')
    assert html.index('id="showcase"') < html.index('id="policies"') < html.index('id="overview"')
    assert 'class="rx-summary"' not in html
    assert html.count('data-viewer="video"') == 4
    gallery = html.split('class="showcase-track"',1)[1].split('class="showcase-pagination"',1)[0]
    assert '<video' not in gallery
    assert gallery.count('<figure') == 8
    assert html.count('tennis-stage-') == 8
    for row in json.loads((SITE / 'assets/media/showcase/manifest.json').read_text()):
        path = SITE / 'assets/media/showcase' / row['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        assert '/' not in row['source']


def test_showcase_accessibility_and_preserved_dimensions():
    js = (SITE / 'assets/js/showcase.js').read_text() + (SITE / 'assets/js/media-viewer.js').read_text()
    css = (SITE / 'assets/css/research.css').read_text()
    for behavior in ('prefers-reduced-motion', 'visibilitychange', 'dialog.close()', 'aria-hidden', 'preventScroll', 'IntersectionObserver'):
        assert behavior in js
    assert 'object-fit: contain' in css
    assert 'grid-template-columns: 64fr 36fr' in css
    assert '#method-overview { width: 100%' in css


def test_album_zoom_and_author_placement():
    html = (SITE / 'index.html').read_text()
    assert 'embla-carousel.umd.js' in html and 'panzoom.min.js' in html
    assert 'maximum-scale' not in html and 'user-scalable=no' not in html
    assert 'data-viewer="table"' in html
    assert 'class="method-ledger"' in html
    if 'id="authors"' in html:
        assert html.index('class="hero-bottom"') < html.index('id="authors"') < html.index('</header>')
    css = (SITE / 'assets/css/experience.css').read_text()
    assert 'pan-y pinch-zoom' in css
    for library in ('embla', 'panzoom'):
        assert 'MIT' in (SITE / f'assets/vendor/{library}-LICENSE.txt').read_text()


def test_gallery_sources_remain_uncropped():
    rows = json.loads((SITE / 'assets/media/gallery/manifest.json').read_text())
    assert len(rows) == 6
    for row in rows:
        path = SITE / 'assets/media/gallery' / row['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        assert row.get('crop') is None


def test_author_line_and_inline_policy_summary():
    html = (SITE / 'index.html').read_text()
    authors = re.search(r'<div class="rx-authors"[^>]*>\s*<p>(.*?)</p>', html, re.S)
    if authors:
        assert authors[1].startswith('<span>')
        assert authors[1].count('<span>') == 10
    summary = re.search(r'<p class="rx-policy-summary">(.*?)</p>', html, re.S)
    assert summary
    for field in ('task-error', 'task-horizon', 'task-note', 'policy-protocol'):
        assert f'id="{field}"' in summary[1]
    assert 'class="rx-task-results"' not in html


def test_clean_hero_and_scroll_linked_album():
    html = (SITE / 'index.html').read_text()
    hero = html.split('<header', 1)[1].split('</header>', 1)[0]
    assert 'hero-context' not in hero
    assert 'hero-expand' not in hero
    assert 'View policy rollouts' not in hero
    assert '<html lang="en" data-theme="dark">' in html
    js = (SITE / 'assets/js/showcase.js').read_text()
    assert 'const AUTOPLAY_MS = 4000' in js
    assert "on('scroll', requestPaint)" in js


def test_motion_lifecycle_and_atomic_workflow():
    js = (SITE / 'assets/js/showcase.js').read_text()
    assert 'cancelAnimationFrame(autoplayFrame)' in js
    assert 'pagination.scrollTo' in js
    assert 'await Promise.all' in js and 'workflowRequest' in js
    assert 'aria-busy' in js
    viewer = (SITE / 'assets/js/media-viewer.js').read_text()
    assert "addEventListener('cancel'" in viewer
    assert 'function closeViewer()' in viewer
    assert 'prefers-reduced-motion' in viewer
    assert 'viewer-closing' in viewer
