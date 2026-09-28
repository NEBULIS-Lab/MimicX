import hashlib
import json
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
