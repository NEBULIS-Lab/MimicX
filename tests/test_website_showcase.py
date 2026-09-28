import hashlib
import json
from pathlib import Path

SITE = Path(__file__).resolve().parents[1] / 'docs'


def test_showcase_order_and_media_inventory():
    html = (SITE / 'index.html').read_text()
    assert html.index('</header>') < html.index('id="showcase"') < html.index('class="research-nav"')
    assert 'class="rx-summary"' not in html
    assert html.count('data-viewer="video"') == 6
    assert html.count('tennis-stage-') == 8
    for row in json.loads((SITE / 'assets/media/showcase/manifest.json').read_text()):
        path = SITE / 'assets/media/showcase' / row['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        assert '/' not in row['source']


def test_showcase_accessibility_and_preserved_dimensions():
    js = (SITE / 'assets/js/showcase.js').read_text()
    css = (SITE / 'assets/css/research.css').read_text()
    for behavior in ('prefers-reduced-motion', 'visibilitychange', 'dialog.close()', 'aria-hidden', 'preventScroll', 'IntersectionObserver'):
        assert behavior in js
    assert 'object-fit: contain' in css
    assert '#method-overview { width: 65%' in css
    assert '#method-overview { width: 100%' in css
