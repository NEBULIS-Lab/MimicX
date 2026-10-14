import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image, ImageChops


SITE = Path(__file__).resolve().parents[1] / "docs"


def test_compact_hero_derivatives_are_traced_and_smaller():
    records = json.loads((SITE / "assets/hero-delivery-manifest.json").read_text())
    assert len(records) == 2
    for row in records:
        source, target = SITE / row["source"], SITE / row["file"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row["source_sha256"]
        assert hashlib.sha256(target.read_bytes()).hexdigest() == row["sha256"]
        assert target.stat().st_size == row["bytes"]
        assert target.stat().st_size < source.stat().st_size * .55
    with Image.open(SITE / "assets/media/hero-tennis.png") as source:
        with Image.open(SITE / "assets/media/hero-tennis.webp") as target:
            assert source.size == target.size == (4480, 1600)


def test_logo_geometry_and_embedded_rgba_are_unchanged():
    roots = [ET.parse(SITE / f"assets/branding/{name}").getroot() for name in
             ("mimicx-full-logo.svg", "mimicx-full-logo.web.svg")]
    nodes = [list(root.iter()) for root in roots]
    assert len(nodes[0]) == len(nodes[1])
    for source, target in zip(*nodes):
        assert source.tag == target.tag
        assert source.text == target.text
        assert source.attrib.keys() == target.attrib.keys()
        for key, value in source.attrib.items():
            other = target.attrib[key]
            if value.startswith("data:image/png;base64,"):
                assert other.startswith("data:image/webp;base64,")
                a = Image.open(BytesIO(base64.b64decode(value.split(",", 1)[1]))).convert("RGBA")
                b = Image.open(BytesIO(base64.b64decode(other.split(",", 1)[1]))).convert("RGBA")
                assert a.size == b.size
                assert all(channel.getbbox() is None for channel in ImageChops.difference(a, b).split())
            else:
                assert value == other


def test_website_uses_compact_assets_but_readme_retains_original():
    html = (SITE / "index.html").read_text()
    assert 'src="assets/media/hero-tennis.webp"' in html
    assert 'src="assets/branding/mimicx-full-logo.web.svg"' in html
    assert 'href="assets/branding/logo-x.svg"' not in html
    assert 'href="assets/branding/favicon-192.png"' in html
    assert 'docs/assets/branding/mimicx-full-logo.svg' in (SITE.parent / "README.md").read_text()
