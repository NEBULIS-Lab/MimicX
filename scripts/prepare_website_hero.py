"""Build compact first-screen media without resizing or changing brand geometry."""

import argparse
import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image


def build(site):
    records = []

    def record(source, target, encoding):
        records.append({
            "source": source.relative_to(site).as_posix(),
            "file": target.relative_to(site).as_posix(),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "source_bytes": source.stat().st_size,
            "bytes": target.stat().st_size,
            "encoding": encoding,
        })

    source = site / "assets/media/hero-tennis.png"
    target = source.with_suffix(".webp")
    with Image.open(source) as image:
        image.save(target, "WEBP", quality=96, method=6)
    record(source, target, "WebP quality 96; original 4480 x 1600 resolution")

    source = site / "assets/branding/mimicx-full-logo.svg"
    target = source.with_name("mimicx-full-logo.web.svg")
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    tree = ET.parse(source)
    images = 0
    for node in tree.iter():
        for key, value in list(node.attrib.items()):
            if not value.startswith("data:image/png;base64,"):
                continue
            with Image.open(BytesIO(base64.b64decode(value.split(",", 1)[1]))) as image:
                encoded = BytesIO()
                image.save(encoded, "WEBP", lossless=True, quality=100, method=6, exact=True)
            node.set(key, "data:image/webp;base64," + base64.b64encode(encoded.getvalue()).decode("ascii"))
            images += 1
    if images != 1:
        raise ValueError(f"Expected one embedded brand image, found {images}")
    tree.write(target, encoding="utf-8", xml_declaration=True)
    record(source, target, "SVG geometry unchanged; embedded RGBA image encoded as lossless WebP")

    (site / "assets/hero-delivery-manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=Path(__file__).resolve().parents[1] / "docs")
    build(parser.parse_args().site.resolve())
