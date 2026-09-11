"""Export favicon sizes from the unmodified, author-designed transparent X."""

import hashlib
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BRAND = ROOT / 'docs/assets/branding'
SOURCE = BRAND / 'logo-x.png'


def main():
    records = []
    with Image.open(SOURCE) as image:
        if image.mode != 'RGBA' or image.getchannel('A').getextrema()[0] != 0:
            raise ValueError('The author logo must retain transparent RGBA pixels')
        records.append({'file': SOURCE.name, 'role': 'Unmodified author-designed X',
                        'sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest()})
        for name, size in [('favicon-32.png', 32), ('favicon-192.png', 192), ('apple-touch-icon.png', 180)]:
            target = BRAND / name
            image.resize((size, size), Image.Resampling.LANCZOS).save(target, optimize=True)
            records.append({'file': name, 'operation': f'Lanczos resize to {size}x{size}; preserve alpha',
                            'source': SOURCE.name, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    (BRAND / 'manifest.json').write_text(json.dumps(records, indent=2) + '\n')
    print(f'Prepared {len(records)} brand assets')


if __name__ == '__main__':
    main()
