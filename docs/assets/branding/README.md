# MimicX Brand Assets

`mimicx-full-logo.svg` is the author's updated complete MimicX wordmark. It is
used directly in the repository README and retained as the website master. Its SHA-256 digest
is `3facedeea2d814082d5fa42672dc7cdefea67f94ec02ef03af6e4a45f8abb824`.

`logo-x.svg` is the matching standalone human/G1 X master. Its SHA-256 digest is
`491f61166b1d2c79eaa58abe945d011705aa44f00c2333ab3f51d5e600fdb8ca`.

`logo-x.png` is the author's original transparent X, formed by a human model
and a G1 robot. Its bytes are unchanged. It is a project mark, not an
experimental result, and must not be replaced by generated artwork.

The website no longer constructs the mark from live text and a separate X.
The complete author-provided SVG is the single visual source.

`logo-x.png`, `favicon-32.png`, `favicon-192.png` and `apple-touch-icon.png`
provide the PNG master and compact browser icons. `manifest.json` records their hashes. Site
styling is in `../css/brand.css`.

The website hero uses `mimicx-full-logo.web.svg`: identical vector geometry,
text and transparency, with the embedded RGBA bitmap encoded as lossless WebP.
The browser favicon uses the existing 32/192 px PNGs rather than downloading
the roughly 1 MB SVG master for a tiny icon. The original files are unchanged.
Rebuild the delivery derivatives with `python scripts/prepare_website_hero.py`.
`../hero-delivery-manifest.json` records source and delivery hashes and sizes.

The X is author-provided branding. The project's Apache-2.0 code license does
not grant rights to third-party marks or independently licensed robot assets.
