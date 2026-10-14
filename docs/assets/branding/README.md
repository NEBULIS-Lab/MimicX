# MimicX Brand Assets

`mimicx-full-logo.svg` is the author's updated complete MimicX wordmark. It is
used directly in the repository README and retained as the website master..

`logo-x.svg` is the matching standalone human/G1 X master..

`logo-x.png` is the author's original transparent X, formed by a human model
and a G1 robot. Its bytes are unchanged. It is a project mark, not an
experimental result, and must not be replaced by generated artwork.

The website no longer constructs the mark from live text and a separate X.
The complete author-provided SVG is the single visual source.

`logo-x.png`, `favicon-32.png`, `favicon-192.png` and `apple-touch-icon.png`
provide the PNG master and compact browser icons. `manifest.json` lists the files. Site
styling is in `../css/brand.css`.

The website hero uses `mimicx-full-logo.web.svg`: identical vector geometry,
text and transparency, with the embedded RGBA bitmap encoded as lossless WebP.
The browser favicon uses the existing 32/192 px PNGs rather than downloading
the roughly 1 MB SVG master for a tiny icon. The original files are unchanged.
Rebuild the delivery derivatives with `python scripts/prepare_website_hero.py`.
`../hero-delivery-manifest.json` records source files, delivery formats and sizes.

The X is author-provided branding. The project's Apache-2.0 code license does
not grant rights to third-party marks or independently licensed robot assets.
