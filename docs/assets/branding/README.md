# MimicX Brand Assets

`mimicx-full-logo.svg` is the author's complete MimicX wordmark. It is used
as the primary identity at the beginning of the repository README. Its
SHA-256 digest is
`7d2878daeaa47f21378460288e69c1a62c8e315e4d850381628c27b4d92cf0f9`.

`logo-x.png` is the author's original transparent X, formed by a human model
and a G1 robot. Its bytes are unchanged. It is a project mark, not an
experimental result, and must not be replaced by generated artwork.

The website combines live Montserrat Bold text (`Mimic`, weight 700) with this image as
the final `X`. The text transitions from charcoal to terracotta; both native
letter-i dots are coral. The complete wordmark has one accessible `MimicX`
name. There are no shadows, outlines or glow effects.

`favicon-32.png`, `favicon-192.png` and `apple-touch-icon.png` are transparent
resizes of the original. Run `python scripts/package_brand_assets.py` from the
repository root with Pillow installed to recreate them. `manifest.json`
records original and derivative hashes. Site styling is in `../css/brand.css`.

The X is author-provided branding. The project's Apache-2.0 code license does
not grant rights to third-party marks or independently licensed robot assets.
