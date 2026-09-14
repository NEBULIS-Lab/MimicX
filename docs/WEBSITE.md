# Project Website

Open `index.html` directly in a browser. The static page uses local CSS,
JavaScript, images and H.264 videos; it does not require a build server, CDN,
analytics account or runtime data fetch.

## Layout and Theme

The hero uses the author's human/G1 X mark, combined with self-hosted
Montserrat Bold text as a centered MimicX wordmark. Charcoal-to-terracotta
letterforms and coral i dots sit above the bold black full paper title.
Navigation and the bottom image information strip use dark translucent
backgrounds. The same original X supplies the site's transparent favicon.
Brand sources, rights notes and resizing provenance are in `assets/branding/`;
the font and its OFL license are in `assets/fonts/`. `assets/css/brand.css`
owns the hero branding; the rest of the site's layout remains in `mimicx.css`.
Top navigation uses a unified dark-button treatment with unmodified Simple
Icons brand marks and providers in parentheses. The arXiv link temporarily
points to `https://example.com` at the author's request; replace it after
upload. Release links are repeated in the compact footer rather than a
standalone release section. The upper bar uses 82% opacity and the lower strip
retains 58%. Only the scene image gently
desaturates/fades when the pointer leaves the hero; the wordmark and text stay
unchanged. Touch devices retain full color, keyboard focus restores full
color, and reduced-motion preferences disable the transition.

The static image hero, section navigation, method and experiment sequence adapt the
RoboSplat project page structure. MimicX uses coral `#D45B4C`, pale coral
`#FBECE9`, blue `#3B78A8`, teal `#23866B`, grey `#7A7F87` and ink `#20252B`.
Solid robots represent executed policies; transparent robots represent the
reference. Presentation-scene props are not evidence of ball-contact training.

Numerical CSVs, provenance and metric definitions live only in `assets/results`.
The page contains all four core tasks and links every CSV, rather than selecting
only favorable outcomes. Videos are illustrative recorded trials, not an
aggregation of verification repeats. Exact source filenames and byte hashes
are recorded in `assets/media/manifest.json`. Three transparent reconstruction
and reference assets have web derivatives: trim alpha-only empty margins,
downsample without changing aspect ratio, and center on a transparent canvas.
The manifest records both source and derivative hashes and the crop bounds.
Other displayed images and all plots/videos are byte-identical source copies.

The six workflow images contain no added captions; labels are native HTML.
The first four stages show input and motion preparation, while the last two
show Fixed Reference and the verified policy at recorded step 131. The input
illustrates the source video; it is not claimed to be a frame-synchronous
six-way comparison. Four square SVG plots share one desktop row: tracking
channels, paired execution horizons, dense tennis training dynamics and HLoop
wall time. Each links to its full-size vector original.

## Author-Drawn Overview

`index.html` reserves `figure#method-overview` before the workflow frames.
When the author supplies the illustration, replace that figure's placeholder
contents with an image and descriptive alt text, remove the
`overview-placeholder` class, and register the new asset in the media manifest.
Do not substitute a generated diagram or experimental figure montage.

## Maintenance

Header and footer links point to the published
[Policies](https://huggingface.co/Shuaijun/MimicX-Policies) and
[Assets](https://huggingface.co/datasets/Shuaijun/MimicX-Assets) repositories.
The recommended-policy entry supports one-task downloads; complete artifacts
remain on HF, not duplicated in the website. Keep these links synchronized
when updating release navigation.

```bash
python scripts/build_website_results.py
python -m pytest tests/test_website_release.py -q
```

Run from the repository root. The builder creates the HTML table and download
links from the website's CSV files. It never writes result data into code or
training configuration directories. To update approved media, use
`python scripts/package_website_media.py --source /path/to/approved-media --prompt-source /path/to/prompt-materials`
(requires Pillow) and review
desktop/mobile playback before publishing.

For the optional browser audit, install `playwright` in a development
environment, install its Chromium browser, and run:

```bash
python scripts/verify_website.py --chromium /path/to/chromium --output runs/website-qa
```

This check disables GPU rendering, starts a temporary local HTTP server, tests
all four paired-video controls at desktop/tablet/mobile/landscape sizes, writes screenshots,
and shuts down the browser and server. It is not required to view the page.

For GitHub Pages, select **Deploy from a branch**, branch **main**, folder
**/docs** in repository Pages settings. The expected address is
`https://nebulis-lab.github.io/MimicX/` once Pages is enabled. Pushing the source
alone does not establish that Pages has been enabled or deployed.

## Attribution

Template structure: [RoboSplat](https://yangsizhe.github.io/robosplat/), based on
[NeRFies](https://nerfies.github.io/) and
[UMI on Legs](https://github.com/umi-on-legs/umi-on-legs.github.io/).
Adapted website template files (`index.html`, `assets/css/mimicx.css` and
`assets/js/mimicx.js`) use CC BY-SA 4.0. Bulma retains its embedded MIT notice.
MimicX research code remains Apache-2.0. Research videos, human video frames,
robot assets and scene imagery retain their respective rights; the template
license does not relicense those assets. Original analytics and third-party
tracking scripts were not copied.
