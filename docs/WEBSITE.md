# Project Website

Open `index.html` directly in a browser. The static page uses local CSS,
JavaScript, images and H.264 videos; it does not require a build server, CDN,
analytics account or runtime data fetch.

## Layout and Theme

The video hero, section navigation, method and experiment sequence adapt the
RoboSplat project page structure. MimicX uses coral `#D45B4C`, pale coral
`#FBECE9`, blue `#3B78A8`, teal `#23866B`, grey `#7A7F87` and ink `#20252B`.
Solid robots represent executed policies; transparent robots represent the
reference. Presentation-scene props are not evidence of ball-contact training.

Numerical CSVs, provenance and metric definitions live only in `assets/results`.
The page contains all four core tasks and links every CSV, rather than selecting
only favorable outcomes. Videos are illustrative recorded trials, not an
aggregation of verification repeats. Exact source filenames and byte hashes
are recorded in `assets/media/manifest.json`.

## Maintenance

Header and resource links point to the published
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
`scripts/package_website_media.py --source /path/to/approved-media` and review
desktop/mobile playback before publishing.

For the optional browser audit, install `playwright` in a development
environment, install its Chromium browser, and run:

```bash
python scripts/verify_website.py --chromium /path/to/chromium --output runs/website-qa
```

This check disables GPU rendering, starts a temporary local HTTP server, tests
all four paired-video controls at desktop/mobile sizes, writes screenshots,
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
