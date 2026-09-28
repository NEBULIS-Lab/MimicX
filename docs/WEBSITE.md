# Project Website

Open `index.html` directly, or visit https://nebulis-lab.com/MimicX.
The site is static HTML, CSS and JavaScript with local images and recordings.
No build step, external font, CDN, analytics or runtime data fetch is needed.

## September 29 editorial update

The complete region after the hero and before the footer has been rebuilt:
navigation, overview, policy viewer, method, results and HLoop. This is a new
markup and component stylesheet, not a reskin of the previous middle sections.
D-JEPA's research page informed the compact typography, task navigation and
persistent light/dark presentation; its purple palette and application code
were not copied.

The approved hero and compact footer are preserved byte-for-byte. Their
styles remain in `mimicx.css` and `brand.css`. The new research body is owned
by `research.css`, scoped to `#research` and `.research-nav`. It must not
override hero/footer styles or their inherited color variables.

Dark is the default for the rebuilt body. The navigation's theme control
switches to light and stores the choice under `mimicx-theme`. It deliberately
does not recolor the preserved hero, artwork or footer. Storage-denied browsers
still support switching for the current visit. Semantic colors are coral for
MimicX, blue for Fixed Reference and teal for complementary mechanisms.

## Research content

- Navigation and public authors sit between the hero and showcase, centered;
  the anonymous version omits the identity block. The navigation includes
  Motions & scenes as well as direct anchors to the remaining sections.
- Showcase: eight naturally proportioned images, including the reviewed
  parkour, track, stairs, platform and forest evolution figures. An automatic
  4.8-second sequence uses a fade/slide transition and per-image progress.
  Named selectors, keyboard navigation and touch swipes support manual browsing.
  Hover, focus, an open dialog or an offscreen stage pauses progression;
  reduced-motion settings default to manual playback.
- Overview: paper-selected input, human reconstruction, simulated policy and
  reconstructed-scene frames, with Tennis and Forest example selection.
- Recordings: four keyboard-accessible task tabs, paired native video players,
  shared play/pause/restart and timeline. Readouts come from the generated
  static results table, not a second numerical dataset. This player is part
  of the showcase, followed by exactly four collision-scene recordings:
  Parkour, Stairs, Platform and Forest. Parkour is a diagnostic tracking
  rollout, not a claim of full-horizon completion. Other showcase items are stills.
- Method: the final hand-drawn supervision diagram and mechanism notes form
  a top-aligned 64:36 desktop row, stacked at full width on mobile.
- Results: four compact charts in one desktop row, plus a full-row dense
  training curve. Tablet uses two columns and mobile one. Compact exports
  preserve label readability; dialogs open the large vector originals.
  Every plot has native dark/light SVG exports, not color inversion.
- HLoop: the verification/scheduling artwork, workload, executor definitions,
  measured median timings and report-selection parity in balanced columns.

The initial video posters come from the same recordings at 0.1 seconds.
Videos use lazy preload; switching tasks replaces both sources and resets
the shared timeline. Loading failures have a retry message and individual
native controls remain available. Task tabs support arrow, Home and End keys.
Paired comparisons never autoplay. No videos play in the image carousel.
All research figures and scene videos open inside a dismissible page dialog,
with a fixed close control, Escape/backdrop dismissal and return of keyboard
focus. Video playback stops when its viewer closes. The paired player pauses
when scrolled out of view, when the document is hidden or a dialog opens.

## Evidence and assets

`assets/results/` remains the website-only numerical record. Do not move
results into the root code repository or change values as part of web design.
The table retains all four tasks. Videos illustrate individual recorded trials,
not seed-averaged outcomes. Protected outputs and horizon conventions are
described alongside the relevant evidence.

`assets/media/manifest.json` records the original experimental media.
`assets/media/research/manifest.json` records the new poster and final-artwork
derivatives, their source names and SHA-256 hashes. Paper drawings are exported
at 2400 pixels wide. Neither source PDFs nor videos are modified.
`assets/icons/lucide/` contains unmodified official Lucide 0.468.0 control
icons and their license. Existing resource-button brand icons are unchanged.

Original transparent workflow derivatives remain available in the asset archive.
The current page uses the four corresponding paper-selected panels instead of
the older six-thumbnail sequence. Source frames and transformations are recorded
in the showcase manifest. Court props provide presentation context; measurements
concern humanoid motion tracking.

## Maintenance

From the repository root:

```bash
python scripts/build_website_results.py
python -m pytest tests/test_website_release.py -q
python scripts/prepare_website_refresh.py --artwork /path/to/final_artwork
python scripts/verify_website.py --chromium /path/to/chromium --output runs/website-qa
```

The media exporter needs PyMuPDF, Pillow and CPU FFmpeg; `--ffmpeg` accepts a
binary path. The browser audit needs Playwright and Chromium, disables hardware
acceleration, uses a temporary local server and stops it after the audit.
It covers six viewport sizes, both themes, persistence, asset loading,
all task pairs, synchronized controls, table integrity and horizontal overflow.
Screenshots are review artifacts, not dependencies of the site.

Header and footer retain their approved design. Resource links are configured
per release; anonymous releases use local artifact documentation instead of
identifying model-hosting links. The arXiv button remains a placeholder until
an identifier is available. GitHub Pages uses main/docs;
a push alone is not proof that a hosted deployment has completed.

## Attribution

Preserved template structure: [RoboSplat](https://yangsizhe.github.io/robosplat/),
based on [NeRFies](https://nerfies.github.io/) and
[UMI on Legs](https://github.com/umi-on-legs/umi-on-legs.github.io/).
Adapted website template files use CC BY-SA 4.0. Bulma retains its MIT notice.
Lucide retains its ISC/MIT notices. Research videos, input frames, robot and
scene imagery retain their respective rights; template licensing does not
relicense these assets. Research code remains Apache-2.0.

## Current export and QA commands

```bash
python scripts/prepare_website_showcase.py --project /path/to/research-archive --ffmpeg /path/to/ffmpeg
python scripts/prepare_website_cases.py --project /path/to/research-archive --ffmpeg /path/to/ffmpeg
python scripts/prepare_website_gallery.py --project /path/to/research-archive --ffmpeg /path/to/ffmpeg
python scripts/build_website_evidence.py --paper /path/to/paper-source
python scripts/build_website_results.py
python -m pytest tests/test_website_release.py tests/test_website_showcase.py tests/test_website_evidence.py tests/test_website_cases.py -q
python scripts/verify_website_editorial.py --chromium /path/to/chromium --output /tmp/website-qa
```

`assets/media/showcase/manifest.json` records the approved source frames,
source checksums, crops inherited from the paper and CPU video derivatives.
`assets/media/evidence/manifest.json` maps each figure to its data and metric
contract. The paper's current direct comparison and five collision scenes
are separate cohorts; the older adapter comparison stays under `archive/`.

Before publishing, verify that all HTML/media dependencies are tracked in Git;
generic media ignore rules may otherwise omit newly exported files. The two
releases have independent commit histories. Do not merge a public release's
author history into an anonymous repository.
