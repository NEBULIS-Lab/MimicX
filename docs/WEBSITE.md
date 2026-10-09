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

The approved hero artwork, full logo and compact footer are preserved. The
author band and narrow-screen layout are refined in `experience.css`; base
styles remain in `mimicx.css` and `brand.css`. The research body is owned by
`research.css`, scoped to `#research` and `.research-nav`.

Dark is the default for the rebuilt body. The navigation's theme control
switches to light and stores the choice under `mimicx-theme`. It deliberately
does not recolor the preserved hero, artwork or footer. Storage-denied browsers
still support switching for the current visit. Semantic colors are coral for
MimicX, blue for Fixed Reference and teal for complementary mechanisms.

## Research content

- Public authors sit inside the hero's lower translucent band, above the
  section navigation. The anonymous version omits the identity block. Navigation includes
  Motions & scenes as well as direct anchors to the remaining sections.
  All names use identical inline elements and aligned text baselines; the
  correspondence marker does not change their line height.
- Showcase: eight naturally proportioned images, including the reviewed
  parkour, track, stairs, platform and forest evolution figures. An automatic
  4-second sequence uses a centered looping album with visible previous/next
  images and per-image progress. Scale, opacity, angle and image positioning
  follow the actual scroll position, with smooth interpolation instead of
  separate selection-triggered transitions. Embla handles drag,
  loop continuity and resize. Named selectors, arrows and touch swipes support browsing.
  Focus, an open dialog or an offscreen stage pauses progression; hovering does not;
  reduced-motion settings default to manual playback.
  On narrow screens, the selected figure name scrolls into view without moving
  the page. The autoplay clock stops scheduling frames while paused, offscreen,
  backgrounded or blocked by a media dialog.
- Overview: paper-selected input, human reconstruction, simulated policy and
  reconstructed-scene frames, with Tennis and Forest example selection.
  Each selection pre-decodes all four frames and replaces them together with
  a short staggered fade. Rapid selections cannot mix stages across tasks.
- Recordings: four directly visible task rows. Each has five square players:
  original human input, Fixed Reference, BeyondMimic, SONIC and latest MimicX.
  There is no duplicate comparison section or task/baseline selector. On narrow
  screens the five-player row scrolls horizontally. Landscape videos are
  center-cropped in the page; the portrait Kung Fu input is bottom-aligned, so
  only its top is cropped. Enlarged task videos retain their original frames.
  Reference overlays remain available in the in-page media viewer. The
  BeyondMimic Dance/Kung Fu archives retain the original-reference label;
  they are not relabeled as matched repaired-reference comparisons. The paper's
  measurements and checkpoints are unchanged. Then follow four collision-scene recordings:
  Parkour, Stairs, Platform and Forest. Parkour is a diagnostic tracking
  rollout, not a claim of full-horizon completion. Other showcase items are stills.
  Scene previews and playback use square derivatives with the top black band
  removed; uncropped originals remain in the media archive. Crop rectangles,
  hashes and full-timeline checks are recorded in `recording-grid/manifest.json`.
- Method: the final hand-drawn supervision diagram and compact refinement
  ledger form a top-aligned 64:36 row, stacked on mobile. The ledger connects
  rollout localization, coordinated updates and repeated verification.
- Results: four compact charts in one desktop row, plus a full-row dense
  training curve. Tablet uses two columns and mobile one. Compact exports
  preserve label readability; dialogs open the large vector originals.
  Every plot has native dark/light SVG exports, not color inversion.
- HLoop: the verification/scheduling artwork, workload, executor definitions,
  measured median timings and report-selection parity in balanced columns.

Video posters come from their corresponding recordings. Videos preload only
when needed; visible rows play and loop at their native rates. Source lengths
differ, so these independent loops are not claimed as synchronized comparisons.
Per-row pause and native player controls remain available. Offscreen rows,
backgrounded documents and open dialogs suspend playback. Reduced-motion
preferences default to paused playback. No videos play in the image carousel.
All research figures and scene videos open inside a dismissible page dialog,
with a fixed close control, Escape/backdrop dismissal and return of keyboard
focus. Dialog dimensions follow the actual media aspect ratio, constrained by
the available viewport and measured toolbar height. Panoramas, square charts
and portrait frames therefore do not share a fixed-height canvas. Images
support Panzoom pinch, pan, zoom buttons and fit-to-view. The
numerical table has the same zoomable viewer, keeping its original cells and
values. The hero no longer includes a scene label, rollout shortcut or expand
button; its author band remains in the public edition. Native page pinch zoom is not
disabled. Video playback stops when its viewer closes. Eligible visible rows
resume, except those the reader explicitly paused.

The four primary MimicX players show the reference-overlay (`*-ghost.mp4`)
recordings, including matching posters and enlarged playback. Each task's
`Robot only` link retains the corresponding overlay-free recording.

The `assets/media/recovery/manifest.json` inventory records full-decode checks,
source hashes, transformations and the additional-continuation cohort. It also
includes the high-resolution source/human/G1-reference composites used by the
submission media package. Reference-stage media and learned-policy recordings
retain distinct labels. `scripts/verify_website_recordings.py` checks desktop
and portrait layouts, autoplay, native controls and modal playback.

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
python scripts/verify_website_details.py --site docs --chromium /path/to/chromium --output runs/website-details
python scripts/verify_website_interactions.py --site docs --chromium /path/to/chromium --output runs/website-interactions
```

The media exporter needs PyMuPDF, Pillow and CPU FFmpeg; `--ffmpeg` accepts a
binary path. The browser audit needs Playwright and Chromium, disables hardware
acceleration, uses a temporary local server and stops it after the audit.
It covers six viewport sizes, both themes, persistence, asset loading,
all task pairs, synchronized controls, table integrity and horizontal overflow.
Screenshots are review artifacts, not dependencies of the site.
The interaction audit also saves browser recordings. Media dialogs have short
opening/closing transitions, preload their control icons, retain native media
proportions and restore focus after closing. All new motion respects reduced-motion
preferences. Navigation includes the nested rollout section and keeps the active
link visible on phones. Dataset values and research media are not altered.

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
python scripts/verify_website_mobile.py --site docs --chromium /path/to/chromium --output /tmp/mobile-qa
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
