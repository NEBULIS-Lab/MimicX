# Project Website

Open `index.html` directly, or visit https://nebulis-lab.com/MimicX.
The site is static HTML, CSS and JavaScript with local images and recordings.
No build step, external font, CDN, analytics or runtime data fetch is needed.

## September 28 research-body rebuild

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

- Overview: task-averaged tracking and horizon improvements, tennis completion,
  and the separately measured feedback speedup.
- Recordings: four keyboard-accessible task tabs, paired native video players,
  shared play/pause/restart and timeline. Readouts come from the generated
  static results table, not a second numerical dataset.
- Method: the final author-drawn supervision diagram, diagnosis/refinement/
  verification descriptions, and six real preparation/execution images.
- Results: four full-resolution linked vector plots in a desktop row, the
  complete four-task table, and downloadable numerical data and protocols.
- HLoop: the final verification/scheduling artwork and measured median timings.

The initial video posters come from the same recordings at 0.1 seconds.
Videos use lazy preload; switching tasks replaces both sources and resets
the shared timeline. Loading failures have a retry message and individual
native controls remain available. Task tabs support arrow, Home and End keys.
Playback is never started automatically.

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

Transparent workflow images preserve their alpha channels. Court props supply
presentation context; the measurements concern humanoid motion tracking.
Input/motion-preparation images illustrate the sequence; the two rendered
policy frames use the same recorded step 131.

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

Header and footer retain the released Code, Policies and Assets links. The
arXiv button remains the author's approved placeholder until an identifier
is available. The paper-source link is unchanged. GitHub Pages uses main/docs;
a push alone is not proof that a hosted deployment has completed.

## Attribution

Preserved template structure: [RoboSplat](https://yangsizhe.github.io/robosplat/),
based on [NeRFies](https://nerfies.github.io/) and
[UMI on Legs](https://github.com/umi-on-legs/umi-on-legs.github.io/).
Adapted website template files use CC BY-SA 4.0. Bulma retains its MIT notice.
Lucide retains its ISC/MIT notices. Research videos, input frames, robot and
scene imagery retain their respective rights; template licensing does not
relicense these assets. Research code remains Apache-2.0.
