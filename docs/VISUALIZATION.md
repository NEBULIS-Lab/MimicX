# Visual Exports

For source-matched cameras, baseline recording contracts, and dynamically
simulated sports-object contact, see [Recording and Contact](RECORDING_AND_CONTACT.md).
The original paper rollout commands below remain unchanged.

Use the tracking environment and path setup from [Quickstart](QUICKSTART.md).
Render a real checkpoint under the same evaluation task, motion and seed used
for its numerical evaluation:

```bash
python scripts/rollout_mjlab_motion_smoke.py \
  --task Unitree-G1-Tracking-MimicX-Curriculum \
  --motion-file "$MOTION_FILE" --checkpoint-file "$CHECKPOINT" \
  --num-envs 1 --steps "$HORIZON" --seed 1001 --disable-joint-init-noise \
  --video-output outputs/skill/ours_overlay.mp4 \
  --video-width 1920 --video-height 1080 --video-fps 50 \
  --ghost-color 0.831,0.357,0.298,0.30 \
  --screenshot-dir outputs/skill/screenshots --screenshot-every 100 \
  --metrics-output outputs/skill/metrics.json \
  --step-metrics-output outputs/skill/metrics_steps.csv \
  --replay-output outputs/skill/replay
```

Despite its historical filename, this exporter accepts the entire task horizon
and writes full rollout telemetry. With no checkpoint it uses zero actions;
that diagnostic is not a learned-policy result. Leave terminations enabled for
scientific comparison. `--reference-joint-policy` is a separate diagnostic.

Run again with `--hide-reference-visual` and another output path for a clean
robot-only video. Run the baseline checkpoint with identical framing, horizon,
noise setting and seed. Use distinct reference colors: Fixed Reference gray
`0.45,0.48,0.52,0.30`, Failure Curriculum blue `0.30,0.47,0.66,0.30`,
Task-Aware Refinement teal `0.20,0.60,0.56,0.30`, MimicX coral as above.

For the intermediate motion stage:

```bash
python scripts/export_mjlab_ghost_only_video.py \
  --task Unitree-G1-Tracking-MimicX-Curriculum \
  --motion-file "$MOTION_FILE" --video-output outputs/skill/reference_only.mp4 \
  --video-width 1920 --video-height 1080
```

Ghost ground alignment is display-only: it does not change the registered
reference, observations, rewards or policy checkpoint. Preserve exact replay
arrays for offline rendering and preserve the original input video separately.
The replay manifest records joints, body names, frame timing and provenance.

The reusable visualization modules support MJCF geometry, source-video object
registration, motion-phase matching and grasp meshes. Calibrated object props
are presentation geometry, not evidence of trained ball or racket contact.
Full paper scene compositions require separately obtained assets and camera
calibrations; those rendered media are not bundled in this source release.

## Complete-Clock Comparisons

The saved-state exporter in `scripts/visualization/source_camera.py` now checks
every decoded frame's zero-based timestamp and duration, not only frame count
and average FPS. It rejects missing tails, duplicate timestamps, gaps and clock
offsets before publishing the video. `mimicx.visualization.video_audit.audit_video`
also exposes this CPU-only check for existing recordings:

```python
from mimicx.visualization.video_audit import audit_video

audit = audit_video("inputs/replay.mp4", frames=301, fps=30, size=(1280, 720))
assert audit["full_clock_passed"]
```

Frame rates retain their supplied meaning: `29.97` and `59.94` are literal
decimal rates, distinct from `30000/1001` and `60000/1001`. The source exporter
passes an explicit rational input rate to FFmpeg to avoid ImageIO's two-decimal
rounding. For NTSC clocks, supply the corresponding floating-point quotient
(approximately `29.97002997002997` or `59.94005994005994`); the rational conversion
uses a maximum denominator of 1,000,000. Canonical 50 Hz remains exactly 50 Hz.
The clock audit tolerance is unchanged and rejects substituting a decimal rate
for its NTSC counterpart once accumulated drift exceeds that tolerance.

For fixed-camera comparisons, use the existing source-camera render command with
`--require-constant-camera`. This checks every supplied camera matrix and records
a numeric camera digest. `--display-contrast` additionally opts into neutral
lighting/background and a non-colliding display floor at z=0. Both flags are
optional; existing moving-camera exports and original scenes remain supported.
See [Recording and Contact](RECORDING_AND_CONTACT.md#comparison-display-contract)
for scene restrictions and the distinction between display geometry and physics.

## Synchronous Review Montage

`scripts/visualization/comparison.py` builds exactly one 2x2 comparison from four
existing clips. It performs CPU full-decode clock checks on all inputs and the
output, verifies identical fixed cameras, uses uniform scaling only, and adds
24px method/outcome headers **outside** the images. There is no cropping, looping,
retiming, endpoint padding, task selection, trajectory editing or new evaluation.
Keep and distribute the original full-size clips alongside the montage.

The configuration uses paths relative to its own directory. Example
`inputs/comparison.json`:

```json
{
  "frames": 301,
  "fps": 30,
  "source_size": [1280, 720],
  "tile_size": [640, 360],
  "clips": [
    {"video": "method-a.mp4", "camera": "camera.json", "label": "Method A", "physical_complete": true},
    {"video": "method-b.mp4", "camera": "camera.json", "label": "Method B", "physical_complete": false},
    {"video": "method-c.mp4", "camera": "camera.json", "label": "Method C", "physical_complete": true},
    {"video": "method-d.mp4", "camera": "camera.json", "label": "Method D", "physical_complete": false}
  ]
}
```

```bash
python scripts/visualization/comparison.py \
  --config inputs/comparison.json --output outputs/review/comparison.mp4
```

Dependencies are the visualization extra and a discoverable `DejaVuSans.ttf`, or
an explicit TrueType font via `--font`. Long headers fail instead of overflowing;
shorten the label or increase the tile size while retaining the source aspect
ratio. The Python API is `mimicx.visualization.comparison.export_comparison`.

The sibling JSON receipt records source/output hashes, camera digests, frame-clock
audits, tile positions and the supplied physical outcomes, without absolute input
paths. `motion complete` is not strict tracking qualification. Camera JSON and
outcome labels must come from the original recording evidence; matching supplied
JSON cannot itself establish a video's camera provenance. The exporter preserves
negative outcomes and declares visual review pending, not publication approval.
Existing MP4s and receipts are refused; video publication is atomic and never
replaces a competing export. Video and receipt are separate files, not a
transactional pair: retain/check the receipt before distributing the output.

## Integration Boundary

- Already present and reused: replay interpolation/SLERP, source-camera
  projection and stabilization, named joints, robot-pixel protection with a
  silhouette guard, translucent ghosts, full decode and no-overwrite publication.
- Ported from the reviewed display/QA workflow: per-frame clock validation,
  fixed/shared camera checks, neutral contact-free display compilation/FK checks,
  and a bounded synchronous four-clip montage. Separate reference-clock endpoint
  holds are now reported alongside actual-robot holds.
- Intentionally local-only: delivery inventories/bindings, selected tasks and
  method palettes, batch orchestration, GPU launchers, host paths, original
  media, contact sheets and publication/review decisions. None are bundled or
  regenerated by this integration.

CPU regression coverage (synthetic media, no graphics context):

```bash
python -m pytest tests/test_visual_export_contracts.py tests/test_visual_comparison.py \
  tests/test_recording_release.py tests/visualization -q
```

These tests verify clocks, camera invariants, compiled display geometry, source
preservation and montage tile pixels. They do not establish visual quality of
new robot renders or constitute a new experiment/result export.
