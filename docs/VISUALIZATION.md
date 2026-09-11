# Visual Exports

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
