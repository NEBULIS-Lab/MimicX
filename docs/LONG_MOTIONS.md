# Continuous Motion and Multi-Phase Evaluation

MimicX's reference interface accepts continuous robot motions as well as motions
reconstructed from video. The supervision loop keeps the registered trajectory
fixed while adapting objectives, practice and policy continuation. Long-motion
evaluation uses this same loop, not concatenated short-policy demonstrations.

## Audit the Actual Horizon

The CPU-only audit distinguishes source length, the scored source window and
continuous execution coverage. Video duration alone is not a completion metric.

```bash
python scripts/experiments/audit_motion_horizon.py /path/to/reference.npz \
  --minimum-duration-s 60 --output /path/to/new-reference-audit.json
```

The reference requires explicit `fps` or `frame_rate` and supported position
arrays. Duration is `(frames - 1) / fps`: a 60-second reference at 50 Hz has
3,001 samples. Reference-only inspection reports input coverage, not policy
success. Existing short-motion results do not become long-motion results when
looped or replayed more slowly.

With explicit rollout metadata, audit continuity and optionally declared stages:

```bash
python scripts/experiments/audit_motion_horizon.py /path/to/reference.npz \
  --rollout /path/to/rollout-metadata.npz --timestamp-kind simulation \
  --minimum-duration-s 60 --stages-json /path/to/stages.json \
  --require-verified --output /path/to/new-execution-audit.json
```

Rollout metadata records synchronized timestamps, zero-based reference frame
IDs, and explicit reset/failure evidence. Declare whether timestamps are measured
wall time or simulator time. The tool does not infer missing frame IDs from
poses, synthesize timestamps, interpolate gaps or join episodes. Missing evidence
remains unknown. A negative or unknown audit produces a nonzero exit code when
`--require-verified` is supplied. This is a timing/coverage audit, not a replacement
for [physical execution and fidelity scoring](SONIC_STUDY.md).

An optional stage file contains named, contiguous source intervals with shared
inclusive endpoints, for example:

```json
[
  {"name": "first interval", "start_frame": 0, "end_frame_inclusive": 1500},
  {"name": "second interval", "start_frame": 1500, "end_frame_inclusive": 3000}
]
```

These labels are supplied by the experimenter; fixed-duration bins are not
automatically semantic skills. Report actual transitions, local/global error
channels and stage-level coverage together with full-motion completion.

## Matched Continuation

Use the [motion study executor](MOTION_STUDY_EXECUTION.md) with a new input
manifest, runtime and output directory. The generic development profile
`joint_adaptive_precision_v6` supports separately declared task sets; the
[12-motion confirmation freeze](CONFIRMATION.md) remains unchanged and cannot
be repurposed by changing its task count or seeds.

For a controlled long-motion study, freeze before execution:

- A continuous source interval, canonical joint/body names, units and source
  timing, with no target repair or outcome-dependent interval selection.
- The same shared warmstart and total continuation budget per native method.
- The same checkpoint-selection opportunities and selection seeds, with
  independent final evaluation seeds.
- No reference wrap or physical reset counted as continuous completion.
- Completion, first failure, world and root-local fidelity, stage coverage and
  all candidate training costs, including discarded candidates.

BeyondMimic/MjLab and Static Search supply native training controls. Released
SONIC is an execution baseline, not an equal-pretraining-cost control. Reference
length, evaluation protocol, sampling units and learning costs travel with each
result. Reusing the refinement algorithm across individually trained motions is
different from a single policy tracking unseen motions without adaptation.

## Framework Roles

Motion reconstruction and retargeting prepare the reference. PPO is the default
learner; MuJoCo/MjLab and native Isaac Lab provide simulation. HLoop schedules
feedback work. The [method guide](METHOD.md) explains how these roles compose;
support for another simulator is not itself evidence of cross-robot transfer or
long-motion performance. Policy and source-result releases remain separate from
the availability of these evaluation tools.
