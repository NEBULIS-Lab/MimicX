# Numerical Evidence and Evaluation

This website's `assets/results/` stores the current numerical evidence snapshot. These are
previously completed research experiments, not new measurements from the
source-release CPU tests. Positive, neutral and negative rows are retained.

## Core Comparison

Four tasks (Tennis Swing, Football Juggling, Dance Sequence, Kung Fu Sequence),
four methods, three continuation seeds and three evaluation seeds give 48
task/method/continuation-seed trials and 144 final evaluation rollouts.
Continuation seeds share a task-specific warmstart. Protected policies in
Dance and Kung Fu retain that warmstart rather than representing three
independently trained final networks.

| Internal data ID | Public method name | Intervention |
|---|---|---|
| `m0_open_loop` | Fixed Reference | Fixed-reference continuation |
| `m1_policy_window` | Failure Curriculum | Failure-window sampling |
| `m2_task_hierarchy` | Task-Aware Refinement | Task-aware objective and curriculum |
| `m3_full_mimicx` | MimicX | Candidate search with repeated execution verification |

Do not use the internal IDs as paper or figure labels. The original IDs remain
in CSVs to preserve traceability to the experiment records.

| Task | Fixed Reference first-failure horizon | MimicX first-failure horizon |
|---|---:|---:|
| Tennis Swing | 322.0 | 801.0 (full budget completed) |
| Football Juggling | 53.7 | 427.3 |
| Dance Sequence | 193.3 | 341.3 |

Tennis strict success improves from 11.1% to 100%; aligned worst-body temporal
mean error decreases from approximately 0.241 m to 0.157 m. Football's horizon
improves substantially but its strict full-budget success remains 0% in this
cohort. Component methods can outperform the protected output on particular
tasks; all four method rows are published in the CSV, not only the selected
examples above.

Input preparation is part of the evaluated pipeline. Methods using a repaired
reference are evaluated against that registered reference; these results do
not substitute for a shared-original-reference fidelity evaluation.

## Metric Definitions

- Strict success: no strict termination throughout the fixed rollout budget.
- First-failure horizon: earliest failure across the repeated rollouts; no
  failure is encoded as `H + 1`, not an extra executed step.
- Body error: aligned positional maximum across tracked bodies at each step,
  then temporal/repeat aggregation. It is not mean per-joint position error.
- Anchor error: world-coordinate torso-anchor positional error.
- EE height: active termination-test end effectors, the two ankles for the
  main strict task. It is not a generic hand-accuracy metric.
- Reward: dimensionless; objective changes mean it is not the primary
  cross-method acceptance criterion.
- `reward_std` in the exported method summary is the original population SD;
  manuscript sample-SD tables must recompute `ddof=1` from the 48 trial rows.

## Verification and Broader Evidence

`accept_protect_decisions.csv` contains all twelve decisions: six accepted,
six protected. `reward_higher_regression_rejection.csv` preserves a separate
historical Kung Fu example of rejecting higher-reward execution regressions.
The four additional-video rows and fourteen motion-input rows extend task
coverage. Motion datasets test reference-to-policy behavior, not monocular
video reconstruction. See the cohort labels in the files.

`sonic_comparison.csv` compares a released SONIC policy with task-specific
MimicX outputs using the recorded evaluation adapter. It is a released-policy
transfer comparison, not equal-budget retraining of both methods.

## HLoop

| Executor | Median seconds | Speed relative to sequential |
|---|---:|---:|
| Sequential | 241.221 | 1.000x |
| Bulk-synchronous | 122.258 | 1.973x |
| MimicX-HLoop | 124.017 | 1.945x |

Five repeats per executor used the same fixed-policy workload (eight rollouts,
eight diagnoses and one report selector), with zero PPO updates. Recorded
report selections agree. This timing evidence measures execution efficiency;
it does not imply improved policy quality or faster convergence per PPO step.
