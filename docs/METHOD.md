# MimicX Method

## Policy-in-the-Loop Supervision

MimicX combines video reconstruction, humanoid retargeting, PPO tracking and
execution-conditioned supervision refinement. GVHMR, SMPL-X, GMR and PPO are
external building blocks. MimicX contributes their policy-feedback loop,
task-aware candidate construction and incumbent-protected verification.

An authorized video supplies reconstructed and retargeted motion; an existing
robot motion enters at the same reference interface after input validation.
MjLab and native Isaac provide simulator-specific training and execution.
Both use the same diagnosis, bounded-supervision and repeated-verification
design. Named joints/bodies, reference timing and backend configuration travel
with every run rather than being inferred from a checkpoint filename.

Reference preparation establishes the robot motion used by the task. Within a
released AutoRefine run, incumbent and candidates share the registered motion.
The candidate generator changes objective and curriculum parameters; it does
not rewrite joint-coordinate trajectories during the search.

## Single-Task Loop Profile

The [quickstart](QUICKSTART.md#autorefine) creates a manifest for an individual
task. The following diagnosis and verification settings describe that manifest
profile. The matched-budget comparison uses the joint local/global profile
specified below; each run records its chosen proposal and gate rules.

### Diagnose

`scripts/autorefine/mine_failures.py` consumes aggregate JSON and step CSV.
The current coordinator diagnoses the first scheduled verification rollout.
It locates the first termination, or the largest body-error step when no
termination occurs; the local replay window spans 40 steps on each side,
clipped to the recorded horizon. Error channels identify affected body parts,
root/anchor motion and dynamics. See `mimicx/refinement/proposals.py` for the
bounded mappings from diagnostics to typed patches.

### Propose and Continue

The default candidate family combines local/global objectives with window
replay, stronger replay with dynamics emphasis, and full-start consolidation.
Each candidate continues from the same incumbent with the configured seed,
learning rate, environment count and iteration budget. Reward terms include
body/anchor tracking, selected end effectors, linear/angular velocity,
contact-sensitive foot behavior and action regularization. The exact task
configurations are in
`dependencies/backend_overlay/unitree_rl_mjlab/src/tasks/tracking/`.

The backend uses RSL-RL PPO, not a separately implemented RL optimizer.
`--agent.max-iterations` counts PPO learning iterations; each collects a
rollout batch before minibatch updates. A continuation budget is additional
training from a warmstart, not its lifetime total.

The optional `llm`/`hybrid` proposal adapters accept externally generated JSON
through the same typed validation. They are not required by the default loop
or the released controlled comparison.

### Verify and Protect

Three repeated strict rollouts determine an ordered execution key:

1. More zero-termination rollouts.
2. Later worst first-failure step across repeats.
3. Fewer total termination events.
4. Lower configured guard metrics, in manifest order.
5. Higher mean rollout reward as the last tie-breaker.

The generated manifest uses aligned worst-body error and ankle-height error
as ranking metrics. The body guard permits at most a 20% relative increase
from the current policy. Guard scalars use the last recorded sample; the
separate reported tracking mean aggregates all rollout samples. These are
different reductions of the same telemetry, implemented in `gate.py` and
`evaluation/paper_matrix.py` respectively.

A candidate must improve the ordered execution key and pass all configured
guards. Missing verification evidence rejects it. Otherwise the current
checkpoint remains selected. Saved progress supports restarting the loop without
repeating completed training.

## Matched-Budget Motion Profile

The [motion comparison](CONFIRMATION.md) uses the same supervision primitives
with an adaptive proposal rule and repeated local/global verification:

1. **Incomplete execution:** diagnose the protected incumbent's failure window
   and body channels, then construct an untried bounded supervision patch.
2. **Complete execution:** try the predefined precision patches. If all three
   first-stage selection slots complete but are rejected for a local/global
   tradeoff, inspect the matched-seed evidence. Native-objective full-motion
   replay is used in stage two only when world mean/tail errors do not worsen,
   local mean error worsens for at least one repeat at each slot, and the
   protected incumbent remains unchanged. No final-evaluation data enters this
   decision.
3. **Continue:** train two 1,500-update stages from the protected incumbent,
   selecting among three checkpoints per stage. Accepted intermediate policies
   can become the next stage's parent; rejection retains the incumbent.
4. **Verify:** preserve the physical execution frontier and require nonincreasing
   world-body and root-local FK mean/P95 errors for every matched selection seed
   over the same valid, pre-reset interval. Acceptance additionally requires a
   longer frontier or more than a 2% reduction in the sum of those mean errors,
   subject to native safety checks. A donor guard prevents cumulative drift.

BeyondMimic/MjLab continues the native tracker; Static Search trains two
predetermined branches without feedback-conditioned proposals. All three receive
the same shared 3,000-update warmstart, 3,000 continuation updates, six checkpoint
selection slots and repeated-selection seeds. Final nominal and perturbed
evaluations occur only after selection. Released SONIC is an execution baseline
with no per-task continuation in this protocol, not an equal-training-cost arm.

The configuration identifier `joint_adaptive_precision_confirmation_v1` preserves
the exact implemented rule. Stable identifiers in saved runs are compatibility
keys, not additional named methods. Full execution, native precision qualification
and common-clock tracking error are reported separately; a complete recording is
not itself evidence of a physically completed motion.

## HLoop

HLoop schedules dependency-ready CPU, GPU and I/O jobs while preserving their
commands, inputs, seeds, budgets and dependency edges. Sequential and
bulk-synchronous modes provide execution baselines. The reported timing
workload used fixed policies and zero PPO updates; its selection comparison
concerns report selection, not a newly retrained policy.

## Source Map

| Operation | Source |
|---|---|
| Experiment configuration | `mimicx/refinement/manifest.py` |
| Failure diagnosis | `scripts/autorefine/mine_failures.py` |
| Typed proposals | `mimicx/refinement/proposals.py` |
| Continuation and restart | `mimicx/refinement/closed_loop.py` |
| Repeated gate | `mimicx/refinement/gate.py` |
| Matched-budget proposals and joint gate | `mimicx/experiments/motion_study.py` |
| Comparison protocol | `mimicx/experiments/motion_study_confirmation.py` |
| Native Isaac training and rollout | `scripts/isaac/` and `mimicx/backends/isaaclab/` |
| Atomic persistence | `mimicx/refinement/state.py` |
| PPO integration | `dependencies/backend_overlay/unitree_rl_mjlab/scripts/train.py` |
| Task objectives | `dependencies/backend_overlay/unitree_rl_mjlab/src/tasks/tracking/mdp/rewards.py` |
| Scheduler and parity | `mimicx/runtime/hloop.py` |
