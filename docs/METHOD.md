# Implemented Method

## Policy-in-the-Loop Supervision

MimicX combines video reconstruction, humanoid retargeting, PPO tracking and
execution-conditioned supervision refinement. GVHMR, SMPL-X, GMR and PPO are
external building blocks. MimicX contributes their policy-feedback loop,
task-aware candidate construction and incumbent-protected verification.

Reference preparation establishes the robot motion used by the task. Within a
released AutoRefine run, incumbent and candidates share the registered motion.
The candidate generator changes objective and curriculum parameters; it does
not rewrite joint-coordinate trajectories during the search.

## Diagnose

`scripts/autorefine/mine_failures.py` consumes aggregate JSON and step CSV.
The current coordinator diagnoses the first scheduled verification rollout.
It locates the first termination, or the largest body-error step when no
termination occurs; the local replay window spans 40 steps on each side,
clipped to the recorded horizon. Error channels identify affected body parts,
root/anchor motion and dynamics. See `mimicx/refinement/proposals.py` for the
bounded mappings from diagnostics to typed patches.

## Propose and Continue

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

## Verify and Protect

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
checkpoint remains selected. Atomic state writes, content hashes and completed
command records support restart without repeating successful training.

## HLoop

HLoop schedules dependency-ready CPU, GPU and I/O jobs while preserving their
commands, inputs, seeds, budgets and dependency edges. Sequential and
bulk-synchronous modes provide execution baselines. The reported timing
workload used fixed policies and zero PPO updates; its selection comparison
concerns report selection, not a newly retrained policy.

## Source Map

| Operation | Source |
|---|---|
| Configuration and hashes | `mimicx/refinement/manifest.py` |
| Failure diagnosis | `scripts/autorefine/mine_failures.py` |
| Typed proposals | `mimicx/refinement/proposals.py` |
| Continuation and restart | `mimicx/refinement/closed_loop.py` |
| Repeated gate | `mimicx/refinement/gate.py` |
| Atomic persistence | `mimicx/refinement/state.py` |
| PPO integration | `dependencies/backend_overlay/unitree_rl_mjlab/scripts/train.py` |
| Task objectives | `dependencies/backend_overlay/unitree_rl_mjlab/src/tasks/tracking/mdp/rewards.py` |
| Scheduler and parity | `mimicx/runtime/hloop.py` |
