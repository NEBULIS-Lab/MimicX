# Motion Study Execution

## Supported Scope

This is an executable study coordinator, not a completed study. `prepare` freezes
the task matrix and code/input identities without launching anything. `run` and
`resume` execute MjLab scratch warmstart and all three continuation arms.
Requesting a continuation first runs or verifies its shared warmstart dependency.
All three arms reference that same task/backend/training-seed warmstart.

BeyondMimic trains continuously; Static Search trains two independently frozen
branches; MimicX diagnoses the protected incumbent before each of two stages.
All arms validate six checkpoints, protect/select with the existing repeated
gate, and evaluate the selected checkpoint on untouched nominal and perturbed
holdout seeds. Isaac training and SONIC scoring are separate integrations.
`completed` means that one arm's protocol finished, not that its policy qualified
or that the full matrix ran. No completed experiment is implied by preparing jobs.

## Local Inputs

Keep runtime JSON, generated campaigns and licensed motions outside the source
repository. No data is bundled or downloaded. Manifest paths resolve relative to
the manifest; runtime paths resolve relative to the runtime JSON. Prepare at the
destination after relocating a code/data snapshot, not by rewriting an existing
campaign's absolute paths.

The dataset interface is JSON `{"tasks": [...]}`. Every task has `task_id`,
`split` (`development` or `confirmation`), `source`, `source_path`, `motion_path`,
`frames`, `fps`, `duration`, `sha256`, and `quality_status` (or `quality.status`).
Only passed/pass/valid/approved inputs may train. Blocked tasks remain visible;
their missing motion/hash may be null. IDs must be filesystem-safe and unique.
The current runtime requires 50 Hz. Elapsed duration `(frames-1)/fps` is accepted.
The interpreter path is made absolute without following symlinks: a venv's
`bin/python` must not be replaced by its base interpreter. Backend/input paths
are resolved normally. Dataset `protocol` and `protocol_sha256`, when present,
are bound exactly into the campaign. Ground-contact tasks additionally require
their task-level protocol ID/hash to match before they may run.
The dataset agent owns provenance, 29-DoF/body-order, finite-value, timing, FK and
physical-reference QA; a file hash is not a substitute for those checks.

Example private runtime JSON (replace placeholder paths):

```json
{
  "backend_root": "/path/to/clean-pinned-backend",
  "python": "/path/to/tracking-environment/bin/python",
  "device": "cuda:0",
  "num_envs": 4096,
  "steps_per_env": 24,
  "backends": ["mjlab"],
  "seeds": {"development": [101, 202], "confirmation": [101, 202, 303]}
}
```

The backend must be fully installed and overlaid, not the overlay-only source
directory. Its actual `scripts/train.py` must expose `TrainConfig.from_task` and
the `Unitree-G1-Tracking` registration. Check in that environment, without GPU
visibility, before preparing:

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH="$BACKEND:$BACKEND/src" \
  "$TRACKING_PYTHON" -c \
  'import runpy,sys; import src.tasks.tracking.config.g1; script=sys.argv.pop(1); runpy.run_path(script,run_name="__main__")' \
  "$BACKEND/scripts/train.py" Unitree-G1-Tracking --help
```

The minimal worker explicitly imports `src.tasks.tracking.config.g1`, then imports
the real train module and its default factory. A backend with an empty
`src/tasks/__init__.py` needs the same explicit registration before its standalone
CLI can recognize the task. It
preserves native rewards, curriculum, adaptive PPO schedule, entropy, optimizer
and normalization settings. It changes only the motion file, matched environment
count, shared seed, offline logging and save cadence. It uses the same MjLab PPO
runner as the recovery worker; backend-specific ONNX exports are omitted.
Before every environment construction, scratch and continuation training install
the same initial-reference cache refresh used by evaluation. The patched class
is used directly and the original class is restored afterward; no simulation
step, reference retiming, reward change or termination relaxation is introduced.
No recovery route/objective, reference retiming or repair is applied. Static
Search and MimicX apply only validated AutoRefine patches to this same native
configuration using the backend's `_apply_mimicx_autorefine_patch` transformer.
They do not inherit hidden dynamic-task/MimicX base coefficients. Both retain
the same native PPO schedule and common termination contract as BeyondMimic.

## Commands

```bash
python scripts/experiments/motion_study.py prepare \
  --manifest /path/to/data/manifest.json --runtime /path/to/runtime.json \
  --output /path/to/campaign
python scripts/experiments/motion_study.py jobs --campaign /path/to/campaign/campaign.json
python scripts/experiments/motion_study.py run \
  --campaign /path/to/campaign/campaign.json --task TASK_ID --seed 101 \
  --backend mjlab --method warmstart
python scripts/experiments/motion_study.py resume \
  --campaign /path/to/campaign/campaign.json --task TASK_ID --seed 101 \
  --backend mjlab --method beyond_mimic
python scripts/experiments/motion_study.py run \
  --campaign /path/to/campaign/campaign.json --task TASK_ID --seed 101 \
  --backend mjlab --method static_search
python scripts/experiments/motion_study.py run \
  --campaign /path/to/campaign/campaign.json --task TASK_ID --seed 101 \
  --backend mjlab --method mimicx
python scripts/experiments/motion_study.py validate-checkpoint \
  --campaign /path/to/campaign/campaign.json --task TASK_ID --seed 101 \
  --backend mjlab --method warmstart \
  --checkpoint /path/to/warmstart/model_499.pt \
  --output /path/to/campaign/diagnostics/warm500
```

`run` and `resume` have identical dependency/resume semantics. Help, prepare and
jobs are CPU-only and do not import Torch or simulators. A caller may select
`device: cpu` for an actual CPU runtime; CPU tests use an explicitly fake runner.
Resource acquisition, launch environment and process placement belong to the
parent launcher, not this coordinator. Ambient patch/device overrides are
rejected. The inherited visibility mask is never rewritten.
Standard `cpu`, `cuda`, and `cuda:N` devices are supported. There is no public
requirement for a specific launcher, visibility-mask format or resource manager.
Workers default `GIT_PYTHON_REFRESH=quiet` when it is unset, allowing GitPython
imports without a Git executable. Set it before importing `native_configs` in
standalone profiling tools as well; no fake Git executable is created.
`validate-checkpoint` runs the frozen repeated selection seeds for a development
warmstart checkpoint, including intermediate checkpoints. It writes diagnostic
receipts/costs separately from primary job records and never marks a warmstart
complete or selects a policy. It rejects confirmation tasks and outputs inside
the primary `jobs/` tree. The first 500-update checkpoint is `model_499.pt`.

An older isolated warmstart may be reused by a newer frozen coordinator:

```bash
python scripts/experiments/motion_study.py adopt-warmstart \
  --campaign /path/to/new-campaign/campaign.json --task TASK_ID --seed 101 \
  --backend mjlab --method warmstart \
  --from-job /path/to/old-campaign/jobs/TASK_ID/mjlab/seed101/warmstart
```

Adoption verifies donor artifact hashes, scratch iteration 0 through 2999,
task/seed/reference/timing, environment/rollout budget, native backend source
identity and the current native PPO configuration. It refuses a used destination.
New donor contracts explicitly record initial-reference refresh. Earlier aligned
pilot donors need their unchanged frozen training source to verify both refresh
installation and use of the aligned environment class; pre-fix donors are refused.
`adoption.json` preserves the donor receipt and cost origin; reused training is
not zero-cost training. Do not edit a running snapshot to add continuation code.

### Expanded Legacy Asset Coverage

Older donors may record only the canonical XML, while a newer campaign also
records its complete asset closure. This is not silently treated as a backend
change or silently ignored. All previously recorded backend paths/hashes must
match exactly; deletion, code changes and new non-asset records still fail closed.
For added canonical asset records, create an independently audited proof on the
original Git checkout using the approved full commit ID:

```bash
python scripts/experiments/motion_study.py prove-backend-assets \
  --backend-root /path/to/audited-backend \
  --model-xml /path/to/audited-backend/src/assets/robots/unitree_g1/xmls/g1.xml \
  --git-commit APPROVED_FULL_COMMIT --output /path/to/backend-assets-proof.json
```

This reads immutable Git blobs and requires every current asset to match their
bytes. Pin the resulting file path, SHA-256 and Git commit in the NEW runtime JSON
as `legacy_backend_asset_proof`, `legacy_backend_asset_proof_sha256` and
`legacy_backend_asset_commit`. Keep the proof outside `backend_root`; the new
campaign records its hash. Transfer the proof unchanged. Consumers need no Git or
network, and verify both donor/current asset bytes against that pinned proof.
The proof is trusted only through the operator-approved digest/commit and audited
transfer provenance; it does not manufacture historical donor hashes.

Before waiting for checkpoint completion or launching a continuation queue, run:

```bash
python scripts/experiments/motion_study.py check-adoption \
  --campaign /path/to/new-campaign/campaign.json \
  --donor-campaign /path/to/old-campaign/campaign.json \
  --output /path/to/backend-adoption-preflight.json
```

This requires no checkpoint and performs no training. Normal `adopt-warmstart`
then repeats the backend guard plus all existing weight/budget/config checks.
`adoption.json.backend_identity_audit` distinguishes matching old records from
newly audited asset coverage, pins the proof and commit, and explicitly states
that the legacy worker did not record those additional hashes. Current rsync or
Git-clean checks are current consistency evidence, not pre-run hash records.

## Budgets And Resume

One task/backend/seed gets 3,000 scratch updates shared by all arms. Each arm
gets 3,000 more: BeyondMimic one continuous trajectory; Static Search two
independent predetermined 1,500-update branches from warmstart; MimicX two
feedback-guided 1,500-update stages from the protected incumbent. Every attempted
candidate, rejected or accepted, must consume that arm's budget. No early-stop
qualification can create unequal budgets.

BeyondMimic starts at iteration zero, ends warmstart at 2999, then continues from
3000 through 5999. It saves selection slots at 3499/3999/4499/4999/5499/5999.
Other arms declare three slots per stage/branch at 500/1000/1500 updates. The
unchanged warmstart remains available to all selection rules. Each stage trains
one full 1,500-update candidate, including when it is subsequently rejected.
Every slot uses identical selection seeds and the versioned
`motion_study_gate.accept` criterion. Static patches are generated and
frozen during `prepare`, before any rollout outcomes; MimicX uses the existing
failure miner/body-channel diagnosis and bounded proposal generator. After each
stage, the next MimicX stage starts from the protected best checkpoint, which may
be an intermediate checkpoint or the unchanged warmstart. No qualification gate
stops training early.
MimicX records prior candidate configurations and chooses the first structurally
untried proposal from the existing bounded generator. IDs alone cannot make an
identical configuration new; rejecting stage zero cannot repeat its identical
patch/parent/seed in stage one.

## Validation And Selection

All methods use one native play registration, phase-zero initialization, the
original motion, enabled physical terminations, and exactly `frames-1` steps at
50 Hz. This reaches the final reference frame without wrapping. The worker uses
the existing rollout CLI and exports full step metrics and exact replay states.
It verifies the replay clock, 29-joint contract, finite arrays and artifact hashes.
Warmstart validation is shared once per task/backend/training seed, not counted
as an extra checkpoint-selection slot for any arm.

Default selection seeds are 1001/1002/1003. Development uses three unseen nominal
holdouts (2001-2003) and three perturbed holdouts (3001-3003); confirmation uses
ten of each (2001-2010 and 3001-3010). The frozen `validation_protocol` runtime
object may configure these sets, but they must remain unique/disjoint with at
least these repeat counts. Joint initialization noise is +/-0.005 rad for
selection/nominal and +/-0.01 rad for the predefined perturbation. This is a
joint-initialization perturbation, not a claim about push/disturbance robustness.

### Development Refinement

The default strategy is `legacy_v3`. Set `"refinement_strategy":
"joint_neutral_v4"` in a **new** runtime JSON to test an alternative development
strategy. This requires a development-only task manifest and canonical FK
capture; it rejects confirmation tasks. Prepare a new campaign, and use the
same `run`/`resume` commands above. Existing campaigns and their source snapshots
remain unchanged.

When every selection repeat completes, MimicX retains the native objective and
proposes full-motion replay with sampling ratios 0.45 and 0.65 for its two stages.
It adds no body-specific reward terms in that case. If selection still contains
failures, the existing failure-conditioned diagnosis and proposal generator apply.
The second stage continues from the protected incumbent, not necessarily the
latest training endpoint.

All three methods use the same experimental acceptance rule. In addition to
native safety/horizon checks, every matched selection seed must retain or improve
the mean and p95 of both canonical world-body and root-local FK body errors.
Comparisons use the same-clock, pre-reset common prefix; physical execution
frontiers cannot regress. Acceptance requires a longer execution frontier or at
least a 2% decrease in the sum of the two metre-valued mean errors. A separate
warmstart check prevents successive local/global regressions from accumulating.
This strict rule is an experimental choice, not a demonstrated optimum.

Training budgets remain 3,000 shared warmstart updates and 3,000 additional
updates per arm, with six candidate checkpoints and three selection seeds each.
The selected checkpoint is frozen before three nominal and three perturbed
holdouts. Reused warmstarts retain their original training cost. In this strategy,
`final.pt` copies the selected policy for **all** methods; `selection.json` and
`report.json` record its original checkpoint and whether the warmstart was kept.
The legacy BeyondMimic endpoint convention described below is unchanged for v3.

CPU tests check contracts, symmetric selection, independent holdouts and resume
receipts. New physical training/evaluation is needed to establish improvement;
offline re-selection of old candidates is not evidence of improved training.

Scoring, first-failure censoring, common-prefix error guards and native
qualification reuse `policy_recovery.metrics`. The isolated study gate preserves
the original rule for positive-prefix incumbents and adds per-seed horizon
non-regression. For a zero-prefix startup failure, it permits positive-prefix
recovery only with fixed mean/p95 bounds: world-anchor .08/.15 m, aligned-body
max .10/.20 m, and rotation .20/.35 rad. This includes partial recovery, without
claiming full execution or using training reward. Every gate records its version,
bounds, checks and reasons; the original metrics API remains unchanged.
Diagnosis passes only pre-reset rows to `mine_failures.py`, then uses
the existing repeated-rollout failure-window rule with its body/channel findings.
A first-step failure records that no pre-reset body evidence exists and supplies
only failure timing to the existing generic proposal family. Terminal/reset rows
never supply precision evidence. Failed policies remain outcomes.

These native operational metrics include world anchor and aligned-body errors.
The replay's reward-space reference is policy-anchor aligned; it is **not** the
original world-motion target. When the dataset supplies its canonical named
29-joint/30-body contract, evaluation captures actual joint position/velocity and
all link poses/world velocities at frame zero and after each real step. It calls
the existing cross-controller `sonic_study.common_metrics` against the immutable
NPZ, never the aligned replay reference. All 30 canonical bodies are scored; a
14-body controller input is not substituted for a 30-body scoring contract.
The canonical XML hash must match input QA. Its default location is
`backend_root/src/assets/robots/unitree_g1/xmls/g1.xml`; `common_model_xml` may
point to the identical asset elsewhere. Its source and scorer identity are frozen.
With common scoring enabled, `source_identity` explicitly includes both
`sonic_study.py` and its vendored `sonic_helpers.py`, alongside the canonical XML's
transitive include/mesh/texture closure, `visualization/replay_package.py`, and
motion-study helpers. The model dependency closure reuses the existing SONIC
asset-file resolver. Missing dependencies or any recorded hash drift fail
closed. Freeze a new campaign to adopt this expanded identity; never rewrite an
active snapshot or retroactively change an existing diagnostic receipt.

The native command's cached rotation metric is computed before its reference
advances, unlike the exported position fields. The study therefore preserves
`steps.native_raw.csv` and the per-row `error_anchor_rot_native_cached` value,
then recomputes `error_anchor_rot` from captured anchor quaternions and the
immutable reference at the matching control frame using the existing quaternion
error function. `rotation_clock.json` records the mapping and delta; reset/tail
rows remain unmodified and excluded. No precision threshold is relaxed.
This is evaluation telemetry postprocessing for the new study only. The same
transformation applies to BeyondMimic, Static Search, MimicX and their shared
warmstart evaluations, independent of method or outcome. Cached telemetry compares
the post-step robot with reference frame k; the corrected field compares that
robot with frame k+1, matching the exported position fields. Training, rewards,
native termination, reference data and active warmstarts are unchanged. Historical
snapshots, receipts and numerical tables are never rewritten. Saved actual
captures and copied CSVs suffice for offline verification of this postprocessing;
the correction does not itself require further GPU runs.
Both native-termination metadata fields explicitly describe MjLab automatic reset
and prefix censoring; SONIC's no-automatic-reset description is not inherited.

The shared physical protocol uses strict `>` predicates: pelvis world-z error
.25 m, maximum four-EE pelvis-relative z error .25 m, or pelvis quaternion angle
1 rad. OR the predicates before checking five consecutive samples; failure is
backdated to the run's onset. There is no grace/recovery or absolute standing-height
guard. Common prefix is inclusive 0..1 s (51 samples), or the full shorter motion;
clock tolerance is .03 s, state gap .045 s, and duration ratios .97..1.03.
The complete dataset protocol object/hash is retained, including native termination
definitions. Native failure and common physical failure remain separate.
After the first native reset, capture validity is prefix-only. Nonfinite data is
protocol-invalid, not success. Missing common-scoring metadata in a lightweight
manifest leaves common qualification false, never a fabricated positive result.

Native schedule/entropy remain unchanged, including across warmstart continuation.
Actor, critic, optimizer, iteration and adaptive learning rate restore together.
Atomic durable saves and the existing checkpoint validator reject partial or
out-of-budget weights. A restart advances to the saved iteration plus one;
physical episodes/RNG are restarted, so this is not bitwise resume. Updates lost
after the most recent durable checkpoint may be repeated and must not be hidden
as exact interaction matching. Exact attempted transitions are therefore null;
durable logical transitions and all attempt costs are reported separately.

## Evidence And Costs

Each job stores `job.json`, immutable `contract.json`, `effective_training.json`,
full effective YAML configuration, append-only `run.log`, per-attempt JSON,
selection checkpoints, `final.pt`, `training.json`, and artifact-hashed `done.json`.
The latter is written only after a successful child exit and required evidence
checks. Reuse verifies hashes; source/reference changes fail closed. Duplicate
guards serialize one task/backend/seed and its shared dependency, not resources.
Use the public run/resume entry point, not the private `_worker` subcommand.
Completed arms additionally store `selection.json`, six `gates/slot*.json`,
per-seed validation and holdout receipts/replays, stage patches/diagnosis, and
`report.json`. `report.json.selected_checkpoint` is authoritative: BeyondMimic's
`final.pt` remains its last trained checkpoint, which need not win selection;
Static Search/MimicX root `final.pt` copies the protected selected checkpoint.
All candidate weights remain preserved and hashed, including rejected candidates.
Canonical evaluation additionally stores `native_actual.npz`, `common_steps.npz`
and `common_metrics.json`. Reports retain native qualification and common physical
qualification separately; overall qualification requires both selection and
holdout evidence. Completion of the execution protocol does not imply either.

Attempt JSON includes command, working directory, coordinator PID, inherited
execution context, log/output, exit/error, wall time and child CPU time. Assigned
GPU wall seconds are elapsed allocation exposure, **not** measured GPU compute;
GPU compute and energy stay null without a parent telemetry collector. Sum all
top-level arm attempts, including failed/replayed work. Nested evaluation and
diagnosis attempts are labelled as included in their enclosing arm total; do not
double-count them. Report shared warmstart costs
once at campaign level and separately from arm continuation costs. Actual
transitions per durable PPO update are `num_envs * 24`; updates alone are not
GPU-hours. Running attempts left by host/process death require reconciliation.

## Remaining Integration Gates

The parent must supply clean installed backend/code snapshots and validate a real
scratch/reload/continuation, full evaluation and replay before scaling the matrix.
CPU tests exercise all arms with an explicitly fake simulator and actual existing
diagnosis/scoring/gate logic; they are not physical experiment evidence.
Native configuration and patch CLI checks do not establish policy performance.
Remaining runtime gates include actual full-motion capture/scoring against the
frozen common contract, native closed-loop training/evaluation evidence, Isaac
runtime execution and SONIC deployment/asset-bridge validation.
Confirmation stays scientifically gated on development strategy/budget freeze;
preparing confirmation job records does not authorize launching the full matrix.
