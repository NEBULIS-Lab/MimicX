# Prospective Motion Study Input Audit

The approved shortlist contains 18 task identities: six development and twelve
confirmation tasks (12 BONES-SEED, four AMASS/ACCAD, two LaFAN1). It is not
selected by training performance. The original four video tasks are separate.
Neither historical collaborator policy outcomes nor failure-window tables are
inputs to this implementation. No reserve replacements are selected here.

## Preparation

Run from a source checkout using an existing Python environment with NumPy and
MuJoCo. This command performs CPU FK only, with no renderer, training, network,
or optional simulator startup. Supply authorized local sources explicitly:

```bash
python scripts/datasets/prepare_motion_study.py \
  --output /path/to/new-empty-data-directory \
  --g1-xml /path/to/canonical/unitree_g1/xmls/g1.xml \
  --bones-root /path/to/bones/converted/g1/csv \
  --bones-metadata /path/to/bones/metadata/seed_metadata_v004.csv \
  --amass-root /path/to/amass_g1/mjlab_npz \
  --amass-original /path/to/ACCAD \
  --lafan-root /path/to/lafan1_g1_full41 \
  --lafan-csv /path/to/lafan1_g1
```

Exit 0 means all entries passed; exit 2 means a complete audit was written with
blocked entries. Existing nonempty outputs are never overwritten. Sources are
copied byte-for-byte and hash-verified; derived motion files are separate.
Keep outputs outside public source control. Dataset permission and licenses
still apply; the code does not grant redistribution rights.

## Runner Contract

`manifest.json` uses schema `mimicx.motion-study.v1`, `root: "."`, and a `tasks`
list. Each task has `task_id`, `split`, `source`, `source_path`, `motion_path`,
`frames`, `fps`, `duration`, `sha256`, `quality_status`, and `quality.status`.
Statuses are exactly `passed` or `blocked`; both status fields must agree.
Missing or malformed inputs can have null paths/numerical fields/hashes.
Blocked but well-formed inputs retain prepared files for inspection, not training.

Paths are relative to the manifest directory, or an explicitly supplied data
root after transfer. `sha256` identifies the prepared NPZ, not the source NPZ.
`duration = (frames - 1) / fps` measures elapsed sample time; it is not `frames / fps`.
Windows include both boundary frames. The NPZ preserves world-coordinate arrays,
50 Hz timing and WXYZ quaternion order; names are recovered from the canonical
asset only after full-source and selected-window FK agreement. Window offsets
and original source FPS/frame counts are recorded separately.

`load_eligible_manifest(path, subset="all", root=None)` verifies hashes and fails
the selected cohort closed if any task is blocked. `all` always means all 18,
not just development or successful inputs. `development` and `confirmation`
contain six and twelve entries respectively. Runners selecting an individual
pilot still must check its status and hash. Never omit blocked tasks silently
from study denominators or substitute tasks based on outcomes.

## Input Gates

- Full-array schema, finite values, positive FPS, and unit rotations.
- Source elapsed time versus converted elapsed time within one output interval
  (20 ms at 50 Hz), allowing endpoint rounding but not silent retiming.
- Nine spread-out frames for full-source and selected-window canonical G1 CPU
  MuJoCo FK, at most 2 mm position and 0.002 rad rotation error across all bodies.
- Every source frame against canonical joint limits, with 0.0001 rad tolerance.
- LaFAN source CSV joint interpolation comparison at 0.0001 rad tolerance.
- Full takes by default; prospective LaFAN repeated-jump window 5.5-11.5 s and
  first complete fall/get-up window 3.0-19.5 s. These were chosen from reference
  pelvis/ankle height traces, not policy performance. The latter intentionally
  exceeds 12 s to retain the full transition. No visual/physics certification
  is implied by kinematic window inspection.
- Crawl/get-up remain blocked until shared ground-contact termination semantics
  are frozen. Absolute standing pelvis-height termination is inappropriate.

BONES provenance uses the original release metadata path and frame count at
120 Hz, together with transformed NPZ identity and independent FK checks. Its
large raw CSV archive is not re-extracted or re-retargeted by this tool; this is
not a claim of byte-for-byte reproduction of the original retargeting process.
LaFAN uses the documented 30 Hz robot CSV source, not purported paired human
SMPL data. AMASS reads frame count/FPS from original ACCAD files without loading
pickle objects. Original sources and FPS are retained even when timing fails.

No interpolation or repair is silently applied to these already-50-Hz inputs.
Timing/provenance discrepancies require a separately reviewed new conversion
version. `Form_1` and the misclassified ACCAD entry are not in the shortlist.
Contact ranges are diagnostic link heights, not dynamic feasibility or collider
contact checks; input eligibility does not promise tracking success.

## Explicit Source-Time Rebuild

`--regenerate-sources` creates a separate audit version from the six original
robot CSVs. Also supply `--amass-csv`, `--gmr-timing-source` (GMR `utils/smpl.py`),
`--csv-converter-source` (Unitree `scripts/csv_to_npz.py`), and
`--amass-producer-source` (`retarget_amass_smplx_to_g1.py`). These producer files
are archived with hashes; their endpoint resampling and named CSV joint order
are checked. This is an input conversion correction, not outcome-based repair.

The inspected AMASS chain uses `floor(N/4)` retargeted frames spanning source
indices `0..N-1`, then incorrectly treats those rows as nominal 30 Hz. Rebuild
uses `linspace(0, (N-1)/120, M)` source timestamps and the original 120 Hz elapsed
time, not the rounded FPS in historical conversion metadata. Adjacent CSV joints
must first reproduce the inherited nominal-30-Hz conversion within 0.0001 rad.

LaFAN CSVs retain their documented 30 Hz clock. The inherited long-sequence
interpolation differs from float64 reference interpolation by approximately
0.00024-0.00028 rad; exact float32 producer replay did not fully explain it, so
the old files remain blocked. Rebuilt arrays use float64 linear interpolation
for positions/joints, SLERP for quaternions, and CPU canonical FK for every frame.
Velocities use MuJoCo position differentiation and body-origin Jacobians in world
coordinates, not COM velocities. No QA threshold changes. Both old diagnostics
and new source comparisons remain in the new manifest.

The rebuilt audit resolves the six numerical input issues. Crawl/get-up still
require the common termination contract. Keep the old manifest immutable and
explicitly select the new version for all methods before training.

## Prospective Ground-Contact Protocol

The jointly reviewed `ground_contact_protocol` in the shortlist config freezes
the same common evaluator for every method. It is separate from native training
termination and does not alter reference arrays or numerical input QA.

- Root: `pelvis`; absolute difference in actual/reference world z, strict >0.25 m.
- Critical bodies: `left_ankle_roll_link`, `right_ankle_roll_link`,
  `left_wrist_yaw_link`, `right_wrist_yaw_link`. Maximum absolute error of their
  pelvis-relative z against the reference, strict >0.25 m.
- Pelvis orientation: quaternion geodesic `2*acos(abs(dot(q,q_ref)))` after
  normalization and clamping, strict >1 rad. This includes yaw error.
- OR the three predicates before counting five consecutive 50 Hz bad samples;
  backdate failure to the first sample of that run. No grace or recovery.
- Common prefix includes 0..1 s (51 samples), or the entire shorter reference.
  Missing prefixes are null, not extrapolated. Startup failure is frame zero.
- Clock tolerance 0.03 s; maximum state gap 0.045 s; controller/simulator duration
  ratios in [0.97, 1.03]. No onset fitting, time warping, or pose alignment.
- Nonfinite data is protocol-invalid, never success. Native reset masks are
  prefix-only. Native failures remain separately reported; an actual early stop
  cannot count as completion. Ground contact itself is not a failure condition.

The native three-arm training contract stays unchanged: `torso_link` reference-z
error >0.25 m, absolute difference of torso projected-gravity z >0.8 (not radians),
or maximum reference-z error of the same four end effectors >0.25 m, each checked
on the current sample with zero initialization grace. There is no absolute
standing-height or native bad-contact guard. These native operational predicates
must not be relabeled as the common physical evaluator.

After both owners agree to the exact protocol digest, issue a new artifact:

```bash
python scripts/datasets/prepare_motion_study.py \
  --freeze-protocol-from /path/to/v4/manifest.json \
  --output /path/to/new-protocol-version \
  --agreements /path/to/local-owner-agreements.json
```

Agreement records have `native_coordinator` and `sonic_scorer` entries containing
`protocol_sha256` and a nonempty `reference` to the explicit agreement. The digest
is SHA-256 of UTF-8 JSON sorted by key with compact separators. These are local
audit records, not cryptographic signatures. The new manifest embeds `protocol`,
`protocol_sha256`, agreements, and its parent manifest hash; every task records
the protocol ID/hash. Only the exact unresolved-protocol reason is removed from
the two ground-contact tasks. Any other blocker remains. Runtime consumers must
match the frozen protocol, not substitute their own thresholds or persistence.
All prior artifacts remain immutable. Eligibility does not certify backend
readiness or override separate asset compatibility gates.

`export_isaac_pilot(manifest, task_id, live_articulation_json, output)` converts
one eligible task using the Isaac adapter and records a named permutation,
hashes, and source-canonical FK. The actual live Isaac asset FK is explicitly
pending: reordered source FK does not establish cross-simulator equivalence.

## CPU Tests

```bash
python -m unittest discover -s tests -p test_motion_study.py -v
```

Without MuJoCo the CPU FK tests skip explicitly. Run the same suite in an
existing physics environment for complete coverage. Tests use generated arrays
and a synthetic articulated fixture; no licensed motion data is bundled.
