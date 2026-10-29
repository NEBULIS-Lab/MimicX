# Expanding the motion study

MimicX includes an offline input-audit tool for a prospective, grouped motion
study. It complements the original 18-input preparation workflow without
changing existing experiments.

The current design contains **60 distinct source clips**, split into 12
development and 48 confirmation inputs. The original six development and twelve
confirmation inputs retain their splits, motion arrays and scored windows. The
additional inputs are selected deterministically using source identities and
input quality checks, not policy performance. A further 24 inputs are declared
as reserves for documented input defects. These numbers describe prepared
inputs, not completed policies or experimental results.

## Prepare inputs

Provide a local JSON configuration containing the source paths accepted by
`mimicx.datasets.motion_scaleup.load_roots`. The required inputs include the
authorized BONES G1 archive and metadata, existing converted motions, the
canonical G1 model, the original study manifest, and the conversion source used
for provenance. No dataset or licensed model is downloaded by this tool.
Use a CPU environment with NumPy, SciPy and MuJoCo installed.

```bash
CUDA_VISIBLE_DEVICES='' python scripts/datasets/prepare_motion_scaleup.py inventory \
  --roots /path/to/local-source-roots.json \
  --output /path/to/new-inventory.json

CUDA_VISIBLE_DEVICES='' python scripts/datasets/prepare_motion_scaleup.py prepare \
  --roots /path/to/local-source-roots.json \
  --inventory /path/to/new-inventory.json \
  --declared-at 2026-10-29T07:00:00+00:00 \
  --output /path/to/new-cohort
```

Use the actual declaration time for a new study. Outputs must be new and empty.
The tool streams the archive, captures the needed raw inputs, verifies original
timing and units, compares converted values to the raw source, and checks
canonical joint limits and forward kinematics. Mirrors and related takes are
grouped to prevent overlap between development and confirmation sets. Original
clip duration and evaluated-window duration are recorded separately.

The output includes the input manifest, cohort declaration, development subset,
raw-source provenance and candidate audit. Rejected inputs remain in the audit;
they are not silently counted as eligible. The larger coverage list is a
prospective input list, not evidence of numerical eligibility, successful
execution, or a trained MimicX policy for every entry.

## Audit the larger coverage list

The prepared cohort also supplies a separate prospective coverage list. Audit
it without training or policy evaluation:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/datasets/prepare_motion_scaleup.py coverage \
  --roots /path/to/local-source-roots.json \
  --inventory /path/to/new-inventory.json \
  --plan /path/to/new-cohort/coverage-input-plan.json \
  --output /path/to/new-coverage-audit
```

This CPU-only pass converts the fixed 1,000-input list to canonical 50 Hz motion
arrays and retains every record, including missing or invalid inputs. It uses a
bounded output budget and does not replace rejected inputs. Consumers must check
that the manifest is complete, validate the referenced files, and evaluate only
eligible motions while retaining the full input denominator in reporting.

## Run and report

The generic `scripts/experiments/motion_study.py` coordinator accepts the prepared
manifest. Prepare shared native warmstarts separately from refinement runs.
Keep confirmation inputs locked until the development strategy, candidate
budget, selection rule and evaluation protocol are frozen. The existing
development-only refinement guard remains in force; changing split labels is
not a way to enable confirmation experiments.

For comparisons, report distinct source clips, training seeds, evaluation seeds,
shared warmstart cost and continuation/search cost separately. BeyondMimic's
MjLab implementation and released SONIC have different training and deployment
roles; released-policy execution is not additional PPO training. Preserve valid
unsuccessful outcomes alongside improvements and protected incumbents.

Source data, local path configurations, run outputs and operational scheduling
records are not part of the public code release. Dataset licenses continue to
apply.

## Tests

```bash
python -m pytest tests/test_motion_scaleup.py tests/test_motion_study.py -q
```

These CPU tests cover source identity, timing/unit conversion, group-disjoint
selection and input validation. They do not certify simulator deployment or
policy performance.
