# Frozen Confirmation

The optional `joint_adaptive_precision_confirmation_v1` route executes the
provided-motion confirmation protocol using the frozen v6 proposal rule and the
unchanged `joint_neutral_v4` local/global gate. It does not alter the original
video-to-policy cohort or legacy development defaults. V7 is not used or bundled.

## Fixed Contract

| Item | Required value |
|---|---|
| Eligible inputs | Exactly 12 unique tasks, all labeled `confirmation` |
| Backend | `mjlab` only; native Isaac remains a separate integration |
| Training seeds | 101 and 202 |
| Shared warmstart | 3000 updates per task/seed, fresh scratch training |
| Continuations | BeyondMimic, Static Search, MimicX; 3000 additional updates each |
| Selection | Six slots per continuation; seeds 1001, 1002, 1003 |
| Final nominal seeds | 21001, 21002, 21003 |
| Final perturbed seeds | 22001, 22002, 22003 |
| Joint initialization noise | Selection/nominal 0.005; perturbed 0.01 |
| Replication version | `mimicx.confirmation-12x2-3plus3.v1` |

The full matrix contains 24 shared warmstarts and 72 continuations, totaling
96 training phases and 288,000 updates. Donor-selection, candidate-selection and
final trained-policy evaluation counts are 72, 1296 and 432 respectively.
Released SONIC's separate 72 executions do not receive per-task PPO training.
No completed experiment is implied by preparing this matrix.

The v6 rule uses v5 precision proposals unless selection-only feedback meets its
explicit native-replay condition. This dependency does not introduce a different
confirmation strategy. Checkpoint selection is recorded before holdout execution;
negative, tied and failed outcomes remain outcomes, not reasons to retune the
freeze. The fixed gate and all three methods retain equal selection budgets.

## Prepare at the Destination

Install the pinned [tracking backend](../dependencies/README.md) in a separate
environment. Supply the [motion-study input manifest](MOTION_STUDY_EXECUTION.md#local-inputs)
with canonical 29-joint/30-body names, XML identity, passed input QA and the frozen
common physical protocol. Inputs and runtime configuration stay outside Git.

Copy code and authorized inputs to their final destination before preparing a
new campaign. Source and input identities are byte hashes, independent of Python
AST versions. A previously prepared campaign binds absolute paths; do not rewrite
its paths or recreate its contract after outcomes. Keep original frozen runs intact.

Create a new runtime JSON, replacing these illustrative paths with installed
locations and retaining the dataset's common protocol:

```json
{
  "backend_root": "/path/to/pinned-backend",
  "python": "/path/to/tracking-env/bin/python",
  "common_model_xml": "/path/to/pinned-backend/src/assets/robots/unitree_g1/xmls/g1.xml",
  "device": "cuda:0",
  "num_envs": 4096,
  "steps_per_env": 24,
  "backends": ["mjlab"],
  "refinement_strategy": "joint_adaptive_precision_confirmation_v1",
  "seeds": {"confirmation": [101, 202]},
  "validation_protocol": {
    "selection_seeds": [1001, 1002, 1003],
    "selection_noise": 0.005,
    "nominal_noise": 0.005,
    "perturbed_noise": 0.01,
    "holdout": {
      "confirmation": {
        "nominal": [21001, 21002, 21003],
        "perturbed": [22001, 22002, 22003]
      }
    }
  }
}
```

This CPU-only preparation runs from the source checkout. Its output contract
and campaign directory must be new; no simulator or Torch is imported.
The runtime JSON must not belong to an existing run.

```bash
python - /path/to/manifest.json /path/to/runtime.json /path/to/freeze.json /path/to/campaign <<'PY'
import json
import sys
from pathlib import Path
from mimicx.experiments.motion_study import digest, prepare
from mimicx.experiments.motion_study_confirmation import (
    build_confirmation_contract, verify_confirmation_contract,
)

manifest, runtime_path, freeze_path, output = map(Path, sys.argv[1:])
if freeze_path.exists() or output.exists():
    raise SystemExit("Use new freeze and campaign paths")
build_confirmation_contract(manifest, runtime_path, freeze_path)
runtime = json.loads(runtime_path.read_text())
runtime["confirmation_contract"] = str(freeze_path.resolve())
runtime["confirmation_contract_sha256"] = digest(freeze_path)
runtime_path.write_text(json.dumps(runtime, indent=2) + "\n")
campaign = prepare(manifest, runtime_path, output)
verify_confirmation_contract(campaign)
assert len(campaign["jobs"]) == 96
print("Prepared and verified 96 jobs; none executed")
PY
python scripts/experiments/motion_study.py jobs --campaign /path/to/campaign/campaign.json
```

Preparation and execution verify the external contract's hash and complete
normalized payload: source closure, reference/manifest identities, task records,
jobs/stages, seeds, protocol, budgets and runtime. Changed or missing contracts
fail closed. Confirmation donor adoption is disabled. Preparation does not certify
input provenance/group separation or grant access to compute resources.

## Execute and Resume

The existing `run` and `resume` interfaces are unchanged. After obtaining suitable
compute resources, invoke one declared job in the prepared tracking environment:

```bash
python scripts/experiments/motion_study.py run \
  --campaign /path/to/campaign/campaign.json --task TASK_ID \
  --seed 101 --backend mjlab --method warmstart
python scripts/experiments/motion_study.py resume \
  --campaign /path/to/campaign/campaign.json --task TASK_ID \
  --seed 101 --backend mjlab --method mimicx
```

Use `beyond_mimic` and `static_search` for the other arms, and both declared
training seeds. The coordinator checks dependencies, source identity and durable
completion receipts. Resource scheduling belongs to the caller. Starting a process
or receiving exit code zero alone is not evidence of a completed policy run.

Legacy `joint_neutral_v4`, `joint_precision_v5` and `joint_adaptive_precision_v6`
remain development-only. The legacy protocol still defaults to 10+10 confirmation
holdouts and three training seeds; only the explicit frozen route selects the
reduced protocol. Do not change these historical defaults to replay this campaign.

## CPU Validation

```bash
CUDA_VISIBLE_DEVICES= python -m pytest -q \
  tests/test_confirmation_release.py tests/test_motion_study_confirmation.py \
  tests/test_motion_study_v4.py tests/test_motion_study_v5.py tests/test_motion_study_v6.py \
  tests/test_motion_study_campaign.py tests/test_motion_study_adoption.py
CUDA_VISIBLE_DEVICES= python -m pytest -q tests/isaac tests/autorefine tests/runtime
```

Tests cover freeze/prepare/verify, tampering, exact budgets and seeds, worker
dispatch, byte-frozen proposal/gate functions, checkpoint selection, resume and
holdout isolation using CPU fixtures. They do not claim new physical evaluation.
