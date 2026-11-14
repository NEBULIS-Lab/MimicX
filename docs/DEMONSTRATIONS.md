# Released Policies and Controllers

Use this entry point for the latest recorded demonstrations. The frozen paper
archive remains available through [Paper Reproduction](REPRODUCIBILITY.md);
continuation checkpoints do not replace its trial identities or measurements.

## Download One Task

```bash
python -m pip install -e '.[artifacts]'
python -m mimicx.runtime.demonstrations list
python -m mimicx.runtime.demonstrations download --task tennis --output artifacts
```

Repeat `--task` for several policies. Omit it to select all nine: `tennis`,
`football`, `dance`, `kungfu`, `track`, `stairs`, `forest`, `bench`, `descent`.
`bench` is Platform Jump. Only selected weights and the shared
motion/runtime bundle are fetched, not the historical checkpoint archive.
Downloading does not execute code.

The shared release also includes scene records. The downloader
selects compatible policy and asset versions automatically using
`mimicx/configs/releases/demonstrations.json`; no file identifiers need to be
copied manually.

The unified HF layout stores actors at `policies/<task>/policy.pt`, with the
paired policy records in root `manifest.json`. Assets use `tasks/<task>/` for
motions and configuration, `runtimes/demonstrations/run_policy.py` for the
runner, and `runtimes/FILES.json` for its runtime/task checksums. The nine
actors and their motion inputs are unchanged by this path migration.
The packaged release spec pins the compatible Policies and Assets revisions;
do not mix either snapshot with an unrelated `main` download. A custom pair
requires an explicit `--release-spec` JSON before the subcommand.

Root `policy_index.json` now uses `mimicx.policy-index.v2`; it is not the
historical `trials` index. The 12-motion comparison checkpoints are separate,
at `comparisons/<task>/<method>/seed-101/policy.pt`, not replacements for the
nine demonstration actors. Licensed BONES raw motions are not redistributed;
`benchmarks/motion-tracking/references.json` supplies the acquisition/conversion
recipe and expected hashes, not the motion bytes.

## Verify and Evaluate

Verification is CPU-only and checks the selected checkpoint, reference,
scene, profile, and complete bundled runtime:

```bash
python -m mimicx.runtime.demonstrations verify \
  --policies artifacts/MimicX-Policies \
  --assets artifacts/MimicX-Assets --task tennis

python -m mimicx.runtime.demonstrations evaluate \
  --policies artifacts/MimicX-Policies \
  --assets artifacts/MimicX-Assets --task tennis \
  --runtime-python /path/to/tracking/bin/python \
  --output outputs/tennis --video --dry-run
```

Inspect the command, then omit `--dry-run` to evaluate in your allocated
compute environment. GPU visibility is inherited unchanged. The tracking
Python environment needs the versions in the downloaded
`MimicX-Assets/runtimes/demonstrations/runtime/requirements.txt`;
the lightweight `artifacts` extra does not install a simulator or PyTorch.
The default seed is the first registered holdout seed; use `--seed` to select
another. Existing output directories are refused to protect prior evidence.

The bundled runner recreates the original evaluation environment.
For further training using the editable implementation, see
[Continuation](CONTINUATION.md). Restoring a checkpoint must preserve its
observation layout, joint ordering, action scaling and motion timing.

## Execution Modes

| Mode | Guide | Meaning |
| :--- | :--- | :--- |
| Released PPO policies | This page | Actor inference in its paired environment |
| Policy continuation | [Continuation](CONTINUATION.md) | PPO updates from an existing checkpoint |
| Parkour completion | [Hybrid Control](HYBRID_CONTROL.md) | Recorded actor prefix followed by predictive control |
| Sports display | [Recording and Contact](RECORDING_AND_CONTACT.md) | Fixed robot-state replay with a dynamically simulated ball |

Parkour is not listed as a standalone PPO actor; its unchanged prefix actor,
controller configuration and recorded trajectory have a separate download in
the [hybrid guide](HYBRID_CONTROL.md). Sports replay preserves
robot states and is not a newly trained ball-interaction policy. Both modes
retain execution provenance.

## Original Paper Archive

For full **historical** HF snapshots, `scripts/reproduction/hf_artifacts.py`
provides CPU operations `verify`, `core-inputs`, `prepare-eval`, and
`verify-gates`. Download **both** repositories at
`archive-before-unified-20261114` into separate directories using the
[archive commands](REPRODUCIBILITY.md#exact-core-input-bundle). Do not point
these operations at unified `main` or a selective demonstration download.

```bash
python scripts/reproduction/hf_artifacts.py verify \
  --root artifacts/archive-before-unified-20261114/MimicX-Policies
python scripts/reproduction/hf_artifacts.py verify-gates \
  --assets artifacts/archive-before-unified-20261114/MimicX-Assets --code .
```

Full-archive verification needs root `FILES.json` and every indexed file; a
selective demonstration download is intentionally smaller. Gate replay
reconstructs decisions from raw archived rollout telemetry. Keep original
paper comparisons and later continuations versioned separately.
