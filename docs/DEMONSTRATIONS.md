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
`bench` is Platform Jump. Only selected weights and the shared, checksummed
motion/runtime bundle are fetched, not the historical checkpoint archive.
Downloading does not execute code.

The shared release also contains its scene records and example media, because
the immutable runner verifies the complete runtime release before execution.

`mimicx/configs/releases/demonstrations.json` pins both repositories and the
SHA-256 hashes of their manifest and runtime dependency indexes:

| Artifact | Immutable revision |
| :--- | :--- |
| Policies | `2ecacde275a3b70c97fe82b44ce45178e8df3d01` |
| Assets | `3c7eb5214e6137bafb1a101c32d8c4c42c32a8e3` |

`releases/2026-10-14-demonstrations` reuses the frozen scene runtime in
`releases/2026-10-11-scenes`. These snapshots form a pair. Do not mix actors
with unrelated references, scenes or observation configurations. A custom
pair requires an explicit `--release-spec` JSON.

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
Python environment needs the versions in the downloaded `requirements.txt`;
the lightweight `artifacts` extra does not install a simulator or PyTorch.
The default seed is the first registered holdout seed; use `--seed` to select
another. Existing output directories are refused to protect prior evidence.

The immutable runner carries the original environment construction code.
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

Parkour is not listed as a standalone PPO actor; its controller configuration
and recorded trajectory accompany the HF release. Sports replay preserves
robot states and is not a newly trained ball-interaction policy. Both modes
retain execution provenance.

## Original Paper Archive

For full HF snapshots, `scripts/reproduction/hf_artifacts.py` provides CPU
operations `verify`, `core-inputs`, `prepare-eval`, and `verify-gates`.

```bash
python scripts/reproduction/hf_artifacts.py verify --root /path/to/full/MimicX-Policies
python scripts/reproduction/hf_artifacts.py verify-gates \
  --assets /path/to/full/MimicX-Assets --code .
```

Full-archive verification needs root `FILES.json` and every indexed file; a
selective demonstration download is intentionally smaller. Gate replay
reconstructs decisions from raw archived rollout telemetry. Keep original
paper comparisons and later continuations versioned separately.
