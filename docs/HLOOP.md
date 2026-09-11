# MimicX-HLoop

The `mimicx.hloop-plan.v1` schema describes a fixed DAG of commands, dependencies,
artifact paths/hashes, resource classes (`cpu`, `gpu`, `io`), candidate IDs,
seeds and budgets. Templates support `{repo_root}` and `{run_dir}`. Inspect
`tests/runtime/test_hloop.py` for a small deterministic plan.

```bash
python scripts/runtime/run_hloop.py run --plan plan.json \
  --run-dir runs/sequential --mode sequential
python scripts/runtime/run_hloop.py run --plan plan.json \
  --run-dir runs/hloop --mode hloop
python scripts/runtime/run_hloop.py run --plan plan.json \
  --run-dir runs/bulk-sync --mode bulk-sync
python scripts/runtime/run_hloop.py compare \
  --baseline runs/sequential/hloop_run.json --candidate runs/hloop/hloop_run.json
```

Use a new empty directory for each execution. GPU commands require an existing
compute allocation. In the compatibility schema, `cuda_visible_devices` is a
single **logical visible device index**; the executor forwards it as
`MIMICX_DEVICE_ID` and does not replace the inherited CUDA visibility mask.
MimicX's rollout and training entry points consume that variable. An external
GPU command must likewise select this logical device explicitly; do not assume
that arbitrary third-party commands consume it automatically.

Concurrent GPU jobs are serialized per logical device; CPU and I/O jobs have
separate worker limits. Report comparison checks semantic-plan identity,
candidate IDs, job outcomes, input/output hashes, and selection artifacts.
The CPU fixture provides a deterministic software test, not a performance
benchmark. Real scheduler speed should be measured on the same workload and
resource allocation in each mode, with repeated wall-clock measurements.

The numerical snapshot includes five repeats of each executor over the fixed
rollout/diagnosis workload. HLoop approaches bulk-synchronous throughput while
supporting dependency-ready execution; see [results](assets/results/PROTOCOL.md).

## Frozen Paper Workload

First follow [paper input configuration](REPRODUCIBILITY.md). The resulting
`hloop.yaml` registers the four Fixed Reference seed-101 checkpoints, two
evaluation seeds, three executor modes and five measured repetitions per mode.

```bash
python scripts/autorefine/run_native_hloop_benchmark.py \
  --config runs/paper-config/hloop.yaml --run-dir runs/paper-hloop --prepare-only
```

This CPU preparation writes the fixed 17-job workload: eight rollouts, eight
diagnoses and one report selector. To execute it, rerun without `--prepare-only`
inside a two-visible-GPU compute allocation. It performs no PPO updates. The
selector is a fixed-report reduction, distinct from the repeated policy gate.
