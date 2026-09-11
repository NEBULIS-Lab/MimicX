# Numerical Snapshot

| File | Rows | Contents |
|---|---:|---|
| `core_method_task_summary.csv` | 16 | Four methods across four tasks |
| `core_trial_results.csv` | 48 | Per-continuation-seed results |
| `accept_protect_decisions.csv` | 12 | Complete execution-gate decisions |
| `hloop_timings.csv` | 15 | Three executor modes, five repeats |
| `sonic_comparison.csv` | 2 | Released-policy transfer comparison |
| `additional_video_tasks.csv` | 4 | Additional single-person inputs |
| `motion_breadth.csv` | 14 | Motion-input task coverage |
| `reward_higher_regression_rejection.csv` | 3 | Historical reward/execution disagreement |
| `tennis_training_dynamics_3000_points.csv` | 3000 | Logged PPO iteration telemetry |

`provenance.json` records source and published SHA-256 hashes and the exact
metadata columns omitted. Numeric values are preserved as strings during CSV
export; rows are not filtered by outcome. No checkpoint paths, device-placement
metadata or private log paths are published. Regenerate with
`scripts/export_release_results.py --source-dir ... --output-dir ...`.

Definitions and comparison protocols are in [PROTOCOL.md](PROTOCOL.md).
The time series contains twelve 250-point traces (four methods, three seeds),
not a single 3000-iteration training run. HLoop's internal `e0` ID means
MimicX-HLoop. Checkpoint weights are not included in this snapshot.
