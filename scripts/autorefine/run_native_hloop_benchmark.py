#!/usr/bin/env python3
"""Run the native sequential/bulk-sync/HLoop AutoRefine execution benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import shutil
import statistics
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mimicx.runtime.hloop import ArtifactSpec, HLoopJob, HLoopPlan, digest_json, run_hloop_plan, sha256_file
from mimicx.runtime.hloop_baselines import run_hloop_plan_bulk_sync


MODES = ("sequential", "bulk_sync", "e0")


def _task_payload(task: str, config: dict[str, object]) -> dict[str, object]:
    return yaml.safe_load((Path(str(config["task_dir"])) / f"{task}.yaml").read_text())


def build_plan(config: dict[str, object]) -> HLoopPlan:
    runtime_python = Path(str(config["runtime_python"]))
    mjlab_root = Path(str(config["backend"]))
    jobs: list[HLoopJob] = []
    diagnosis_ids: list[str] = []
    candidate_args: list[str] = []
    gpus = [str(item) for item in config["gpus"]]
    index = 0
    for task in config["tasks"]:
        payload = _task_payload(str(task), config)
        checkpoint = Path(str(payload["hloop_checkpoint"]))
        motion = Path(str(payload["motion_file"]))
        for seed in config["evaluation_seeds"]:
            candidate = f"{task}_seed{seed}"
            rollout_id = f"rollout_{candidate}"
            diagnosis_id = f"diagnose_{candidate}"
            gpu = gpus[index % len(gpus)]
            index += 1
            metrics = f"{{run_dir}}/candidates/{candidate}/metrics.json"
            steps = f"{{run_dir}}/candidates/{candidate}/metrics_steps.csv"
            report = f"{{run_dir}}/candidates/{candidate}/failure_report.json"
            jobs.append(HLoopJob(
                job_id=rollout_id,
                resource="gpu",
                cuda_visible_devices=gpu,
                candidate_id=candidate,
                seed=int(seed),
                budget={"rollouts": 1, "policy_updates": 0},
                cwd=str(mjlab_root),
                environment={
                    "PYTHONPATH": f"{mjlab_root}:{mjlab_root / 'src'}:{ROOT}",
                },
                command=(
                    str(runtime_python), str(ROOT / "scripts/rollout_mjlab_motion_smoke.py"),
                    "--task", "Unitree-G1-Tracking-MimicX-Curriculum",
                    "--motion-file", str(motion), "--num-envs", "1",
                    "--steps", str(payload["horizon"]), "--seed", str(seed),
                    "--device", "cuda:0", "--disable-joint-init-noise",
                    "--checkpoint-file", str(checkpoint),
                    "--metrics-output", metrics, "--step-metrics-output", steps,
                ),
                inputs=(
                    ArtifactSpec("motion", str(motion), sha256_file(motion)),
                    ArtifactSpec("checkpoint", str(checkpoint), sha256_file(checkpoint)),
                ),
                outputs=(ArtifactSpec("metrics", metrics), ArtifactSpec("steps", steps)),
            ))
            jobs.append(HLoopJob(
                job_id=diagnosis_id,
                resource="cpu",
                depends_on=(rollout_id,),
                cwd=str(ROOT),
                command=(
                    str(runtime_python), str(ROOT / "scripts/autorefine/mine_failures.py"),
                    "--metrics-json", metrics, "--step-csv", steps, "--output-json", report,
                ),
                inputs=(ArtifactSpec("metrics", metrics), ArtifactSpec("steps", steps)),
                outputs=(ArtifactSpec("failure_report", report),),
            ))
            diagnosis_ids.append(diagnosis_id)
            candidate_args.extend(("--candidate", candidate, report))
    jobs.append(HLoopJob(
        job_id="select_candidates",
        resource="io",
        depends_on=tuple(diagnosis_ids),
        cwd=str(ROOT),
        command=(
            str(runtime_python), str(ROOT / "scripts/autorefine/select_hloop_native.py"),
            *candidate_args, "--output", "{run_dir}/selection.json",
        ),
        inputs=tuple(
            ArtifactSpec("failure_report", f"{{run_dir}}/candidates/{task}_seed{seed}/failure_report.json")
            for task in config["tasks"] for seed in config["evaluation_seeds"]
        ),
        outputs=(ArtifactSpec("selection", "{run_dir}/selection.json"),),
    ))
    plan = HLoopPlan(
        schema_version="mimicx.hloop-plan.v1",
        plan_id="native_core4_rollout_diagnose_select_v1",
        jobs=tuple(jobs),
        max_cpu_workers=8,
        max_io_workers=1,
        selection_artifact="{run_dir}/selection.json",
    )
    plan.validate()
    return plan


def _run_mode(plan: HLoopPlan, mode: str, output: Path) -> dict[str, object]:
    terminal = output / "hloop_run.json"
    if terminal.is_file():
        return json.loads(terminal.read_text())
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Incomplete benchmark output retained for inspection: {output}")
    if mode == "bulk_sync":
        return run_hloop_plan_bulk_sync(plan, run_dir=output, repo_root=ROOT)
    return run_hloop_plan(plan, run_dir=output, repo_root=ROOT, mode=mode)


def _selection_hash(report: dict[str, object]) -> str:
    selection = report["selection"]["value"]
    decision = {
        "selected_candidate_ids": selection["selected_candidate_ids"],
        "candidate_decisions": [
            {
                "candidate_id": candidate["candidate_id"],
                "accepted": candidate["accepted"],
            }
            for candidate in selection["candidates"]
        ],
    }
    return digest_json(decision)


def run_benchmark(plan: HLoopPlan, config: dict[str, object], run_dir: Path) -> dict[str, object]:
    repetitions = int(config["repetitions"])
    records = []
    warmup_count = 1 if config.get("warmup", True) else 0
    for repetition in range(-warmup_count, repetitions):
        label = "warmup" if repetition < 0 else f"repetition_{repetition:02d}"
        reports = {}
        order = MODES[repetition % len(MODES):] + MODES[:repetition % len(MODES)]
        for mode in order:
            reports[mode] = _run_mode(plan, mode, run_dir / label / mode)
        hashes = {_selection_hash(report) for report in reports.values()}
        if len(hashes) != 1:
            raise RuntimeError(f"selection decision parity failed in {label}")
        if repetition >= 0:
            records.append({"repetition": repetition, "order": order, "reports": reports, "selection_hash": hashes.pop()})
    times = {mode: [float(row["reports"][mode]["elapsed_seconds"]) for row in records] for mode in MODES}
    medians = {mode: statistics.median(values) for mode, values in times.items()}
    summary = {
        "schema": "mimicx.native-hloop-benchmark-summary.v1",
        "repetitions": repetitions,
        "plan_semantic_hash": plan.semantic_hash(),
        "selection_hashes": sorted({row["selection_hash"] for row in records}),
        "decision_parity": len({row["selection_hash"] for row in records}) == 1,
        "median_seconds": medians,
        "mean_seconds": {mode: statistics.fmean(values) for mode, values in times.items()},
        "speedup_hloop_vs_sequential": medians["sequential"] / medians["e0"],
        "speedup_hloop_vs_bulk_sync": medians["bulk_sync"] / medians["e0"],
        "every_repetition_hloop_faster_than_sequential": all(
            row["reports"]["e0"]["elapsed_seconds"] < row["reports"]["sequential"]["elapsed_seconds"] for row in records
        ),
        "timings": [
            {"repetition": row["repetition"], **{mode: row["reports"][mode]["elapsed_seconds"] for mode in MODES}}
            for row in records
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Native MimicX-HLoop Benchmark", "",
        "| Executor | Median (s) | Mean (s) |", "|---|---:|---:|",
        *[f"| {mode} | {medians[mode]:.3f} | {summary['mean_seconds'][mode]:.3f} |" for mode in MODES],
        "", f"HLoop vs sequential: **{summary['speedup_hloop_vs_sequential']:.3f}x**.",
        f"HLoop vs bulk-sync: **{summary['speedup_hloop_vs_bulk_sync']:.3f}x**.",
    ]
    (run_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpus", default=None, help="comma-separated runtime GPU override")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config.get("schema") != "mimicx.native-hloop-benchmark.v1":
        raise ValueError("unsupported native HLoop benchmark schema")
    if args.gpus:
        config["gpus"] = [int(item) for item in args.gpus.split(",") if item.strip()]
    plan = build_plan(config)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {**asdict(plan), "semantic_hash": plan.semantic_hash()}
    (args.run_dir / "materialized_plan.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    print(f"prepared HLoop jobs={len(plan.jobs)} hash={plan.semantic_hash()}")
    if args.prepare_only:
        return 0
    run_benchmark(plan, config, args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
