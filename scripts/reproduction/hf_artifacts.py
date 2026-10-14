#!/usr/bin/env python3
"""CPU-only preparation of paired MimicX Hugging Face artifact snapshots."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys
import tempfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes snapshot: {name}")
    return path


def verify(root):
    root = Path(root).resolve()
    records = json.loads((root / "FILES.json").read_text())
    for record in records:
        path = safe_path(root, record["path"])
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise ValueError(f"Missing or changed artifact: {record['path']}")
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"Size mismatch: {record['path']}")
    return {"files": len(records), "bytes": sum(row["bytes"] for row in records)}


def resolve_uri(uri, policies, assets):
    for prefix, root in (("policies://", policies), ("assets://", assets)):
        if uri.startswith(prefix):
            return safe_path(root, uri[len(prefix):])
    raise ValueError(f"Not a bundled artifact: {uri}")


def materialize_core(policies, assets, output):
    verify(policies)
    verify(assets)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    recipe = json.loads((Path(assets) / "configs/core_input_map.json").read_text())
    for target, uri in recipe["files"].items():
        source = resolve_uri(uri, policies, assets)
        destination = safe_path(output, target)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    shutil.copy2(Path(assets) / "configs/core_inputs.json", output / "manifest.json")
    return {"materialized_assets": len(recipe["files"]), "manifest": str(output / "manifest.json")}


def prepare_evaluation(policies, assets, code, backend, runtime_python, trial_id, output, external_motion=None, video=False):
    policies, assets = Path(policies).resolve(), Path(assets).resolve()
    index = json.loads((policies / "policy_index.json").read_text())
    matching = [row for row in index["trials"] if row["id"] == trial_id]
    if len(matching) != 1:
        raise ValueError(f"Unknown or ambiguous trial ID: {trial_id}")
    row = matching[0]
    checkpoint = resolve_uri(row["checkpoint"], policies, assets)
    if sha256(checkpoint) != row["checkpoint_sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    if row["reference"].startswith("assets://"):
        motion = resolve_uri(row["reference"], policies, assets)
    elif external_motion:
        motion = Path(external_motion).resolve()
    else:
        raise ValueError("This licensed dataset reference is not bundled; provide --external-motion")
    if sha256(motion) != row["reference_sha256"]:
        raise ValueError("Reference checksum mismatch; this is not the registered paper input")
    code, backend = Path(code).resolve(), Path(backend).resolve()
    rollout = code / "scripts/rollout_mjlab_motion_smoke.py"
    if not rollout.is_file() or not (backend / "scripts/train.py").is_file():
        raise FileNotFoundError("Provide a MimicX checkout and its installed tracking backend")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    commands = []
    lines = ["#!/usr/bin/env bash", "set -euo pipefail",
             f"cd {shlex.quote(str(backend))}",
             f"export PYTHONPATH={shlex.quote(str(code) + ':' + str(backend) + ':' + str(backend / 'src'))}:${{PYTHONPATH:-}}",
             "export MUJOCO_GL=egl", "export WANDB_MODE=offline"]
    lines.append("unset MIMICX_AUTOREFINE_PATCH_FILE")
    if row.get("evaluation_patch"):
        patch = resolve_uri(row["evaluation_patch"], policies, assets)
        if not patch.is_file():
            raise FileNotFoundError(patch)
        lines.append(f"export MIMICX_AUTOREFINE_PATCH_FILE={shlex.quote(str(patch))}")
    for seed in row["verification_seeds"]:
        prefix = output / f"seed_{seed}"
        argv = [str(Path(runtime_python).absolute()), str(rollout), "--task", row["evaluation_task"],
                "--motion-file", str(motion), "--checkpoint-file", str(checkpoint),
                "--num-envs", "1", "--steps", str(row["horizon"]), "--seed", str(seed),
                "--device", "cuda:0", "--disable-joint-init-noise",
                "--metrics-output", str(prefix / "metrics.json"),
                "--step-metrics-output", str(prefix / "metrics_steps.csv")]
        if video:
            argv.extend(["--video-output", str(prefix / "rollout.mp4"),
                         "--video-width", "1920", "--video-height", "1080",
                         "--screenshot-dir", str(prefix / "screenshots"), "--screenshot-every", "100"])
        lines.append(f"mkdir -p {shlex.quote(str(prefix))}")
        lines.append(shlex.join(argv))
        commands.append(argv)
    script = output / "evaluate.sh"
    script.write_text("\n".join(lines) + "\n")
    (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    return {"trial": trial_id, "commands": len(commands), "script": str(script), "executed": False}


def verify_gates(assets, code):
    sys.path.insert(0, str(Path(code).resolve()))
    from mimicx.refinement.gate import aggregate_repeats, decide_acceptance, load_rollout_metrics
    from mimicx.refinement.manifest import GuardSpec, VerificationSpec
    assets = Path(assets).resolve()
    records = json.loads((assets / "evidence/core_gate_index.json").read_text())
    outcomes = []
    for record in records:
        selection = json.loads(safe_path(assets, record["selection"]).read_text())
        config = record["verification"]
        spec = VerificationSpec(repeats=int(config["repeats"]), seeds=tuple(config["seeds"]),
                                metric_keys=tuple(config["metric_keys"]),
                                guards={key: GuardSpec(**value) for key, value in config["guards"].items()})
        with tempfile.TemporaryDirectory(prefix="mimicx_gate_") as temporary:
            aggregates = []
            for original in [selection["incumbent"], *selection["candidates"]]:
                repeats = []
                for item in original["records"]:
                    source = item["source"]
                    if not source.startswith("assets://"):
                        raise ValueError(f"Missing exported rollout source: {source}")
                    metrics = safe_path(assets, source[len("assets://"):])
                    target = Path(temporary) / original["candidate_id"] / f"seed_{item['seed']}" / "metrics.json"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(metrics, target)
                    compressed = metrics.with_name("metrics_steps.csv.gz")
                    with gzip.open(compressed, "rb") as src, target.with_name("metrics_steps.csv").open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    repeats.append(load_rollout_metrics(target, horizon=item["horizon"],
                                                        seed=item["seed"], metric_keys=spec.metric_keys))
                aggregate = aggregate_repeats(original["candidate_id"], repeats)
                # Check the actual raw telemetry, not only the archived decision label.
                generated = aggregate.as_dict()
                for key in ("zero_termination_repeats", "worst_first_failure_step", "total_terminations", "median_reward", "median_metrics"):
                    if generated[key] != original[key]:
                        raise ValueError(f"Telemetry aggregate changed: {record['id']} / {original['candidate_id']} / {key}")
                aggregates.append(aggregate)
            decision = decide_acceptance(aggregates[0], aggregates[1:], spec)
            expected = selection["decision"]
            if decision.accepted != expected["accepted"] or decision.selected_id != expected["selected_id"]:
                raise ValueError(f"Gate replay mismatch: {record['id']}")
            outcomes.append({"id": record["id"], "accepted": decision.accepted, "selected_id": decision.selected_id})
    return {"replayed": len(outcomes), "accepted": sum(row["accepted"] for row in outcomes), "outcomes": outcomes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("verify")
    check.add_argument("--root", type=Path, required=True)
    core = commands.add_parser("core-inputs")
    evaluate = commands.add_parser("prepare-eval")
    for command in (core, evaluate):
        command.add_argument("--policies", type=Path, required=True)
        command.add_argument("--assets", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--code", type=Path, required=True)
    evaluate.add_argument("--backend", type=Path, required=True)
    evaluate.add_argument("--runtime-python", type=Path, required=True)
    evaluate.add_argument("--trial", required=True)
    evaluate.add_argument("--external-motion", type=Path)
    evaluate.add_argument("--video", action="store_true", help="Prepare 1080p video and screenshot exports")
    gates = commands.add_parser("verify-gates")
    gates.add_argument("--assets", type=Path, required=True)
    gates.add_argument("--code", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        result = verify(args.root)
    elif args.command == "core-inputs":
        result = materialize_core(args.policies, args.assets, args.output)
    elif args.command == "prepare-eval":
        result = prepare_evaluation(args.policies, args.assets, args.code, args.backend,
                                    args.runtime_python, args.trial, args.output, args.external_motion, args.video)
    else:
        result = verify_gates(args.assets, args.code)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
