#!/usr/bin/env python3
"""Portable Parkour replay and actor+MPC (no new PPO training)."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from mimicx.parkour import hybrid
from mimicx.parkour.predictive import Planner, PlannerConfig, recover_nominal
from mimicx.parkour.servo import ActionAffine, actuator_names, load_actions, load_model, prepare_model, save_json


def read_json(path):
    return json.loads(Path(path).read_text())


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare-model", help="Compile native robot + scene on CPU")
    prep.add_argument("--scene", type=Path, required=True)
    prep.add_argument("--motion", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    summary = commands.add_parser("summarize", help="Aggregate six existing evaluation reports")
    summary.add_argument("--reports", type=Path, required=True)
    summary.add_argument("--index", type=int, default=2)
    summary.add_argument("--output", type=Path)
    for name in ("state-replay", "servo-replay", "plan", "recover-nominal", "actor-mpc"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument("--names", type=Path, required=True)
        cmd.add_argument("--motion", type=Path, required=name != "recover-nominal")
        if name != "actor-mpc":
            models = cmd.add_mutually_exclusive_group(required=True)
            models.add_argument("--model-xml", type=Path)
            models.add_argument("--model-mjb", type=Path)
            cmd.add_argument("--traces", type=Path, required=True)
            cmd.add_argument("--index", type=int, default=2)
        if name not in ("state-replay", "actor-mpc"):
            cmd.add_argument("--affine", type=Path, required=True)
        if name in ("plan", "actor-mpc"):
            cmd.add_argument("--config", type=Path, help="Strict PlannerConfig JSON; omitted fields use native defaults")
            cmd.add_argument("--nominal-actions", type=Path)
            cmd.add_argument("--nominal-index", type=int)
            cmd.add_argument("--nominal-sha256")
        if name in ("plan", "servo-replay"):
            cmd.add_argument("--initial-state", type=Path, help="Optional explicit qpos/qvel vectors at true start")
            cmd.add_argument("--steps", type=int, default=269)
        if name == "recover-nominal":
            cmd.add_argument("--planner", type=Path, required=True)
            cmd.add_argument("--start", type=int, default=190)
        if name == "actor-mpc":
            cmd.add_argument("--scene", type=Path, required=True)
            cmd.add_argument("--checkpoint", type=Path, required=True)
            cmd.add_argument("--device", default="cuda:0")
            cmd.add_argument("--seed", type=int, default=29003)
            cmd.add_argument("--six-resets", action="store_true")
            cmd.add_argument("--ppo-only", action="store_true", help="Frozen PPO baseline, never MPC")
    args = parser.parse_args(argv)
    if args.command == "prepare-model":
        prepare_model(args.scene, args.motion, args.output)
        return
    if args.command == "summarize":
        result = hybrid.evaluate_six_resets(lambda seed: hybrid.read_validation_report(
            args.reports / f"seed{seed}.json", args.index))
        if args.output:
            save_json(args.output, result)
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    names = read_json(args.names)
    cfg = PlannerConfig.from_dict(read_json(args.config)) if getattr(args, "config", None) else PlannerConfig()
    nominal = None
    if getattr(args, "nominal_actions", None):
        nominal = load_actions(args.nominal_actions, index=args.nominal_index, horizon=cfg.total_steps,
                               sha256=args.nominal_sha256)
    if args.command in ("plan", "actor-mpc") and cfg.nominal == "recorded" and nominal is None and not getattr(args, "ppo_only", False):
        parser.error("Recorded MPC requires --nominal-actions; use recover-nominal for the HF bundle")
    if args.command == "actor-mpc":
        def evaluate(seed, output):
            return hybrid.evaluate_actor(args.scene, args.motion, args.checkpoint, names, nominal, output,
                                         seed=seed, config=cfg, device=args.device, ppo_only=args.ppo_only)
        if args.six_resets:
            args.output.mkdir(parents=True, exist_ok=False)
            result = hybrid.evaluate_six_resets(lambda seed: evaluate(seed, args.output / f"seed{seed}"))
            save_json(args.output / "summary.json", result)
        else:
            evaluate(args.seed, args.output)
        return
    model = load_model(model_xml=args.model_xml, model_mjb=args.model_mjb)
    traces = read_npz(args.traces)
    if args.command == "state-replay":
        result = hybrid.replay_states(model, traces, names, read_npz(args.motion), index=args.index)
    else:
        affine = ActionAffine(**read_json(args.affine))
        actions = load_actions(args.traces, index=args.index, horizon=269)
        if args.command == "recover-nominal":
            recovered = recover_nominal(actions, read_json(args.planner), affine, actuator_names(model), start=args.start)
            args.output.mkdir(parents=True, exist_ok=False)
            np.savez_compressed(args.output / "nominal.npz", actions=recovered)
            save_json(args.output / "report.json", dict(execution="inverse_recorded_residual",
                usable_from_frame=args.start, prefix_is_not_original_nominal=True,
                byte_identical_to_original_prior=False, training_updates=0))
            return
        motion = read_npz(args.motion)
        if args.initial_state:
            initial = read_npz(args.initial_state)
            qpos, qvel = initial["qpos"], initial["qvel"]
        else:
            qpos, qvel = hybrid.initial_state(model, motion, names, traces, index=args.index)
        context = Planner(model, motion, names, affine, nominal, cfg) if args.command == "plan" else nullcontext(None)
        with context as planner:
            result = hybrid.run_cpu(model, motion, names, affine, actions, qpos, qvel, planner=planner, steps=args.steps)
    hybrid.write_result(args.output, result)
    print(json.dumps(result["report"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
