"""Simulate, fit or audit a numerical ball replay, without graphics or policy inference."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mimicx.visualization.contact_replay import BallConfig, BallReplay, audit_replay, fit_initial_conditions, group_contacts


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_engine(config_path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_path = config_path.parent / config["model"]
    robot_path = config_path.parent / config["robot"]
    root = ET.parse(model_path).getroot()
    if root.findall(".//include"):
        raise ValueError("Use a flattened MJCF; include expansion is not supported by this CLI")
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    for attr in ("meshdir", "texturedir"):
        compiler.set(attr, str((model_path.parent / compiler.get(attr, ".")).resolve()))
    physics = BallConfig(**config.get("physics", {}))
    with np.load(robot_path, allow_pickle=False) as robot:
        engine = BallReplay(ET.tostring(root, encoding="unicode"), robot["timestamps"], robot["robot_root_pose"],
                            robot["robot_joint_positions"], robot["joint_names"].tolist(), config=physics)
    return engine, config, robot_path, model_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("simulate", "fit"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("audit")
    p.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit":
        receipt = json.loads((args.input / "manifest.json").read_text())
        for filename, expected in receipt["artifacts"].items():
            if filename not in {"ball.npz", "contacts.json"} or sha256(args.input / filename) != expected:
                raise ValueError("Replay artifact hash mismatch")
        if set(receipt["artifacts"]) != {"ball.npz", "contacts.json"}:
            raise ValueError("Incomplete replay receipt")
        with np.load(args.input / "ball.npz", allow_pickle=False) as raw:
            out = dict(raw)
        out["contacts"] = json.loads((args.input / "contacts.json").read_text())
        result = audit_replay(out, mass=receipt["physics"]["mass"], gravity=receipt["physics"]["gravity"])
    else:
        engine, config, robot_path, model_path = load_engine(args.config)
        interval = {key: config[key] for key in ("start", "end") if key in config}
        if args.command == "fit":
            search = dict(config["fit"])
            observations = args.config.parent / search.pop("observations")
            with np.load(observations, allow_pickle=False) as raw:
                result = fit_initial_conditions(engine, raw["time"], raw["position"], **search, **interval)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, result)
        else:
            out = engine.simulate(config["initial"], **interval)
            result = audit_replay(out, mass=engine.config.mass, gravity=engine.config.gravity)
            args.output.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.output / "ball.npz", **{k: v for k, v in out.items() if isinstance(v, np.ndarray)})
            write_json(args.output / "contacts.json", out["contacts"])
            receipt = dict(schema="mimicx.contact-replay.v1", policy_evaluation=False,
                           robot_input_sha256=sha256(robot_path), model_input_sha256=sha256(model_path),
                           configuration_sha256=sha256(args.config), physics=asdict(engine.config),
                           initial=config["initial"], interval=[float(out["time"][0]), float(out["time"][-1])],
                           initial_contacts=out["initial_contacts"], audit=result,
                           contact_groups=group_contacts(out["contacts"], dt=engine.config.dt),
                           artifacts={name: sha256(args.output / name) for name in ("ball.npz", "contacts.json")})
            write_json(args.output / "manifest.json", receipt)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
