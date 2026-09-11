#!/usr/bin/env python3
"""Fetch a pinned dependency and apply its MimicX overlay to a clean checkout."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def setup(name, destination, *, source=None):
    spec = json.loads((ROOT / "dependencies/sources.json").read_text())["repositories"][name]
    destination = destination.expanduser().absolute()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}; choose a new directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--no-checkout", str(source or spec["url"]), str(destination)], check=True)
    subprocess.run(["git", "checkout", "--detach", spec["revision"]], cwd=destination, check=True)
    applied = []
    if "overlay" in spec:
        for path in sorted((ROOT / spec["overlay"]).rglob("*.py")):
            relative = path.relative_to(ROOT / spec["overlay"])
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            applied.append(str(relative))
    print(json.dumps({"repository": name, "revision": spec["revision"],
                      "checkout": str(destination), "overlay_files": applied}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=("unitree_rl_mjlab", "GVHMR", "GMR"))
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--source", type=Path, help="Optional existing Git clone to avoid downloading again")
    args = parser.parse_args()
    setup(args.name, args.destination or ROOT / "third_party" / args.name, source=args.source)


if __name__ == "__main__":
    main()
