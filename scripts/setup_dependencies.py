#!/usr/bin/env python3
"""Fetch a pinned dependency and apply its MimicX overlay to a clean checkout."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def setup(name, destination, *, source=None, profile='paper'):
    spec = json.loads((ROOT / "dependencies/sources.json").read_text())["repositories"][name]
    overlays = [spec['overlay']] if 'overlay' in spec else []
    if profile != 'paper':
        if profile not in spec.get('profiles', {}):
            raise ValueError(f'No {profile} profile for {name}')
        overlays.extend(spec['profiles'][profile])
    destination = destination.expanduser().absolute()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}; choose a new directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--no-checkout", str(source or spec["url"]), str(destination)], check=True)
    subprocess.run(["git", "checkout", "--detach", spec["revision"]], cwd=destination, check=True)
    applied = []
    for overlay in overlays:
        for path in sorted((ROOT / overlay).rglob("*.py")):
            relative = path.relative_to(ROOT / overlay)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            applied.append(str(relative))
    print(json.dumps({"repository": name, "profile": profile, "revision": spec["revision"],
                      "checkout": str(destination), "overlay_files": applied}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=("unitree_rl_mjlab", "GVHMR", "GMR"))
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--source", type=Path, help="Optional existing Git clone to avoid downloading again")
    parser.add_argument('--profile', choices=('paper', 'demonstrations'), default='paper')
    args = parser.parse_args()
    default_name = args.name if args.profile == 'paper' else args.name + '_' + args.profile
    setup(args.name, args.destination or ROOT / "third_party" / default_name,
          source=args.source, profile=args.profile)


if __name__ == "__main__":
    main()
