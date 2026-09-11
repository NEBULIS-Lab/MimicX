#!/usr/bin/env python3
"""Audit the staged source release for generated artifacts and credentials."""

import ast
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".npz", ".pkl", ".mp4", ".pdf", ".png", ".jpg", ".pyc"}
FORBIDDEN_ROOTS = {"third_party", ".venv", "runs", "data", "logs", "checkpoints", "dist", "build"}


def main():
    raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    files = [Path(item.decode()) for item in raw.split(b"\0") if item]
    if not files:
        raise RuntimeError("Stage source files before running the release audit")
    failures = []
    total = 0
    for relative in files:
        path = ROOT / relative
        if path.is_symlink() or relative.parts[0] in FORBIDDEN_ROOTS or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"Excluded artifact: {relative}")
            continue
        content = path.read_bytes()
        total += len(content)
        if len(content) > 10_000_000:
            failures.append(f"Large file: {relative}")
        if path.suffix.lower() == ".stl":
            continue
        text = content.decode("utf-8")
        if re.search(r"(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)", text):
            failures.append(f"Credential-like content: {relative}")
        if re.search(r"/(?:data/(?:private|shared)|home)/[A-Za-z0-9_.-]+/", text):
            failures.append(f"Private filesystem path: {relative}")
        if path.suffix == ".py":
            ast.parse(text, filename=str(relative))
    print(json.dumps({"files": len(files), "bytes": total, "failures": failures}, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
