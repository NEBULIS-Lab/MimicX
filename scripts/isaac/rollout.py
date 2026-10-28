#!/usr/bin/env python3
"""Evaluate a local Isaac checkpoint, preserving terminal physical states."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mimicx.backends.isaaclab.runner import main

if __name__ == "__main__":
    raise SystemExit(main("rollout"))
