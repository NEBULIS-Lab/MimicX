#!/usr/bin/env python3
"""Train or continue Isaac tracking from local motion/checkpoint files."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mimicx.backends.isaaclab.runner import main

if __name__ == "__main__":
    raise SystemExit(main("train"))
