"""Offline motion-study campaign entry point; no simulator imports for prepare/help."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mimicx.experiments.motion_study import main

if __name__ == '__main__':
    main()
