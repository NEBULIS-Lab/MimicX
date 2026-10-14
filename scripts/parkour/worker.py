"""Run a configured collision-scene recovery training or evaluation stage."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mimicx.policy_recovery.worker import main


if __name__ == '__main__':
    main(kind='terrain')
