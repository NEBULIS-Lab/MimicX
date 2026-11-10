"""Read-only CPU collection of frozen confirmation receipts and traces."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mimicx.evaluation.confirmation_collect import main

if __name__ == '__main__':
    raise SystemExit(main())
