from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_closed_loop_cli_help_runs_from_project_root() -> None:
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "scripts/autorefine/run_closed_loop.py", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "{prepare,replay,execute}" in completed.stdout
