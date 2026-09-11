"""CPU-only checks of release process placement and portable entry points."""

import json
import os
from pathlib import Path
import subprocess
import sys

from mimicx.refinement.executor import CommandRecord, run_command


def test_executor_preserves_inherited_device_visibility(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-fixture-a,GPU-fixture-b")
    output = tmp_path / "environment.json"
    record = CommandRecord(
        id="placement-fixture",
        argv=(sys.executable, "-c", "import os,json,pathlib; "
              f"pathlib.Path({str(output)!r}).write_text(json.dumps(dict(os.environ)))"),
        cwd=tmp_path,
        log_path=tmp_path / "fixture.log",
        expected_outputs=(output,),
        requires_gpu=True,
    )
    result = run_command(record, gpu_id=1)
    environment = json.loads(output.read_text())
    assert result.exit_code == 0
    assert environment["CUDA_VISIBLE_DEVICES"] == os.environ["CUDA_VISIBLE_DEVICES"]
    assert environment["MIMICX_DEVICE_ID"] == "1"


def test_retarget_help_without_model_or_gpu_dependencies():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/retarget_gvhmr_to_gmr.py", "--help"],
        cwd=root, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--body-models" in result.stdout
