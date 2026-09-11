import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml

from scripts.autorefine.launch_paper_matrix import expand_jobs, load_matrix
from scripts.autorefine.run_native_hloop_benchmark import build_plan

ROOT = Path(__file__).resolve().parents[1]


def test_configure_and_materialize_actual_paper_contract_with_fixture_bytes(tmp_path):
    spec = json.loads((ROOT / "configs/paper/core_inputs.json").read_text())
    bundle = tmp_path / "assets"
    for row in spec["assets"]:
        path = bundle / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        content = ("fixture: " + row["path"]).encode()
        path.write_bytes(content)
        row["sha256"] = hashlib.sha256(content).hexdigest()
        row["bytes"] = len(content)
    manifest = tmp_path / "spec.json"
    manifest.write_text(json.dumps(spec))
    output = tmp_path / "config"
    backend_script = tmp_path / "backend/scripts/train.py"
    backend_script.parent.mkdir(parents=True)
    backend_script.write_text("# CPU fixture, never executed\n")
    subprocess.run([sys.executable, str(ROOT / "scripts/reproduction/configure_core.py"),
                    "--assets", str(bundle), "--backend", str(tmp_path / "backend"),
                    "--output", str(output), "--spec", str(manifest)], check=True)
    matrix = load_matrix(output / "matrix.yaml")
    jobs = expand_jobs(matrix, gpu_ids=(0, 1))
    assert len(jobs) == 48
    assert len([job for job in jobs if job.mode == "closed_loop"]) == 12
    plan = build_plan(yaml.safe_load((output / "hloop.yaml").read_text()))
    assert len(plan.jobs) == 17
    assert len([job for job in plan.jobs if job.resource == "gpu"]) == 8
    assert len([job for job in plan.jobs if job.resource == "cpu"]) == 8
    gpu_jobs = [job for job in plan.jobs if job.resource == "gpu"]
    assert {job.cuda_visible_devices for job in gpu_jobs} == {"0", "1"}
    assert all(job.budget["policy_updates"] == 0 for job in gpu_jobs)
    assert all("CUDA_VISIBLE_DEVICES" not in job.environment for job in plan.jobs)


def test_configuration_rejects_bad_input_before_writing(tmp_path):
    spec = {"assets": [{"path": "motion.npz", "sha256": "0" * 64}]}
    (tmp_path / "motion.npz").write_bytes(b"invalid")
    manifest = tmp_path / "spec.json"
    manifest.write_text(json.dumps(spec))
    output = tmp_path / "config"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/reproduction/configure_core.py"),
                             "--assets", str(tmp_path), "--output", str(output),
                             "--spec", str(manifest)], text=True, capture_output=True)
    assert result.returncode != 0
    assert "Hash mismatch" in result.stderr
    assert not output.exists()
