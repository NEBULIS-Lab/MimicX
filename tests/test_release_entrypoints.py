import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from mimicx.refinement.manifest import load_loop_manifest
from scripts.autorefine.launch_paper_matrix import expand_jobs, load_matrix
from tests.autorefine.test_closed_loop import write_loop_manifest

ROOT = Path(__file__).resolve().parents[1]


def run(script, *args):
    result = subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                            cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    return result


@pytest.mark.parametrize("script", [
    "scripts/create_loop_manifest.py", "scripts/create_comparison.py",
    "scripts/reconstruct_video.py", "scripts/runtime/run_hloop.py",
    "scripts/rollout_mjlab_motion_smoke.py", "scripts/export_mjlab_ghost_only_video.py",
    "scripts/setup_dependencies.py", "scripts/convert_gmr_to_mjlab_csv.py",
])
def test_entrypoint_help_is_cpu_only(script):
    assert "usage:" in run(script, "--help").stdout


def test_create_loop_manifest_for_new_task(tmp_path):
    backend = tmp_path / "backend"
    (backend / "scripts").mkdir(parents=True)
    (backend / "scripts/train.py").write_text("# fixture\n")
    checkpoint = backend / "logs/rsl_rl/g1_tracking/warmstart/model_10.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("fixture checkpoint")
    motion = tmp_path / "motion.npz"
    motion.write_text("fixture motion")
    output = tmp_path / "loop.yaml"
    run("scripts/create_loop_manifest.py", "--task-id", "fixture", "--backend", backend,
        "--motion-file", motion, "--checkpoint", checkpoint, "--horizon", "100", "--output", output)
    manifest = load_loop_manifest(output)
    assert manifest.task.load_run == "warmstart"
    assert manifest.runtime.python == Path(sys.executable).absolute()
    assert manifest.verification.guards["body_pos_error_max"].relative_tolerance == 0.2
    assert manifest.search.candidate_gpus == (0,)


def test_comparison_generator_expands_all_methods_and_seeds(tmp_path):
    manifest = write_loop_manifest(tmp_path, candidate_frontier=90, candidate_body=0.21)
    output = tmp_path / "comparison"
    run("scripts/create_comparison.py", "--loop-manifest", manifest, "--output-dir", output,
        "--failure-window", "20", "80", "--priority-bodies", "right_wrist_yaw_link")
    matrix = load_matrix(output / "matrix.yaml")
    jobs = expand_jobs(matrix, gpu_ids=(0,))
    assert len(jobs) == 12
    assert {job.mode for job in jobs} == {"static", "closed_loop"}
    assert len({job.motion_sha256 for job in jobs}) == 1


def test_numerical_release_matches_recorded_checksums():
    root = ROOT / "benchmarks"
    for record in json.loads((root / "provenance.json").read_text()):
        path = root / record["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["published_sha256"]
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == record["rows"]
        assert all(not value.startswith("/") for row in rows for value in row.values())


def test_backend_training_has_no_visibility_override():
    path = ROOT / "backend_overlay/unitree_rl_mjlab/scripts/train.py"
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            assert all("CUDA_VISIBLE_DEVICES" not in ast.unparse(target) for target in node.targets)
    assert "MIMICX_TRAIN_DEVICE_IDS" in path.read_text()
