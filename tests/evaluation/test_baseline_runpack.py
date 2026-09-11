from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from mimicx.evaluation.baseline_runpack import prepare_baseline_runpack


def make_repo(path: Path) -> str:
    path.mkdir(parents=True)
    (path / "runner.py").write_text("print('baseline')\n", encoding="utf-8")
    (path / "model.pt").write_bytes(b"weights")
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "fixture"], check=True)
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def write_manifest(path: Path, repo: Path, commit: str) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "schema": "mimicx.external-baselines.v1",
                "forbidden_path_tokens": ["unapproved-experimental-branch"],
                "baselines": [
                    {
                        "id": "ready-baseline",
                        "display_name": "Ready Baseline",
                        "repo_path": str(repo),
                        "expected_commit": commit,
                        "required_files": ["runner.py", "model.pt"],
                        "python": sys.executable,
                        "required_imports": ["json"],
                        "matched_protocol_ready": True,
                        "blockers": [],
                        "command": ["python", "runner.py", "--seed", "{seed}"],
                    },
                    {
                        "id": "staged-baseline",
                        "display_name": "Staged Baseline",
                        "repo_path": str(repo),
                        "expected_commit": commit,
                        "required_files": ["runner.py", "model.pt"],
                        "python": sys.executable,
                        "required_imports": ["mimicx_fixture_missing_module"],
                        "matched_protocol_ready": False,
                        "blockers": ["reference adapter pending"],
                        "command": ["python", "runner.py"],
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_prepare_baseline_runpack_records_provenance_and_guards_launch(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "upstream"
    commit = make_repo(repo)
    manifest = tmp_path / "baselines.yaml"
    output = tmp_path / "runpack"
    write_manifest(manifest, repo, commit)

    report = prepare_baseline_runpack(manifest, output)

    assert report.baselines == 2
    assert report.ready == 1
    assert report.staged == 1
    resolved = json.loads((output / "resolved_baselines.json").read_text())
    ready = next(item for item in resolved["baselines"] if item["id"] == "ready-baseline")
    assert ready["observed_commit"] == commit
    assert len(ready["required_files"][1]["sha256"]) == 64
    assert ready["runtime_probe"]["imports_ready"] is True
    staged = next(item for item in resolved["baselines"] if item["id"] == "staged-baseline")
    assert staged["runtime_probe"]["imports_ready"] is False
    assert "mimicx_fixture_missing_module" in staged["runtime_probe"]["missing_imports"]
    command = (output / "ready-baseline.sh").read_text(encoding="utf-8")
    assert "MIMICX_ALLOW_BASELINE_RUN" in command
    assert "CUDA_VISIBLE_DEVICES" in command
    assert "{seed}" in command
    assert not (output / "baseline.started").exists()


def test_prepare_baseline_runpack_rejects_forbidden_provenance_path(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "unapproved-experimental-branch" / "upstream"
    commit = make_repo(repo)
    manifest = tmp_path / "baselines.yaml"
    write_manifest(manifest, repo, commit)

    try:
        prepare_baseline_runpack(manifest, tmp_path / "runpack")
    except ValueError as error:
        message = str(error)
    else:
        raise AssertionError("forbidden provenance path was accepted")

    assert "forbidden provenance token" in message
