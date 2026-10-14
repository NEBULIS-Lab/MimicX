import hashlib
import json
from pathlib import Path

import pytest

from scripts.reproduction.hf_artifacts import prepare_evaluation, resolve_uri, safe_path, verify


def test_snapshot_verification_detects_tampering(tmp_path):
    path = tmp_path / "weight.pt"
    path.write_bytes(b"fixture")
    (tmp_path / "FILES.json").write_text(json.dumps([{"path": path.name, "bytes": 7,
        "sha256": hashlib.sha256(b"fixture").hexdigest()}]))
    assert verify(tmp_path) == {"files": 1, "bytes": 7}
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed artifact"):
        verify(tmp_path)


def test_snapshot_paths_cannot_escape_the_download_directory(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        safe_path(tmp_path, "../outside")
    with pytest.raises(ValueError, match="Not a bundled"):
        resolve_uri("external://reference", tmp_path, tmp_path)
    assert resolve_uri("assets://motions/reference.npz", tmp_path / "policies", tmp_path / "assets") == tmp_path / "assets/motions/reference.npz"


def test_evaluation_preparation_preserves_patch_and_does_not_execute(tmp_path):
    policies, assets, code, backend = [tmp_path / name for name in ("policies", "assets", "code", "backend")]
    for root in (policies, assets, code, backend):
        root.mkdir()
    (policies / "model.pt").write_bytes(b"fixture checkpoint")
    (assets / "motion.npz").write_bytes(b"fixture reference")
    (assets / "patch.json").write_text('{"fixture":true}')
    for root, script in ((code, "rollout_mjlab_motion_smoke.py"), (backend, "train.py")):
        (root / "scripts").mkdir()
        (root / "scripts" / script).write_text('raise RuntimeError("must never execute")')
    trial = {"id": "fixture", "checkpoint": "policies://model.pt",
             "checkpoint_sha256": hashlib.sha256(b"fixture checkpoint").hexdigest(),
             "reference": "assets://motion.npz",
             "reference_sha256": hashlib.sha256(b"fixture reference").hexdigest(),
             "evaluation_patch": "assets://patch.json", "verification_seeds": [1, 2, 3],
             "evaluation_task": "Fixture-Task", "horizon": 100}
    (policies / "policy_index.json").write_text(json.dumps({"trials": [trial]}))
    output = tmp_path / "prepared"
    report = prepare_evaluation(policies, assets, code, backend, Path("/usr/bin/python3"),
                                "fixture", output, video=True)
    assert report["executed"] is False and report["commands"] == 3
    script = (output / "evaluate.sh").read_text()
    assert "unset MIMICX_AUTOREFINE_PATCH_FILE" in script
    assert f"export MIMICX_AUTOREFINE_PATCH_FILE={assets / 'patch.json'}" in script
    assert "--video-width 1920 --video-height 1080" in script
