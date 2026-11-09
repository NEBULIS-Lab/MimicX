"""Public confirmation entry points remain separate from legacy protocols."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from mimicx.experiments import motion_study as study
from mimicx.experiments import motion_study_loop as loop
from tests.test_motion_study_campaign import campaign
from tests.test_motion_study_confirmation import inputs


def test_confirmation_release_has_explicit_versioned_freeze_api():
    name = "mimicx.experiments.motion_study_confirmation"
    assert importlib.util.find_spec(name) is not None, "Missing public confirmation freeze module"
    from mimicx.experiments.motion_study_confirmation import (
        CONFIRMATION_STRATEGY, HOLDOUT_SEEDS, PROTOCOL_VERSION,
        build_confirmation_contract, verify_confirmation_contract,
    )
    assert callable(build_confirmation_contract) and callable(verify_confirmation_contract)
    assert study.CONFIRMATION_STRATEGY == CONFIRMATION_STRATEGY
    protocol = loop.validation_protocol({"holdout": {"confirmation": HOLDOUT_SEEDS}},
                                        strategy=CONFIRMATION_STRATEGY)
    assert protocol["protocol_version"] == PROTOCOL_VERSION
    assert len(protocol["holdout"]["confirmation"]["nominal"]) == 3
    assert len(loop.validation_protocol()["holdout"]["confirmation"]["nominal"]) == 10


def test_documented_freeze_example_prepares_without_runtime_imports(inputs, tmp_path):
    root = Path(__file__).resolve().parents[1]
    text = (root / "docs/CONFIRMATION.md").read_text()
    runtime_example = json.loads(text.split("```json\n", 1)[1].split("```", 1)[0])
    manifest, runtime_path = inputs
    runtime = study.read(runtime_path)
    for name in ("backend_root", "python", "common_model_xml", "device", "num_envs"):
        runtime_example[name] = runtime[name]
    study.save_json(runtime_path, runtime_example)
    blocks = [part.split("```", 1)[0] for part in text.split("```bash\n")[1:]]
    for block in blocks:
        subprocess.run(["bash", "-n"], input=block, text=True, check=True, capture_output=True)
    script = blocks[0].split("<<'PY'\n", 1)[1].split("\nPY", 1)[0]
    script += '\nassert not {"torch", "mjlab", "mujoco", "isaaclab", "isaacsim"}.intersection(sys.modules)\n'
    freeze_path, output = tmp_path / "documented-freeze.json", tmp_path / "documented-campaign"
    command = [sys.executable, "-", str(manifest), str(runtime_path), str(freeze_path), str(output)]
    result = subprocess.run(command, input=script, text=True, capture_output=True, cwd=root)
    assert result.returncode == 0, result.stderr
    assert "Prepared and verified 96 jobs; none executed" in result.stdout
    data = study.read(output / "campaign.json")
    study.verify_campaign(data)
    assert len(data["jobs"]) == 96 and sum(job["updates"] for job in data["jobs"]) == 288000
    assert not list(output.rglob("*.pt")) and not list(output.rglob("done.json"))
    before = freeze_path.read_bytes()
    repeated = subprocess.run(command, input=script, text=True, capture_output=True, cwd=root)
    assert repeated.returncode != 0 and "Use new freeze and campaign paths" in repeated.stderr
    assert freeze_path.read_bytes() == before
