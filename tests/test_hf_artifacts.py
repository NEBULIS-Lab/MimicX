import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts.reproduction.hf_artifacts import (
    ARCHIVE_REVISION, archive_policy_index, materialize_core, prepare_evaluation,
    resolve_uri, safe_path, verify, verify_gates,
)


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


@pytest.mark.parametrize('schema', [None, 'mimicx.hf-policy-index.v1'])
def test_historical_index_compatibility(tmp_path, schema):
    index = {'trials': []}
    if schema is not None:
        index['schema'] = schema
    (tmp_path / 'policy_index.json').write_text(json.dumps(index))
    assert archive_policy_index(tmp_path) == index


@pytest.mark.parametrize('index', [
    {'schema': 'mimicx.policy-index.v2', 'policies': []},
    {'schema': 'mimicx.policy-index.v2', 'trials': []},
    {'schema': 'unknown', 'trials': []},
    {'trials': {}},
    [],
])
def test_nonhistorical_indexes_have_actionable_errors(tmp_path, index):
    (tmp_path / 'policy_index.json').write_text(json.dumps(index))
    with pytest.raises(ValueError, match=ARCHIVE_REVISION):
        archive_policy_index(tmp_path)


def test_missing_archive_inventory_has_actionable_error(tmp_path):
    with pytest.raises(ValueError, match=ARCHIVE_REVISION):
        verify(tmp_path)


@pytest.mark.parametrize('operation', ['verify', 'core-inputs', 'prepare-eval'])
def test_unified_policies_rejected_before_historical_resolution(tmp_path, operation):
    policies = tmp_path / 'policies'
    policies.mkdir()
    (policies / 'policy_index.json').write_text(json.dumps({'schema': 'mimicx.policy-index.v2'}))
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match=ARCHIVE_REVISION):
        if operation == 'verify':
            verify(policies)
        elif operation == 'core-inputs':
            materialize_core(policies, tmp_path / 'assets', output)
        else:
            prepare_evaluation(policies, tmp_path / 'assets', tmp_path / 'code',
                               tmp_path / 'backend', '/usr/bin/python3', 'old-trial', output)
    assert not output.exists()


@pytest.mark.parametrize('operation', ['verify', 'verify-gates', 'prepare-eval'])
def test_unified_assets_rejected_before_historical_resolution(tmp_path, operation):
    assets = tmp_path / 'assets'
    assets.mkdir()
    (assets / 'asset_index.json').write_text('{}')
    (tmp_path / 'policy_index.json').write_text(json.dumps({'trials': []}))
    with pytest.raises(ValueError, match=ARCHIVE_REVISION):
        if operation == 'verify':
            verify(assets)
        elif operation == 'verify-gates':
            verify_gates(assets, tmp_path / 'missing-code')
        else:
            prepare_evaluation(tmp_path, assets, tmp_path / 'code', tmp_path / 'backend',
                               '/usr/bin/python3', 'old-trial', tmp_path / 'output')


def test_archived_core_materialization_preserves_bytes(tmp_path):
    policies, assets = tmp_path / 'policies', tmp_path / 'assets'
    policies.mkdir()
    (assets / 'configs').mkdir(parents=True)
    (policies / 'policy_index.json').write_text(json.dumps({
        'schema': 'mimicx.hf-policy-index.v1', 'trials': []}))
    (policies / 'checkpoint.pt').write_bytes(b'frozen checkpoint')
    (assets / 'motion.npz').write_bytes(b'frozen motion')
    recipe = {'files': {'warmstarts/policy.pt': 'policies://checkpoint.pt',
                        'motions/reference.npz': 'assets://motion.npz'}}
    (assets / 'configs/core_input_map.json').write_text(json.dumps(recipe))
    (assets / 'configs/core_inputs.json').write_text('{"fixture": true}')
    for root in (policies, assets):
        records = [{'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size,
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                   for path in root.rglob('*') if path.is_file()]
        (root / 'FILES.json').write_text(json.dumps(records))
    output = tmp_path / 'core'
    report = materialize_core(policies, assets, output)
    assert report['materialized_assets'] == 2
    assert (output / 'warmstarts/policy.pt').read_bytes() == b'frozen checkpoint'
    assert (output / 'motions/reference.npz').read_bytes() == b'frozen motion'
    assert (output / 'manifest.json').read_bytes() == (assets / 'configs/core_inputs.json').read_bytes()


@pytest.mark.parametrize('document', ['REPRODUCIBILITY.md', 'HYBRID_CONTROL.md'])
def test_documented_downloads_are_explicitly_paired_offline(document, monkeypatch):
    from mimicx.runtime import demonstrations

    calls = []
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        snapshot_download=lambda repo, **kwargs: calls.append((repo, kwargs))))
    spec = {'policies_repository': 'test/policies', 'assets_repository': 'test/assets',
            'policies_revision': 'a' * 40, 'assets_revision': 'b' * 40}
    monkeypatch.setattr(demonstrations, 'load_spec', lambda: spec)
    text = (Path(__file__).resolve().parents[1] / 'docs' / document).read_text()
    snippet = text.split("python - <<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
    exec(compile(snippet, document, 'exec'), {})
    assert len(calls) == 2
    assert calls[1][1]['repo_type'] == 'dataset'
    if document == 'REPRODUCIBILITY.md':
        assert [repo for repo, _ in calls] == ['Shuaijun/MimicX-Policies', 'Shuaijun/MimicX-Assets']
        assert all(options['revision'] == ARCHIVE_REVISION for _, options in calls)
        assert all('allow_patterns' not in options for _, options in calls)
    else:
        assert [options['revision'] for _, options in calls] == ['a' * 40, 'b' * 40]
        assert 'controllers/parkour/*' in calls[0][1]['allow_patterns']
        assert 'runtimes/*' in calls[1][1]['allow_patterns']


@pytest.mark.parametrize('document', ['DEMONSTRATIONS.md', 'REPRODUCIBILITY.md', 'HYBRID_CONTROL.md'])
def test_documented_shell_syntax_without_execution(document):
    text = (Path(__file__).resolve().parents[1] / 'docs' / document).read_text()
    for snippet in re.findall(r'```bash\n(.*?)\n```', text, re.DOTALL):
        result = subprocess.run(['bash', '-n'], input=snippet, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
