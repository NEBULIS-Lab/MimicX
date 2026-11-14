import builtins
from fnmatch import fnmatch
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

from mimicx.runtime import demonstrations as release


def record(root, relative, content):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {'path': relative, 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}


@pytest.fixture(params=['legacy', 'unified'])
def bundle(tmp_path, request):
    policies, assets = tmp_path / 'source-policies', tmp_path / 'source-assets'
    legacy = request.param == 'legacy'
    prefix = 'releases/fixture'
    runner = prefix + '/runtime/run_policy.py' if legacy else 'runtimes/demonstrations/run_policy.py'
    runner_record = record(assets, runner, b'raise RuntimeError("must not execute")')
    motion = record(assets, 'tasks/tennis/motion.npz', b'motion')
    checkpoint = record(policies, 'checkpoints/tennis.pt', b'checkpoint')
    index_path = prefix + '/FILES.json' if legacy else 'runtimes/FILES.json'
    index = record(assets, index_path, json.dumps([runner_record]).encode())
    dependency = record(assets, 'dependencies/FILES.json', json.dumps([motion]).encode())
    manifest_path = prefix + '/manifest.json' if legacy else 'manifest.json'
    manifest = {'schema': 'mimicx.demonstrations.v2' if legacy else 'mimicx.demonstrations.v3',
                'assets_revision': 'b' * 40,
                'policies': [{'task': 'tennis', 'checkpoint': checkpoint, 'motion': motion,
                              'holdout_seeds': [7], 'horizon': 100}]}
    manifest_record = record(policies, manifest_path, json.dumps(manifest).encode())
    spec = {'release': prefix, 'manifest_sha256': manifest_record['sha256'],
            'policies_revision': 'a' * 40, 'assets_revision': 'b' * 40,
            'policies_repository': 'test/policies', 'assets_repository': 'test/assets',
            'tasks': ['tennis']}
    if legacy:
        # The legacy dependency pattern must fetch its indexed data as well.
        motion_path = assets / motion['path']
        motion = record(assets, 'dependencies/motion.npz', motion_path.read_bytes())
        manifest['policies'][0]['motion'] = motion
        spec['manifest_sha256'] = record(policies, manifest_path, json.dumps(manifest).encode())['sha256']
        dependency = record(assets, dependency['path'], json.dumps([motion]).encode())
        spec.update(runner='runtime/run_policy.py', assets_index_sha256=index['sha256'],
                    dependency_indexes={dependency['path']: dependency['sha256']})
    else:
        spec.update(manifest_path=manifest_path, runner_path=runner,
                    asset_indexes={index_path: index['sha256']},
                    assets_patterns=['runtimes/*', 'tasks/*', 'asset_index.json'],
                    policy_metadata_patterns=['manifest.json', 'policy_index.json'])
    return policies, assets, spec, runner


def test_verify_and_runner_resolution(bundle, tmp_path):
    policies, assets, spec, runner = bundle
    row, paths = release.verify_task(policies, assets, 'tennis', spec)
    assert row['task'] == 'tennis'
    assert paths['checkpoint'].read_bytes() == b'checkpoint'
    command = release.evaluation_command(policies, assets, 'tennis', tmp_path / 'result', spec=spec)
    assert Path(command[1]) == (assets / runner).resolve()
    assert command[command.index('--seed') + 1] == '7'
    assert not (tmp_path / 'result').exists()


def test_offline_download_patterns_and_no_runtime_imports(bundle, tmp_path, monkeypatch):
    policies, assets, spec, _ = bundle
    calls = []

    def snapshot_download(repo, **kwargs):
        calls.append((repo, kwargs))
        source = assets if repo == 'test/assets' else policies
        for path in source.rglob('*'):
            relative = path.relative_to(source).as_posix()
            if path.is_file() and any(fnmatch(relative, p) for p in kwargs['allow_patterns']):
                destination = kwargs['local_dir'] / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split('.')[0] in {'torch', 'tensorflow', 'mujoco', 'isaaclab', 'isaacgym', 'warp'}:
            pytest.fail('Unexpected runtime import: ' + name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=snapshot_download))
    monkeypatch.setattr(builtins, '__import__', guarded_import)
    importlib.reload(release)
    result = release.download(tmp_path / 'download', ['tennis'], spec)
    assert result['tasks'] == ['tennis']
    assert len(calls) == 3
    assert calls[0][1]['allow_patterns'] == spec.get('policy_metadata_patterns', [spec['release'] + '/*'])
    assert calls[1][1]['allow_patterns'] == ['checkpoints/tennis.pt']
    assert calls[2][1]['allow_patterns'] == spec.get(
        'assets_patterns', [spec['release'] + '/*', 'dependencies/*'])
    assert calls[0][1]['revision'] == calls[1][1]['revision'] == spec['policies_revision']
    assert calls[2][1]['revision'] == spec['assets_revision']
    assert calls[2][1]['repo_type'] == 'dataset'


def test_explicit_indexes_still_check_runtime_bytes(bundle):
    policies, assets, spec, runner = bundle
    (assets / runner).write_bytes(b'tampered')
    with pytest.raises(ValueError, match='checksum'):
        release.verify_task(policies, assets, 'tennis', spec)


def test_asset_index_is_checksum_pinned(bundle):
    policies, assets, spec, _ = bundle
    relative = next(iter(spec['asset_indexes'])) if 'asset_indexes' in spec else spec['release'] + '/FILES.json'
    (assets / relative).write_text('[]')
    with pytest.raises(ValueError, match='checksum'):
        release.verify_task(policies, assets, 'tennis', spec)


def test_explicit_fields_override_legacy_fields(bundle, tmp_path):
    policies, assets, spec, runner = bundle
    if 'manifest_path' not in spec:
        spec['manifest_path'] = spec['release'] + '/manifest.json'
        spec['runner_path'] = runner
        spec['asset_indexes'] = {spec['release'] + '/FILES.json': spec['assets_index_sha256'],
                                 **spec['dependency_indexes']}
    spec.update(release='unused/legacy', runner='unused.py', assets_index_sha256='0' * 64,
                dependency_indexes={'unused/FILES.json': '0' * 64})
    command = release.evaluation_command(policies, assets, 'tennis', tmp_path / 'result', spec=spec)
    assert Path(command[1]) == (assets / runner).resolve()


@pytest.mark.parametrize('schema', ['mimicx.demonstrations.v2', 'mimicx.demonstrations.v3', 'unsupported'])
def test_manifest_schema_versions(bundle, schema):
    policies, assets, spec, _ = bundle
    relative = spec.get('manifest_path', spec['release'] + '/manifest.json')
    manifest = json.loads((policies / relative).read_text())
    manifest['schema'] = schema
    spec['manifest_sha256'] = record(policies, relative, json.dumps(manifest).encode())['sha256']
    if schema == 'unsupported':
        with pytest.raises(ValueError, match='schema'):
            release.verify_task(policies, assets, 'tennis', spec)
    else:
        release.verify_task(policies, assets, 'tennis', spec)


@pytest.mark.parametrize('field', ['manifest_path', 'runner_path', 'asset_indexes'])
def test_explicit_paths_cannot_escape_snapshot(bundle, tmp_path, field):
    policies, assets, spec, _ = bundle
    spec[field] = {'../outside.json': '0' * 64} if field == 'asset_indexes' else '../outside.py'
    with pytest.raises(ValueError, match='escapes'):
        release.evaluation_command(policies, assets, 'tennis', tmp_path / 'result', spec=spec)
