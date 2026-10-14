import hashlib
import json
from pathlib import Path

import pytest

from mimicx.runtime import demonstrations as release


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def snapshot(tmp_path):
    policies, assets = tmp_path/'Policies', tmp_path/'Assets'
    rel = 'releases/fixture'
    for directory in (policies/rel, assets/rel):
        directory.mkdir(parents=True)
    files = {'runtime/runner.py': b'raise RuntimeError("must not execute")',
             'tasks/tennis/motion.npz': b'reference'}
    records = []
    for name, content in files.items():
        path = assets/rel/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        records.append({'path': f'{rel}/{name}', 'sha256': digest(path), 'bytes': len(content)})
    (assets/rel/'FILES.json').write_text(json.dumps(records))
    checkpoint = policies/'checkpoints/weight.pt'
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b'weight')
    row = {'task': 'tennis', 'name': 'Tennis Swing', 'horizon': 100,
           'checkpoint': {'path': 'checkpoints/weight.pt', 'sha256': digest(checkpoint)},
           'motion': records[1], 'holdout_seeds': [4,5,6], 'runtime_kind': 'core_recovery'}
    manifest = {'schema':'mimicx.demonstrations.v2','assets_revision': 'b'*40,
                'policies': [row], 'controllers': [{'task': 'parkour'}]}
    path=policies/rel/'manifest.json'
    path.write_text(json.dumps(manifest))
    spec={'release':rel,'policies_revision':'a'*40,'assets_revision':'b'*40,
          'manifest_sha256':digest(path),'assets_index_sha256':digest(assets/rel/'FILES.json'),
          'dependency_indexes':{},'tasks':['tennis'],'runner':'runtime/runner.py'}
    return policies, assets, spec


def test_selective_checkpoint_validation_without_gpu(snapshot):
    policies, assets, spec = snapshot
    row, paths = release.verify_task(policies, assets, 'tennis', spec)
    assert row['horizon']==100
    assert paths['checkpoint'].read_bytes()==b'weight'
    assert paths['motion'].read_bytes()==b'reference'


def test_validate_every_runtime_byte(snapshot):
    policies, assets, spec = snapshot
    (assets/spec['release']/'runtime/runner.py').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        release.verify_task(policies, assets, 'tennis', spec)


def test_manifest_is_pinned_not_just_self_reported(snapshot):
    policies, assets, spec = snapshot
    (policies/spec['release']/'manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        release.verify_task(policies, assets, 'tennis', spec)


def test_assets_revision_must_match(snapshot):
    policies, assets, spec = snapshot
    spec=dict(spec, assets_revision='c'*40)
    with pytest.raises(ValueError, match='revision'):
        release.verify_task(policies, assets, 'tennis', spec)


def test_hybrid_is_not_an_actor_only_task(snapshot):
    policies, assets, spec = snapshot
    with pytest.raises(ValueError, match='controller'):
        release.verify_task(policies, assets, 'parkour', spec)


def test_duplicate_tasks_rejected(snapshot):
    policies, assets, spec=snapshot
    path=policies/spec['release']/'manifest.json'
    content=json.loads(path.read_text())
    content['policies']*=2
    path.write_text(json.dumps(content))
    spec=dict(spec,manifest_sha256=digest(path))
    with pytest.raises(ValueError, match='ambiguous'):
        release.verify_task(policies,assets,'tennis',spec)


def test_escaped_artifact_path_rejected(tmp_path):
    with pytest.raises(ValueError, match='escapes'):
        release.safe_path(tmp_path,'../other')
    with pytest.raises(ValueError, match='relative'):
        release.safe_path(tmp_path,str(tmp_path/'file'))


def test_prepare_command_never_executes(snapshot,tmp_path):
    policies,assets,spec=snapshot
    command=release.evaluation_command(policies,assets,'tennis',tmp_path/'run',
                                       spec=spec,python=Path('/usr/bin/python3'))
    assert command[0]=='/usr/bin/python3'
    assert '--seed' in command and command[command.index('--seed')+1]=='4'
    assert '--evaluate' in command
    assert not (tmp_path/'run').exists()


def test_existing_output_not_overwritten(snapshot,tmp_path):
    policies,assets,spec=snapshot
    with pytest.raises(FileExistsError):
        release.evaluation_command(policies,assets,'tennis',tmp_path,spec=spec)


def test_release_config_declares_nine_tasks():
    spec=release.load_spec()
    assert len(spec['tasks'])==9
    assert 'descent' in spec['tasks'] and 'parkour' not in spec['tasks']
    assert len(spec['manifest_sha256'])==64


def test_downloader_uses_pinned_revisions_and_selected_weight(snapshot, tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    policies, assets, spec = snapshot
    output = tmp_path/'download'
    output.mkdir()
    policies.rename(output/'MimicX-Policies')
    assets.rename(output/'MimicX-Assets')
    calls = []
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        snapshot_download=lambda repo, **kwargs: calls.append(dict(repo=repo, **kwargs))))
    result = release.download(output, ['tennis'], spec, 'test/policies', 'test/assets')
    assert len(calls)==3
    assert calls[0]['revision']==calls[1]['revision']==spec['policies_revision']
    assert calls[1]['allow_patterns']==['checkpoints/weight.pt']
    assert calls[2]['repo_type']=='dataset'
    assert calls[2]['revision']==spec['assets_revision']
    assert result['tasks']==['tennis']


def test_empty_download_task_list_rejected_before_network(snapshot,tmp_path):
    with pytest.raises(ValueError, match='task'):
        release.download(tmp_path, [], snapshot[2], 'test/policies', 'test/assets')
