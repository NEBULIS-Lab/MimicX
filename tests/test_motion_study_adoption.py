"""CPU-only provenance tests; Git repositories are disposable synthetic fixtures."""
import copy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from mimicx.experiments import motion_study as study
from mimicx.experiments.motion_study_adoption import build_asset_proof, check_backend_compatibility


@pytest.fixture
def legacy(tmp_path):
    backend = tmp_path / 'donor-backend'
    backend.mkdir()
    files = {'src/task.py': b'# fixed native task\n', 'scripts/train.py': b'# fixed trainer\n',
             'src/assets/g1.xml': b'<mujoco><asset><mesh file="mesh.stl"/></asset></mujoco>',
             'src/assets/mesh.stl': b'original tracked asset'}
    for rel, data in files.items():
        path = backend / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(backend), *args], stderr=subprocess.DEVNULL)
    git('init')
    git('add', '.')
    git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', '-c', 'commit.gpgsign=false',
        'commit', '-m', 'Synthetic immutable backend')
    commit = git('rev-parse', 'HEAD').decode().strip()
    proof_path = tmp_path / 'pinned-asset-proof.json'
    proof = build_asset_proof(backend, backend / 'src/assets/g1.xml', commit, proof_path)
    target = tmp_path / 'target-backend'
    shutil.copytree(backend, target)
    old_sources = {str(backend / rel): study.digest(backend / rel) for rel in files if rel != 'src/assets/mesh.stl'}
    new_sources = {str(target / rel): study.digest(target / rel) for rel in files}
    old = dict(runtime=dict(backend_root=str(backend)), sources=old_sources)
    new = dict(runtime=dict(backend_root=str(target), common_model_xml=str(target / 'src/assets/g1.xml'),
                           legacy_backend_asset_proof=str(proof_path), legacy_backend_asset_proof_sha256=proof['sha256'],
                           legacy_backend_asset_commit=commit), sources=new_sources)
    return old, new, proof_path


def test_legacy_expanded_asset_coverage_has_explicit_git_proof(legacy):
    old, new, _ = legacy
    audit = check_backend_compatibility(old, new)
    assert audit['mode'] == 'expanded_asset_coverage_with_pinned_git_proof'
    assert audit['newly_recorded_paths'] == ['src/assets/mesh.stl']
    assert audit['originally_recorded_hashes_match']
    assert audit['verified_asset_files'] == 2
    assert audit['both_current_backend_roots_verified']
    assert not audit['historical_donor_asset_hashes_claimed']
    assert 'not retroactively' in audit['scope']


def test_coverage_growth_without_proof_fails_before_donor_completion(legacy):
    old, new, _ = legacy
    for key in list(new['runtime']):
        if key.startswith('legacy_backend_asset_'):
            new['runtime'].pop(key)
    with pytest.raises(ValueError, match='explicit pinned legacy asset proof'):
        check_backend_compatibility(old, new)


@pytest.mark.parametrize('key', ['legacy_backend_asset_proof_sha256', 'legacy_backend_asset_commit'])
def test_missing_proof_pin_fails(legacy, key):
    old, new, _ = legacy
    new['runtime'].pop(key)
    with pytest.raises(ValueError, match='requires path'):
        check_backend_compatibility(old, new)


@pytest.mark.parametrize('which', ['donor', 'target'])
def test_changed_current_asset_is_not_ignored(legacy, which):
    old, new, _ = legacy
    root = old['runtime']['backend_root'] if which == 'donor' else new['runtime']['backend_root']
    (Path(root) / 'src/assets/mesh.stl').write_bytes(b'changed actual mesh')
    with pytest.raises(ValueError, match='differs from pinned Git blob'):
        check_backend_compatibility(old, new)


def test_changed_recorded_code_and_deleted_record_still_fail(legacy):
    old, new, _ = legacy
    code = str(Path(new['runtime']['backend_root']) / 'src/task.py')
    modified = copy.deepcopy(new)
    modified['sources'][code] = 'f' * 64
    with pytest.raises(ValueError, match='Donor native backend changed'):
        check_backend_compatibility(old, modified)
    new['sources'].pop(code)
    with pytest.raises(ValueError, match='Donor native backend changed'):
        check_backend_compatibility(old, new)


def test_added_code_is_not_misrepresented_as_asset_coverage(legacy):
    old, new, _ = legacy
    path = Path(new['runtime']['backend_root']) / 'src/extra.py'
    path.write_text('# additional implementation\n')
    new['sources'][str(path)] = study.digest(path)
    with pytest.raises(ValueError, match='not exclusively canonical asset'):
        check_backend_compatibility(old, new)


def test_tampered_proof_and_wrong_commit_rejected(legacy):
    old, new, proof_path = legacy
    wrong = copy.deepcopy(new)
    wrong['runtime']['legacy_backend_asset_commit'] = '0' * 40
    with pytest.raises(ValueError, match='Git commit mismatch'):
        check_backend_compatibility(old, wrong)
    proof_path.write_text(proof_path.read_text() + ' ')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        check_backend_compatibility(old, new)


def test_incomplete_proof_cannot_be_accepted_even_if_new_pin_supplied(legacy):
    old, new, proof_path = legacy
    data = study.read(proof_path)
    data['assets'].pop('src/assets/mesh.stl')
    study.save_json(proof_path, data)
    new['runtime']['legacy_backend_asset_proof_sha256'] = study.digest(proof_path)
    with pytest.raises(ValueError, match='exactly the canonical model closure'):
        check_backend_compatibility(old, new)


def test_git_proof_builder_rejects_dirty_assets(legacy, tmp_path):
    old, new, _ = legacy
    backend = Path(old['runtime']['backend_root'])
    (backend / 'src/assets/mesh.stl').write_bytes(b'dirty worktree asset')
    with pytest.raises(ValueError, match='differs from pinned Git blob'):
        build_asset_proof(backend, backend / 'src/assets/g1.xml', new['runtime']['legacy_backend_asset_commit'], tmp_path / 'dirty-proof.json')


def test_same_coverage_keeps_original_strict_mode_without_proof(legacy):
    old, new, _ = legacy
    new['sources'].pop(str(Path(new['runtime']['backend_root']) / 'src/assets/mesh.stl'))
    new['runtime'] = {'backend_root': new['runtime']['backend_root']}
    assert check_backend_compatibility(old, new)['mode'] == 'strict_identity'
