"""Explicit evidence bridge for legacy donors with narrower asset hash coverage."""
import hashlib
from pathlib import Path
import re
import subprocess

PROOF_SCHEMA = 'mimicx.legacy-backend-asset-proof.v1'
PROOF_KEYS = ('legacy_backend_asset_proof', 'legacy_backend_asset_proof_sha256',
              'legacy_backend_asset_commit')


def proof_config(runtime):
    from .motion_study import digest, read
    present = [bool(runtime.get(key)) for key in PROOF_KEYS]
    if not any(present):
        return None
    if not all(present):
        raise ValueError('Legacy asset proof requires path, pinned SHA-256 and pinned Git commit')
    path = Path(runtime[PROOF_KEYS[0]])
    if path.resolve().is_relative_to(Path(runtime['backend_root']).resolve()):
        raise ValueError('Keep the legacy asset proof outside backend_root')
    if not re.fullmatch(r'[0-9a-f]{40}', runtime[PROOF_KEYS[2]]):
        raise ValueError('Legacy asset Git commit must be a full SHA-1 commit ID')
    if digest(path) != runtime[PROOF_KEYS[1]]:
        raise ValueError('Legacy asset proof SHA-256 mismatch')
    proof = read(path)
    if proof.get('schema') != PROOF_SCHEMA or proof.get('git_commit') != runtime[PROOF_KEYS[2]]:
        raise ValueError('Legacy asset proof schema or Git commit mismatch')
    return proof


def build_asset_proof(backend_root, model_xml, commit, output):
    """Run only on the audited checkout; consumers need neither Git nor network."""
    from .motion_study import digest, immutable_json
    from .sonic_study import asset_files
    backend_root, model_xml, output = map(lambda p: Path(p).resolve(), (backend_root, model_xml, output))
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('Supply the independently approved full Git commit ID')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(backend_root), *args])
    if Path(git('rev-parse', '--show-toplevel').decode().strip()).resolve() != backend_root:
        raise ValueError('backend_root must be the audited Git checkout root')
    if git('rev-parse', '--verify', commit + '^{commit}').decode().strip() != commit:
        raise ValueError('Pinned Git commit is not available')
    assets = {}
    for path in sorted(asset_files(model_xml)):
        if not path.is_relative_to(backend_root):
            raise ValueError('Canonical asset escapes the audited backend checkout')
        rel = path.relative_to(backend_root).as_posix()
        records = git('ls-tree', '-z', commit, '--', rel).split(b'\0')
        records = [r for r in records if r]
        if len(records) != 1:
            raise ValueError(f'Asset is not uniquely tracked at the pinned commit: {rel}')
        header, tracked_path = records[0].split(b'\t', 1)
        mode, kind, blob = header.decode().split()
        if tracked_path.decode() != rel or kind != 'blob' or mode not in ('100644', '100755'):
            raise ValueError(f'Asset is not a regular tracked Git blob: {rel}')
        original = git('cat-file', 'blob', blob)
        sha = hashlib.sha256(original).hexdigest()
        if digest(path) != sha:
            raise ValueError(f'Current asset differs from pinned Git blob: {rel}')
        assets[rel] = dict(sha256=sha, git_blob=blob, bytes=len(original))
    proof = dict(schema=PROOF_SCHEMA, git_commit=commit,
        model_relative_path=model_xml.relative_to(backend_root).as_posix(), assets=assets,
        provenance='Pinned Git blobs checked against the audited checkout; not historical donor hash records',
        trust='Consumer must pin this independently audited proof SHA-256 and commit in its new campaign')
    immutable_json(output, proof)
    return dict(path=str(output), sha256=digest(output), git_commit=commit, asset_files=len(assets))


def backend_sources(campaign):
    root = Path(campaign['runtime']['backend_root'])
    return {Path(path).relative_to(root).as_posix(): sha for path, sha in campaign['sources'].items()
            if Path(path).is_relative_to(root)}


def check_backend_compatibility(source_campaign, campaign):
    """Check before donors finish; weights and GPU initialization are unnecessary."""
    from .motion_study import digest
    from .sonic_study import asset_files
    old, new = backend_sources(source_campaign), backend_sources(campaign)
    if not old or not new:
        raise ValueError('Donor/target backend identity records are missing')
    removed = sorted(old.keys() - new.keys())
    changed = sorted(key for key in old.keys() & new.keys() if old[key] != new[key])
    if removed or changed:
        raise ValueError(f'Donor native backend changed: removed={removed}, changed={changed}')
    added = sorted(new.keys() - old.keys())
    audit = dict(status='backend_compatible', originally_recorded_paths=len(old),
                 originally_recorded_hashes_match=True, newly_recorded_paths=added,
                 historical_donor_asset_hashes_claimed=False)
    if not added:
        return {**audit, 'mode': 'strict_identity'}
    runtime = campaign['runtime']
    proof = proof_config(runtime)
    if proof is None:
        raise ValueError('Expanded backend coverage requires an explicit pinned legacy asset proof; run check-adoption before waiting for donors')
    root, source_root = Path(runtime['backend_root']).resolve(), Path(source_campaign['runtime']['backend_root']).resolve()
    xml = Path(runtime['common_model_xml']).resolve()
    if not xml.is_relative_to(root):
        raise ValueError('Expanded legacy bridge requires a canonical model under backend_root')
    model_rel = xml.relative_to(root).as_posix()
    if model_rel not in old or proof['model_relative_path'] != model_rel:
        raise ValueError('Legacy bridge requires the same canonical XML already recorded by the donor')
    current_files = asset_files(xml)
    if any(not p.is_relative_to(root) for p in current_files):
        raise ValueError('Canonical asset closure escapes backend_root')
    closure = {p.relative_to(root).as_posix(): p for p in current_files}
    if not set(added).issubset(closure) or any(Path(p).suffix.lower() == '.py' for p in added):
        raise ValueError('New backend records are not exclusively canonical asset coverage')
    if set(proof['assets']) != set(closure):
        raise ValueError('Legacy asset proof does not cover exactly the canonical model closure')
    donor_files = asset_files(source_root / model_rel)
    if any(not p.is_relative_to(source_root) for p in donor_files):
        raise ValueError('Donor canonical asset closure escapes backend_root')
    source_closure = {p.relative_to(source_root).as_posix() for p in donor_files}
    if source_closure != set(closure):
        raise ValueError('Donor and current canonical asset closure differ')
    for rel, expected in proof['assets'].items():
        if new.get(rel) != expected['sha256'] or (rel in old and old[rel] != expected['sha256']):
            raise ValueError(f'Asset proof conflicts with recorded identity: {rel}')
        for base in {root, source_root}:
            data = (base / rel).read_bytes()
            blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            if (hashlib.sha256(data).hexdigest() != expected['sha256'] or blob != expected['git_blob']
                    or len(data) != expected['bytes']):
                raise ValueError(f'Legacy asset differs from pinned Git blob: {base / rel}')
    return {**audit, 'mode': 'expanded_asset_coverage_with_pinned_git_proof',
            'proof_sha256': runtime[PROOF_KEYS[1]], 'git_commit': proof['git_commit'],
            'verified_asset_files': len(closure), 'both_current_backend_roots_verified': True,
            'scope': 'New asset identities audited at adoption; not retroactively recorded by the legacy donor'}
