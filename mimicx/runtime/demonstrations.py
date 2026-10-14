"""Immutable paired policy downloads and evaluation, separate from paper trials."""
from __future__ import annotations

import argparse
from importlib.resources import files
import json
from pathlib import Path
import subprocess
import sys

from mimicx.refinement.manifest import hash_file


def load_spec(path: Path | None = None) -> dict:
    source = path or files('mimicx').joinpath('configs/releases/demonstrations.json')
    return json.loads(source.read_text())


def safe_path(root: Path, relative: str) -> Path:
    root = Path(root).resolve()
    if Path(relative).is_absolute():
        raise ValueError('Artifact names must be relative')
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Artifact path escapes the snapshot')
    return path


def _check(root: Path, record: dict) -> Path:
    path = safe_path(root, record['path'])
    if not path.is_file() or hash_file(path) != record['sha256']:
        raise ValueError('Missing file or checksum mismatch: ' + record['path'])
    if 'bytes' in record and path.stat().st_size != record['bytes']:
        raise ValueError('Artifact size mismatch: ' + record['path'])
    return path


def _manifest(policies: Path, spec: dict) -> dict:
    path = _check(policies, {'path': spec['release']+'/manifest.json',
                             'sha256': spec['manifest_sha256']})
    manifest = json.loads(path.read_text())
    if manifest.get('schema') != 'mimicx.demonstrations.v2':
        raise ValueError('Unsupported demonstration manifest schema')
    if manifest.get('assets_revision') != spec['assets_revision']:
        raise ValueError('The Policies and Assets revision pair does not match')
    return manifest


def _task(manifest: dict, task: str) -> dict:
    matches = [r for r in manifest['policies'] if r['task'] == task]
    if len(matches) != 1:
        if any(r['task'] == task for r in manifest.get('controllers', [])):
            raise ValueError('This is a hybrid controller, not an actor-only policy; see HYBRID_CONTROL.md')
        raise ValueError('Unknown or ambiguous policy task: ' + task)
    return matches[0]


def verify_task(policies: Path, assets: Path, task: str, spec: dict | None = None):
    """Check selected weights and all bundled runtime bytes without importing a simulator."""
    spec = spec or load_spec()
    row = _task(_manifest(policies, spec), task)
    indexes = {spec['release']+'/FILES.json': spec['assets_index_sha256'],
               **spec.get('dependency_indexes', {})}
    for relative, expected in indexes.items():
        index = _check(assets, {'path': relative, 'sha256': expected})
        for record in json.loads(index.read_text()):
            _check(assets, record)
    paths = {}
    for key in ('checkpoint', 'motion', 'scene', 'profile'):
        if key in row:
            paths[key] = _check(policies if key == 'checkpoint' else assets, row[key])
    return row, paths


def download(output: Path, tasks: list[str], spec: dict | None = None,
             policies_repository: str | None = None, assets_repository: str | None = None) -> dict:
    """Download chosen weights and their immutable companion runtime; never execute code."""
    if not tasks:
        raise ValueError('Select at least one task')
    spec = spec or load_spec()
    policy_repo = policies_repository or spec.get('policies_repository')
    asset_repo = assets_repository or spec.get('assets_repository')
    if not policy_repo or not asset_repo:
        raise ValueError('Provide --policies-repository and --assets-repository for this release')
    from huggingface_hub import snapshot_download
    policies, assets = Path(output)/'MimicX-Policies', Path(output)/'MimicX-Assets'
    snapshot_download(policy_repo, revision=spec['policies_revision'],
                      local_dir=policies, allow_patterns=[spec['release']+'/*'])
    manifest = _manifest(policies, spec)
    chosen = [_task(manifest, name) for name in tasks]
    snapshot_download(policy_repo, revision=spec['policies_revision'], local_dir=policies,
                      allow_patterns=[r['checkpoint']['path'] for r in chosen])
    patterns = [spec['release']+'/*']
    patterns += [str(Path(p).parent)+'/*' for p in spec.get('dependency_indexes', {})]
    snapshot_download(asset_repo, repo_type='dataset', revision=spec['assets_revision'],
                      local_dir=assets, allow_patterns=patterns)
    for task in tasks:
        verify_task(policies, assets, task, spec)
    return {'tasks': tasks, 'policies': str(policies), 'assets': str(assets),
            'policies_revision': spec['policies_revision'], 'assets_revision': spec['assets_revision']}


def evaluation_command(policies: Path, assets: Path, task: str, output: Path, *,
                       spec: dict | None = None, python: Path | None = None,
                       seed: int | None = None, device: str = 'cuda:0', video: bool = False) -> list[str]:
    spec = spec or load_spec()
    row, _ = verify_task(policies, assets, task, spec)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    runner = safe_path(assets, spec['release']+'/'+spec['runner'])
    command = [str(python or sys.executable), str(runner), '--policies', str(Path(policies).resolve()),
               '--assets', str(Path(assets).resolve()), '--task', task, '--evaluate',
               '--seed', str(row['holdout_seeds'][0] if seed is None else seed),
               '--device', device, '--output', str(output)]
    if video:
        command.append('--video')
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-spec', type=Path, help='Optional explicitly selected release definition')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list')
    fetch = commands.add_parser('download')
    fetch.add_argument('--task', action='append', help='Repeat for multiple tasks; default: all nine')
    fetch.add_argument('--output', type=Path, required=True)
    fetch.add_argument('--policies-repository')
    fetch.add_argument('--assets-repository')
    check = commands.add_parser('verify')
    evaluate = commands.add_parser('evaluate')
    for child in (check, evaluate):
        child.add_argument('--policies', type=Path, required=True)
        child.add_argument('--assets', type=Path, required=True)
        child.add_argument('--task', required=True)
    evaluate.add_argument('--output', type=Path, required=True)
    evaluate.add_argument('--runtime-python', type=Path)
    evaluate.add_argument('--seed', type=int)
    evaluate.add_argument('--device', default='cuda:0')
    evaluate.add_argument('--video', action='store_true')
    evaluate.add_argument('--dry-run', action='store_true', help='Validate and print the command without GPU execution')
    args = parser.parse_args()
    spec = load_spec(args.release_spec)
    if args.command == 'list':
        result = {'release': spec['release'], 'policies': spec['tasks'],
                  'separate_controller': 'parkour', 'paper_archive_replaced': False}
    elif args.command == 'download':
        result = download(args.output, args.task or spec['tasks'], spec,
                          args.policies_repository, args.assets_repository)
    elif args.command == 'verify':
        row, paths = verify_task(args.policies, args.assets, args.task, spec)
        result = {'task': row['task'], 'verified': True, 'horizon': row['horizon'],
                  'artifacts': {key: str(value) for key, value in paths.items()}}
    else:
        command = evaluation_command(args.policies, args.assets, args.task, args.output, spec=spec,
                                     python=args.runtime_python, seed=args.seed, device=args.device, video=args.video)
        result = {'command': command, 'executed': not args.dry_run}
        if not args.dry_run:
            # Inherit the caller's allocation and device visibility unchanged.
            subprocess.run(command, check=True)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
