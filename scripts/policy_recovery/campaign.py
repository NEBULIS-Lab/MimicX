"""Unattended train-diagnose-select loop with protected incumbent and resumable stages."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mimicx.policy_recovery.metrics import summarize, accept, qualifies, failure_window
from mimicx.policy_recovery.worker import load_config, runtime_paths


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    args.config = args.config.expanduser().resolve()
    config = load_config(args.config)
    paths = runtime_paths(config)
    out = Path(config['output'])
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for key in ('checkpoint', 'motion'):
        if config.get(key + '_sha256') and digest(config[key]) != config[key + '_sha256']:
            raise RuntimeError(f'Frozen {key} changed')
    source_hashes = config.get('source_hashes')
    if not source_hashes and config.get('snapshot'):
        source_hashes = str(Path(config['snapshot']) / 'source_hashes.json')
    if source_hashes:
        manifest = Path(source_hashes)
        for rel, expected in json.loads(manifest.read_text()).items():
            source = (manifest.parent / rel).resolve()
            if not source.is_relative_to(manifest.parent.resolve()):
                raise ValueError(f'Source path escapes manifest directory: {rel}')
            if digest(source) != expected:
                raise RuntimeError(f'Frozen source changed: {rel}')
    state = dict(status='running', task=config['task'], route=config['route'],
                 started=datetime.now(timezone.utc).isoformat())
    def status(**updates):
        state.update(updates)
        state['updated'] = datetime.now(timezone.utc).isoformat()
        save(out / 'status.json', state)
        print(json.dumps(state), flush=True)
    worker = ROOT / 'scripts' / 'policy_recovery' / 'worker.py'
    def stage(folder, mode, checkpoint, *, window=None, block=0, seed=1001, noise=.005, video=False, hide=False):
        folder.mkdir(parents=True, exist_ok=True)
        worker_output = folder / 'rollout' if mode == 'evaluate' else folder
        cmd = [sys.executable, str(worker), '--config', str(args.config), '--mode', mode,
               '--checkpoint', str(checkpoint), '--output', str(worker_output), '--block', str(block),
               '--seed', str(seed), '--noise', str(noise)]
        if window:
            cmd += ['--window', *map(str, window)]
        if video:
            cmd.append('--video')
        if hide:
            cmd.append('--no-ghost')
        signature = dict(command=cmd, config_sha256=digest(args.config), checkpoint_sha256=digest(checkpoint),
                         inputs_sha256={key: digest(config[key]) for key in ('motion', 'scene', 'profile')
                                        if config.get(key)})
        if source_hashes:
            signature['source_manifest_sha256'] = digest(source_hashes)
        done = folder / 'done.json'
        if done.exists():
            receipt = json.loads(done.read_text())
            if receipt['signature'] != signature:
                raise RuntimeError(f'Cannot reuse mismatched stage: {folder}')
            for name, sha in receipt['artifacts'].items():
                if digest(folder / name) != sha:
                    raise RuntimeError(f'Corrupt stage artifact: {folder / name}')
            return
        if mode == 'evaluate' and worker_output.exists():
            worker_output.rename(folder / f'incomplete_rollout_{time.time_ns()}')
        status(stage=str(folder.relative_to(out)), mode=mode)
        save(folder / 'command.json', signature)
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED='1', WANDB_MODE='disabled', OMP_NUM_THREADS='4')
        env.setdefault('MUJOCO_GL', 'osmesa')
        with (folder / 'run.log').open('a') as log:
            subprocess.run(cmd, env=env, cwd=paths['backend'], stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        names = ['final.pt', 'effective_training.json'] if mode == 'train' else ['rollout/steps.csv', 'rollout/metrics.json']
        if video:
            names += ['rollout/policy.mp4']
        save(done, dict(signature=signature, artifacts={name: digest(folder / name) for name in names}))
    def evaluate(folder, checkpoint, holdout=False):
        seeds = config['holdout_seeds'] if holdout else config['selection_seeds']
        repeats = []
        for seed in seeds:
            run = folder / f'seed{seed}'
            stage(run, 'evaluate', checkpoint, seed=seed, noise=.01 if holdout else .005)
            with (run / 'rollout' / 'steps.csv').open() as stream:
                repeats.append(list(csv.DictReader(stream)))
        summaries = [summarize(rows, config['horizon']) for rows in repeats]
        save(folder / 'summary.json', dict(checkpoint=str(checkpoint), results=summaries,
                                          qualified=qualifies(summaries)))
        return repeats
    try:
        incumbent = latest = Path(config['checkpoint'])
        best_rows = latest_rows = evaluate(out / 'baseline', incumbent)
        history = []
        for block in range(config['blocks']):
            window = failure_window(latest_rows, config['horizon'])
            folder = out / f'block{block:02d}'
            stage(folder / 'train', 'train', latest, window=window, block=block)
            latest = folder / 'train' / 'final.pt'
            latest_rows = evaluate(folder / 'evaluation', latest)
            gate = accept(best_rows, latest_rows, config['horizon'])
            gate.update(block=block, window=window, candidate=str(latest), incumbent=str(incumbent))
            if gate['accepted']:
                incumbent, best_rows = latest, latest_rows
            save(folder / 'gate.json', gate)
            history.append(gate)
            save(out / 'selection.json', dict(checkpoint=str(incumbent), sha256=digest(incumbent), history=history))
            summaries = [summarize(r, config['horizon']) for r in best_rows]
            status(completed_blocks=block + 1, incumbent=str(incumbent), best=summaries)
            if qualifies(summaries):
                held = evaluate(out / f'holdout_block{block:02d}', incumbent, holdout=True)
                if qualifies([summarize(r, config['horizon']) for r in held]):
                    break
        held = evaluate(out / 'final_holdout', incumbent, holdout=True)
        selected = [summarize(r, config['horizon']) for r in best_rows]
        held_results = [summarize(r, config['horizon']) for r in held]
        qualified = qualifies(selected) and qualifies(held_results)
        render_results = []
        for hide in (False, True):
            label = 'no_ghost' if hide else 'with_ghost'
            folder = out / 'renders' / label
            stage(folder, 'evaluate', incumbent, noise=.005, video=True, hide=hide)
            with (folder / 'rollout' / 'steps.csv').open() as stream:
                render_results.append(summarize(list(csv.DictReader(stream)), config['horizon']))
        qualified = qualified and all(r['complete'] for r in render_results)
        report = dict(qualified=qualified, visual_review_required=True, checkpoint=str(incumbent),
                      checkpoint_sha256=digest(incumbent), selection=selected, holdout=held_results,
                      renders=render_results, horizon=config['horizon'],
                      completed_updates=len(history) * config['updates_per_block'])
        save(out / 'final_report.json', report)
        flat = Path(config.get('campaign', out)) / ('qualified_media' if qualified else 'diagnostic_media')
        flat.mkdir(exist_ok=True)
        prefix = config['task'].upper() + '__' + config['route'].upper()
        for label in ('with_ghost', 'no_ghost'):
            folder = out / 'renders' / label / 'rollout'
            shutil.copy2(folder / 'policy.mp4', flat / f'{prefix}__{label.upper()}__1080P.mp4')
            for frame in sorted((folder / 'frames').glob('*.png')):
                shutil.copy2(frame, flat / f'{prefix}__{label.upper()}__{frame.name}')
        save(flat / f'{prefix}__REPORT.json', report)
        status(status='finished', qualified=qualified, completed_updates=report['completed_updates'])
    except Exception as exc:
        status(status='failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
