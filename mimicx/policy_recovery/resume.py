"""Atomic checkpoints and source-compatible PPO block accounting."""
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


def remaining_updates(start, budget, saved_iteration):
    start, budget = int(start), int(budget)
    completed = 0 if saved_iteration is None else int(saved_iteration) - start + 1
    if budget <= 0 or completed < 0 or completed > budget or (saved_iteration is not None and completed == 0):
        raise ValueError('Checkpoint iteration is outside this training block')
    return dict(completed_updates=completed, remaining_updates=budget - completed,
                next_iteration=start + completed)


def load_checkpoint(path):
    """Read trusted local weights on CPU; importing this module does not load Torch."""
    import torch
    state = torch.load(path, map_location='cpu', weights_only=False)
    for key in ('actor_state_dict', 'critic_state_dict'):
        for value in state.get(key, {}).values():
            if torch.is_tensor(value) and not torch.isfinite(value).all().item():
                raise ValueError('Nonfinite checkpoint weights')
    return state


def checkpoint_plan(folder, start, budget):
    folder = Path(folder)
    candidates = sorted(
        (path for path in folder.glob('model_*.pt') if re.fullmatch(r'model_(\d+)\.pt', path.name)),
        key=lambda path: int(path.stem.split('_')[-1]), reverse=True)
    if (folder / 'final.pt').exists():
        candidates.insert(0, folder / 'final.pt')
    invalid = []
    for path in candidates:
        try:
            state = load_checkpoint(path)
            for key in ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict', 'iter'):
                if key not in state:
                    raise ValueError('Checkpoint missing ' + key)
            progress = remaining_updates(start, budget, state['iter'])
            if path.name == 'final.pt' and progress['remaining_updates']:
                raise ValueError('Final checkpoint does not complete this training block')
            if path.name != 'final.pt' and int(path.stem.split('_')[-1]) != state['iter']:
                raise ValueError('Checkpoint filename and saved iteration disagree')
            return dict(**progress, checkpoint=str(path), saved_iteration=state['iter'],
                        invalid_checkpoints=invalid)
        except Exception as exc:
            invalid.append(dict(path=str(path), error=repr(exc)))
    if candidates:
        raise ValueError('No valid resume checkpoint: ' + str(invalid))
    return dict(**remaining_updates(start, budget, None), checkpoint=None,
                saved_iteration=None, invalid_checkpoints=[])


def atomic_save(save_fn, path, infos=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.pending', dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        save_fn(str(temporary), infos=infos)
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_json(path, data):
    payload = json.dumps(data, indent=2, allow_nan=False) + '\n'
    def write(temporary, infos=None):
        Path(temporary).write_text(payload)
    atomic_save(write, path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()
