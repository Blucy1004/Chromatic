"""Atomic local checkpoints with previous-generation recovery and full RNG state."""
import hashlib
import os
from pathlib import Path
import random
import shutil
import numpy as np
import torch

FORMAT = 1


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def save(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    state = dict(state, format=FORMAT, rng=rng_state())
    with temp.open('wb') as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    # Only preserve a readable checkpoint. A failed recovery never overwrites .prev.
    if path.exists():
        try:
            _read(path)
        except Exception:
            pass
        else:
            previous = Path(str(path) + '.prev')
            previous_temp = Path(str(previous) + '.tmp')
            shutil.copyfile(path, previous_temp)
            os.replace(previous_temp, previous)
    os.replace(temp, path)


def _read(path):
    # Full optimizer + Python/NumPy RNG require weights_only=False. Local trusted files only.
    state = torch.load(path, map_location='cpu', weights_only=False)
    required = {'format', 'config', 'model', 'optimizer', 'rng', 'step', 'generator', 'dataset'}
    if not isinstance(state, dict) or not required <= state.keys() or state['format'] != FORMAT:
        raise ValueError('invalid checkpoint schema')
    if any(not torch.isfinite(t).all() for t in state['model'].values()):
        raise ValueError('nonfinite checkpoint weights')
    return state


def load(path):
    failures = []
    for candidate in (Path(path), Path(str(path) + '.prev')):
        try:
            return _read(candidate), candidate
        except Exception as exc:
            failures.append(f'{candidate}: {exc}')
    raise RuntimeError('no recoverable checkpoint: ' + '; '.join(failures))
