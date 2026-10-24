from pathlib import Path
import random
import tempfile
import chess
import numpy as np
import torch
from .checkpoint import digest
from .data import atomic_json
from .encoding import action, features, outcome
from .model import load_model
from .native import Native


def verify(executable, checkpoint, output, positions=100, seed=2026, tolerance=2e-5):
    if positions < 1 or tolerance <= 0:
        raise ValueError('positive positions and tolerance required')
    model, state, actual = load_model(checkpoint)
    torch.set_num_threads(1)
    rng = random.Random(seed)
    board = chess.Board()
    parent = Path(output).resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    error = {'policy': 0, 'wdl': 0}
    with tempfile.TemporaryDirectory(dir=parent) as folder:
        binary = Path(folder) / 'verified.zero'
        model.export(binary)
        binary_hash = digest(binary)
        with Native(executable, binary) as native:
            for i in range(positions):
                if i:
                    if outcome(board):
                        board = chess.Board()
                    board.push(rng.choice(list(board.legal_moves)))
                native.position(board)
                result = native.command('zero eval')
                np.testing.assert_allclose(result['features'], features(board), atol=1e-7, rtol=0)
                if {v['uci']: v['action'] for v in result['legal']} != {m.uci(): action(m) for m in board.legal_moves}:
                    raise AssertionError('native legal moves/action encoding mismatch')
                with torch.no_grad():
                    policy, wdl = model(torch.from_numpy(features(board)).unsqueeze(0))
                    expected = {'policy': policy[0].numpy(), 'wdl': torch.softmax(wdl, -1)[0].numpy()}
                for name in error:
                    err = float(np.max(np.abs(np.asarray(result[name]) - expected[name])))
                    error[name] = max(error[name], err)
                    if err > tolerance:
                        raise AssertionError(f'{name} parity failed: {err} > {tolerance}')
    report = {'passed': True, 'positions': positions, 'seed': seed, 'step': state['step'],
              'checkpoint': str(actual), 'binary_sha256': binary_hash,
              'engine_sha256': digest(executable), 'tolerance': tolerance, 'max_abs_error': error}
    atomic_json(output, report)
    return report
