import json
import os
from pathlib import Path
import numpy as np
import torch
from .encoding import ACTIONS, SCHEMA, action, features, restore_board


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path) + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def write_records(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path) + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, separators=(',', ':')) + '\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def position_record(board, policy, source, **metadata):
    return {'schema': SCHEMA, 'source': source, 'root_fen': board.root().fen(en_passant='fen'),
            'moves': [m.uci() for m in board.move_stack], 'fen': board.fen(en_passant='fen'),
            'policy': policy, **metadata}


def tensors(paths):
    xs, ps, vs, masks = [], [], [], []
    for path in paths:
        with open(path, encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                try:
                    r = json.loads(line)
                    if r['schema'] != SCHEMA or r['source'] not in ('bootstrap', 'selfplay'):
                        raise ValueError('unsupported record schema/source')
                    b = restore_board(r)
                    legal = {action(m) for m in b.legal_moves}
                    if not legal:
                        raise ValueError('terminal position in training data')
                    p = np.zeros(ACTIONS, dtype=np.float32)
                    seen = set()
                    for a, prob in r['policy']:
                        if not isinstance(a, int) or a not in legal or a in seen or not np.isfinite(prob) or prob < 0:
                            raise ValueError('invalid policy target')
                        seen.add(a)
                        p[a] = prob
                    if not np.isclose(p.sum(), 1, atol=1e-5):
                        raise ValueError('policy does not sum to one')
                    v = np.asarray(r['wdl'], dtype=np.float32)
                    if v.shape != (3,) or not np.isfinite(v).all() or (v < 0).any() or not np.isclose(v.sum(), 1, atol=1e-5):
                        raise ValueError('invalid WDL target')
                    mask = np.zeros(ACTIONS, dtype=np.bool_)
                    mask[list(legal)] = True
                    xs.append(features(b)); ps.append(p); vs.append(v); masks.append(mask)
                except Exception as exc:
                    raise ValueError(f'{path}:{line_number}: {exc}') from exc
    if not xs:
        raise ValueError('empty training dataset')
    return tuple(torch.from_numpy(np.stack(items)) for items in (xs, ps, vs, masks))
