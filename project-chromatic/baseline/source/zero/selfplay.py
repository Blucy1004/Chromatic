from pathlib import Path
import chess
from .checkpoint import digest
from .data import atomic_json, position_record, write_records
from .encoding import action, outcome
from .native import Native


def selfplay(executable, model, output, games=8, simulations=128, seed=1,
             max_plies=512, start_fen=chess.STARTING_FEN, noise=0.25, temperature_plies=30):
    if games < 1 or simulations < 1 or max_plies < 1 or not 0 <= noise <= 1:
        raise ValueError('invalid selfplay settings')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if model is None:
        adjacent = Path(executable).resolve().parent / 'default.zero'
        if adjacent.exists():
            model = adjacent
    model_hash = digest(model) if model else 'uniform-baseline'
    settings = {'model': model_hash, 'executable': digest(executable), 'simulations': simulations, 'seed': seed,
                'max_plies': max_plies, 'start_fen': start_fen, 'noise': noise,
                'temperature_plies': temperature_plies}
    manifest = output / 'settings.json'
    if manifest.exists():
        import json
        if json.loads(manifest.read_text(encoding='utf-8')) != settings:
            raise ValueError('selfplay output belongs to different settings/model')
    else:
        atomic_json(manifest, settings)
    accepted, truncated, records_count = 0, 0, 0
    with Native(executable, model) as engine:
        for game in range(games):
            path = output / f'game_{game:06d}.jsonl'
            meta = path.with_suffix('.json')
            if path.exists() and meta.exists():
                import json
                info = json.loads(meta.read_text(encoding='utf-8'))
                if info['sha256'] != digest(path):
                    raise ValueError('selfplay shard checksum mismatch')
            else:
                board = chess.Board(start_fen)
                if not board.is_valid():
                    raise ValueError('invalid selfplay start position')
                records, turns, played = [], [], []
                for ply in range(max_plies):
                    if outcome(board):
                        break
                    engine.position(board)
                    result = engine.search(simulations, seed + game * 1000003 + ply,
                                           noise, 1.0 if ply < temperature_plies else 0.0)
                    if result['terminal']:
                        raise RuntimeError('C++ and Python disagree about termination')
                    move = chess.Move.from_uci(result['bestmove'])
                    if move not in board.legal_moves:
                        raise RuntimeError('native illegal selfplay move')
                    total = sum(v['n'] for v in result['visits'])
                    if total != result['simulations'] or total <= 0:
                        raise RuntimeError('invalid MCTS visit distribution')
                    policy = [[v['action'], v['n'] / total] for v in result['visits'] if v['n']]
                    records.append(position_record(board, policy, 'selfplay', model=model_hash,
                                                   game=game, simulations=total))
                    turns.append(board.turn)
                    board.push(move); played.append(move.uci())
                result = outcome(board)
                if result:
                    for row, turn in zip(records, turns):
                        row['wdl'] = [0, 1, 0] if result.winner is None else [1, 0, 0] if result.winner == turn else [0, 0, 1]
                        row['result'] = result.result()
                    kept = records
                else:
                    # A ply cap is unknown outcome, never a synthetic draw target.
                    kept = []
                write_records(path, kept)
                info = {'game': game, 'accepted': result is not None, 'records': len(kept),
                        'plies': len(played), 'result': result.result() if result else '*',
                        'termination': result.termination.name if result else 'TRUNCATED',
                        'moves': played, 'sha256': digest(path)}
                atomic_json(meta, info)
            accepted += info['accepted']; truncated += not info['accepted']; records_count += info['records']
    summary = {'games': games, 'accepted': accepted, 'truncated': truncated, 'records': records_count}
    atomic_json(output / 'summary.json', summary)
    return summary
