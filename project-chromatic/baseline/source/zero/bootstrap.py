import math
from pathlib import Path
import random
import chess
import chess.engine
from .checkpoint import digest
from .data import atomic_json, position_record, write_records
from .encoding import action, outcome


def bootstrap(teacher, output, positions=128, nodes=2000, multipv=4, seed=1):
    """A bounded one-time teacher dataset; teacher is absent from selfplay runtime."""
    if not 1 <= positions <= 10000 or nodes < 1 or multipv < 1:
        raise ValueError('bootstrap positions must be 1..10000, positive nodes/multipv')
    rng = random.Random(seed)
    rows = []
    teacher_hash = digest(teacher)
    with chess.engine.SimpleEngine.popen_uci(str(Path(teacher).resolve())) as engine:
        teacher_id = dict(engine.id)
        options = {}
        if 'Threads' in engine.options:
            options['Threads'] = 1
        if 'Hash' in engine.options:
            options['Hash'] = 32
        if 'UCI_ShowWDL' in engine.options:
            options['UCI_ShowWDL'] = True
        if options:
            engine.configure(options)
        for i in range(positions):
            board = chess.Board()
            for _ in range(rng.randint(8, 60)):
                if outcome(board):
                    board = chess.Board()
                board.push(rng.choice(list(board.legal_moves)))
            if outcome(board):
                board = chess.Board()
            infos = engine.analyse(board, chess.engine.Limit(nodes=nodes),
                                   multipv=min(multipv, board.legal_moves.count()))
            if not infos or any(not item.get('pv') or 'score' not in item for item in infos):
                raise RuntimeError('teacher did not provide legal PVs and scores')
            scores = [item['score'].pov(board.turn).score(mate_score=30000) for item in infos]
            logits = [max(-80.0, (s - max(scores)) / 100.0) for s in scores]
            weights = [math.exp(s) for s in logits]
            policy = [[action(info['pv'][0]), w / sum(weights)] for info, w in zip(infos, weights)]
            best = infos[0]
            if 'wdl' in best:
                wdl = list(best['wdl'].pov(board.turn))
                wdl = [v / sum(wdl) for v in wdl]
                value_source = 'teacher-wdl'
            else:
                # Explicit soft heuristic label only during bootstrap, never an actual game result.
                expected = 1 / (1 + math.exp(-max(-80.0, min(80.0, scores[0] / 400.0))))
                draw = 0.5 * (1 - abs(2 * expected - 1))
                wdl = [expected - draw / 2, draw, 1 - expected - draw / 2]
                value_source = 'cp-logistic-400'
            rows.append(position_record(board, policy, 'bootstrap', wdl=wdl, teacher=teacher_hash,
                                        value_source=value_source, teacher_nodes=nodes, index=i))
    write_records(output, rows)
    report = {'positions': len(rows), 'teacher_sha256': teacher_hash, 'teacher_id': teacher_id,
              'nodes': nodes, 'multipv': multipv, 'seed': seed, 'sha256': digest(output)}
    atomic_json(Path(output).with_suffix('.meta.json'), report)
    return report
