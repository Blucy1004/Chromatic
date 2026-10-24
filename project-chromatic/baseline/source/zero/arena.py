from pathlib import Path
import math
import random
import chess
from .checkpoint import digest
from .data import atomic_json
from .encoding import outcome
from .native import Native


def arena(executable, candidate, incumbent, output, pairs=8, simulations=128,
          max_plies=512, seed=1, threshold=0.55, confidence=0.95,
          opening_fens=None):
    """Paired colors; a conservative Hoeffding bound over independent opening pairs."""
    if pairs < 1 or not 0 < confidence < 1 or not 0.5 <= threshold <= 1:
        raise ValueError('invalid arena gate settings')
    rng = random.Random(seed)
    games = []
    with Native(executable, candidate) as challenger, Native(executable, incumbent) as champion:
        for pair in range(pairs):
            if opening_fens:
                opening = chess.Board(opening_fens[pair % len(opening_fens)])
            else:
                opening = chess.Board()
                for _ in range(6):
                    opening.push(rng.choice(list(opening.legal_moves)))
            if not opening.is_valid() or outcome(opening):
                raise ValueError('arena opening must be valid and nonterminal')
            for candidate_white in (True, False):
                board = opening.copy(stack=True)
                for ply in range(max_plies):
                    if outcome(board):
                        break
                    engine = challenger if board.turn == candidate_white else champion
                    engine.position(board)
                    result = engine.search(simulations, seed + pair, 0, 0)
                    if result['terminal']:
                        raise RuntimeError('arena terminal disagreement')
                    move = chess.Move.from_uci(result['bestmove'])
                    if move not in board.legal_moves:
                        raise RuntimeError('arena illegal move')
                    board.push(move)
                result = outcome(board)
                score = None if result is None else 0.5 if result.winner is None else float(result.winner == candidate_white)
                games.append({'pair': pair, 'candidate_white': candidate_white,
                              'score': score, 'result': result.result() if result else '*',
                              'opening': opening.fen(en_passant='fen'),
                              'moves': [m.uci() for m in board.move_stack],
                              'termination': result.termination.name if result else 'TRUNCATED'})
    complete = all(g['score'] is not None for g in games)
    score = sum(g['score'] or 0 for g in games) / len(games) if complete else None
    margin = math.sqrt(math.log(1 / (1 - confidence)) / (2 * pairs))
    lower = max(0.0, score - margin) if complete else None
    report = {'candidate': digest(candidate), 'incumbent': digest(incumbent),
              'pairs': pairs, 'completed_games': sum(g['score'] is not None for g in games),
              'score': score, 'lower_bound': lower, 'confidence': confidence,
              'threshold': threshold, 'promote': complete and lower >= threshold,
              'simulations': simulations, 'games': games}
    atomic_json(output, report)
    return report
