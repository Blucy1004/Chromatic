import chess
import numpy as np

INPUTS, ACTIONS, SCHEMA = 1280, 4672, 1


def action(move: chess.Move) -> int:
    if move.promotion in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
        delta = chess.square_file(move.to_square) - chess.square_file(move.from_square)
        return 4096 + move.from_square * 9 + (delta + 1) * 3 + move.promotion - 2
    return move.from_square * 64 + move.to_square


def repetitions(board: chess.Board) -> int:
    return 3 if board.is_repetition(3) else 2 if board.is_repetition(2) else 1


def features(board: chess.Board) -> np.ndarray:
    x = np.zeros((20, 64), dtype=np.float32)
    for square, piece in board.piece_map().items():
        x[(0 if piece.color == chess.WHITE else 6) + piece.piece_type - 1, square] = 1
    x[12] = board.turn == chess.WHITE
    for i, (color, king) in enumerate(((True, True), (True, False), (False, True), (False, False))):
        x[13 + i] = board.has_kingside_castling_rights(color) if king else board.has_queenside_castling_rights(color)
    if board.ep_square is not None:
        x[17, board.ep_square] = 1
    x[18] = min(board.halfmove_clock, 100) / 100
    x[19] = repetitions(board) / 3
    return x.reshape(-1)


def outcome(board: chess.Board):
    """Claim 50-move/threefold as soon as already available; no prospective claims."""
    if board.is_checkmate():
        return chess.Outcome(chess.Termination.CHECKMATE, not board.turn)
    if board.is_stalemate():
        return chess.Outcome(chess.Termination.STALEMATE, None)
    if board.is_insufficient_material():
        return chess.Outcome(chess.Termination.INSUFFICIENT_MATERIAL, None)
    if board.halfmove_clock >= 100:
        return chess.Outcome(chess.Termination.FIFTY_MOVES, None)
    if board.is_repetition(3):
        return chess.Outcome(chess.Termination.THREEFOLD_REPETITION, None)
    return None


def restore_board(record):
    board = chess.Board(record['root_fen'])
    for move in record['moves']:
        board.push_uci(move)
    if board.fen(en_passant='fen') != record['fen']:
        raise ValueError('record history/FEN mismatch')
    return board
