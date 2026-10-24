"""Independent move oracle, UCI lifecycle, native/PyTorch parity, crash recovery."""
import argparse
from pathlib import Path
import json
import random
import shutil
import sys
import tempfile
import time
import unittest
import chess
import chess.engine
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zero import checkpoint
from zero.data import atomic_json, position_record, write_records
from zero.encoding import action, features, outcome
from zero.model import ModelConfig, PolicyWDL, load_model
from zero.native import Native
from zero.train import train, seed_all
from zero.selfplay import selfplay
from zero.loop import loop

ENGINE = None
REPORT = {}
TEMP_ROOT = Path(__file__).resolve().parents[1] / 'runs' / 'test-temp'
TEMP_ROOT.mkdir(parents=True, exist_ok=True)


def equal_nested(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_nested(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal_nested(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    return a == b


class Integration(unittest.TestCase):
    def test_move_oracle_and_encoder(self):
        rng = random.Random(2026)
        positions = [
            chess.Board(),
            chess.Board('1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1'),
            chess.Board('4k3/8/8/8/8/8/p7/1R2K3 b - - 0 1'),
            chess.Board('4k3/8/8/r4pPK/8/8/8/8 w - f6 0 1'),
            chess.Board('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1'),
            chess.Board('7k/6Q1/6K1/8/8/8/8/8 b - - 100 1'),
            chess.Board('7k/5Q2/6K1/8/8/8/8/8 b - - 0 1'),
        ]
        repeat = chess.Board()
        for _ in range(2):
            for text in ['g1f3', 'g8f6', 'f3g1', 'f6g8']:
                repeat.push_uci(text)
        positions.append(repeat)
        board = chess.Board()
        for _ in range(1000):
            if outcome(board) or len(board.move_stack) > 150:
                board = chess.Board()
            board.push(rng.choice(list(board.legal_moves)))
            positions.append(board.copy(stack=True))
        with Native(ENGINE) as native:
            for board in positions:
                native.position(board)
                result = native.command('zero inspect')
                self.assertEqual(result['fen'], board.fen(en_passant='fen'))
                self.assertEqual({r['uci']: r['action'] for r in result['legal']},
                                 {m.uci(): action(m) for m in board.legal_moves})
                np.testing.assert_allclose(result['features'], features(board), atol=1e-7, rtol=0)
                self.assertEqual(result['terminal'], outcome(board) is not None)
        REPORT['move_oracle_positions'] = len(positions)

    def test_native_model_parity_and_invalid_load(self):
        seed_all(17)
        torch.set_num_threads(1)
        model = PolicyWDL(ModelConfig(32, 2)).eval()
        boards = [chess.Board(), chess.Board('1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1')]
        board = chess.Board()
        rng = random.Random(43)
        for _ in range(50):
            if outcome(board):
                board = chess.Board()
            board.push(rng.choice(list(board.legal_moves)))
            boards.append(board.copy(stack=True))
        errors = {'policy': 0, 'wdl': 0}
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            path = Path(folder) / 'model.zero'; model.export(path)
            corrupt = Path(folder) / 'corrupt.zero'; corrupt.write_bytes(path.read_bytes()[:100])
            with Native(ENGINE, path) as native:
                for b in boards:
                    native.position(b)
                    result = native.command('zero eval')
                    with torch.no_grad():
                        policy, wdl = model(torch.from_numpy(features(b)).unsqueeze(0))
                        wdl = torch.softmax(wdl, -1)
                    for name, expected in [('policy', policy[0].numpy()), ('wdl', wdl[0].numpy())]:
                        err = float(np.max(np.abs(np.asarray(result[name]) - expected)))
                        errors[name] = max(errors[name], err)
                        self.assertLess(err, 2e-5)
                    sparse=native.command('zero evallegal')
                    legal_ids=[action(m) for m in b.legal_moves]
                    np.testing.assert_array_equal(np.asarray(sparse['policy'])[legal_ids],np.asarray(result['policy'])[legal_ids])
                    if b is boards[0]:
                        native.configure({'SparsePolicyHead':False})
                        full=native.command('zero evallegal')
                        self.assertEqual(full['policy'],result['policy'])
                        native.configure({'SparsePolicyHead':True})
                    native.configure({'InferenceBackend':'SIMD'})
                    simd=native.command('zero eval')
                    self.assertLess(float(np.max(np.abs(np.asarray(simd['policy'])-policy[0].numpy()))),2e-5)
                    self.assertLess(float(np.max(np.abs(np.asarray(simd['wdl'])-wdl[0].numpy()))),2e-5)
                    native.configure({'InferenceBackend':'Scalar'})
                native.send('setoption name ModelFile value ' + str(corrupt))
                with self.assertRaisesRegex(RuntimeError, 'truncated'):
                    native.ready()
                # Drain the earlier ready response before comparing the old model.
                native.ready()
                old = native.command('zero eval')
                self.assertEqual(old['policy'], result['policy'])
            auto_engine = Path(folder) / ENGINE.name
            shutil.copyfile(ENGINE, auto_engine)
            model.export(Path(folder) / 'default.zero')
            with Native(auto_engine) as automatic:
                automatic.configure({'InferenceBackend':'Scalar'})
                automatic.position(boards[-1])
                self.assertEqual(automatic.command('zero eval')['policy'], result['policy'])
            selfplay(auto_engine, None, Path(folder) / 'automatic-games', games=1,
                     simulations=2, max_plies=1, noise=0, temperature_plies=0,
                     start_fen='7k/8/5KQ1/8/8/8/8/8 w - - 99 1')
            settings = json.loads((Path(folder) / 'automatic-games/settings.json').read_text())
            self.assertEqual(settings['model'], checkpoint.digest(Path(folder) / 'default.zero'))
        REPORT['inference'] = dict(positions=len(boards), tolerance=2e-5, max_abs_error=errors)

    def test_uci_time_stop_and_protocol(self):
        with Native(ENGINE) as native:
            board = chess.Board()
            native.position(board)
            native.send('go nodes 48')
            move = native.until(lambda s: s.startswith('bestmove ')).split()[1]
            self.assertIn(chess.Move.from_uci(move), board.legal_moves)
            start = time.monotonic()
            native.send('go infinite')
            native.ready()
            native.send('stop')
            move = native.until(lambda s: s.startswith('bestmove '), timeout=3).split()[1]
            self.assertIn(chess.Move.from_uci(move), board.legal_moves)
            self.assertLess(time.monotonic() - start, 3)
            start = time.monotonic()
            native.send('go movetime 40')
            native.until(lambda s: s.startswith('bestmove '), timeout=3)
            elapsed = time.monotonic() - start
            self.assertLess(elapsed, 1.5)
            native.send('go wtime 0 btime 0')
            native.until(lambda s: s.startswith('bestmove '), timeout=2)
            native.send('position startpos moves e2e5')
            with self.assertRaisesRegex(RuntimeError, 'illegal move'):
                native.ready()
            native.ready()
            self.assertEqual(native.command('zero inspect')['fen'], board.fen())
            native.position(chess.Board('7k/6Q1/6K1/8/8/8/8/8 b - - 0 1'))
            native.send('go nodes 8')
            self.assertEqual(native.until(lambda s: s.startswith('bestmove ')), 'bestmove 0000')
        # An independent UCI consumer exercises normal GUI-style initialization/play.
        with chess.engine.SimpleEngine.popen_uci(str(ENGINE)) as uci:
            result = uci.play(chess.Board(), chess.engine.Limit(nodes=32))
            self.assertIn(result.move, chess.Board().legal_moves)
        REPORT['uci'] = {'nodes': True, 'stop': True, 'ready_during_search': True,
                         'movetime_40ms_elapsed': elapsed, 'zero_clock': True,
                         'terminal': True, 'python_chess_client': True}

    def test_checkpoint_exact_resume_and_corruption_recovery(self):
        rng = random.Random(9)
        board = chess.Board(); rows = []
        for i in range(24):
            moves = list(board.legal_moves)
            rows.append(position_record(board, [[action(m), 1 / len(moves)] for m in moves],
                                        'selfplay', wdl=[0, 1, 0]))
            board.push(rng.choice(moves))
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            root = Path(folder); data = root / 'data.jsonl'; write_records(data, rows)
            full = root / 'full.pt'; split = root / 'split.pt'
            kwargs = dict(batch_size=8, lr=0.001, seed=67, width=16, blocks=1, save_every=4)
            train([data], full, steps=12, **kwargs)
            train([data], split, steps=4, **kwargs)
            train([data], split, steps=12, resume=split, **kwargs)
            a, _ = checkpoint.load(full); b, _ = checkpoint.load(split)
            for key in ('model', 'optimizer', 'generator', 'rng', 'step'):
                self.assertTrue(equal_nested(a[key], b[key]), key + ' differs after resume')
            split.write_bytes(b'interrupted-checkpoint')
            state, actual = checkpoint.load(split)
            self.assertEqual(str(actual), str(split) + '.prev')
            self.assertEqual(state['step'], 8)
            train([data], split, steps=16, resume=split, **kwargs)
            train([data], full, steps=16, resume=full, **kwargs)
            a, _ = checkpoint.load(full); b, _ = checkpoint.load(split)
            self.assertTrue(equal_nested(a['model'], b['model']))
            changed = root / 'changed.jsonl'; write_records(changed, rows[:1])
            with self.assertRaisesRegex(ValueError, 'identical dataset'):
                train([changed], split, steps=20, resume=split, **kwargs)
        REPORT['checkpoint'] = {'exact_resume': True, 'model_optimizer_rng_generator_equal': True,
                                'corrupt_latest_fallback': True, 'dataset_change_rejected': True}

    def test_selfplay_labels_truncation_and_restartable_loop(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            root = Path(folder)
            # A nonterminal position at move 99 always finishes or reaches a valid draw.
            fen = '7k/8/5KQ1/8/8/8/8/8 w - - 99 1'
            summary = selfplay(ENGINE, None, root / 'games', games=2, simulations=64,
                               seed=3, max_plies=8, start_fen=fen, noise=0, temperature_plies=0)
            self.assertEqual(summary['accepted'], 2)
            self.assertEqual(summary, selfplay(ENGINE, None, root / 'games', games=2, simulations=64,
                                             seed=3, max_plies=8, start_fen=fen, noise=0, temperature_plies=0))
            rows = [json.loads(line) for p in (root / 'games').glob('*.jsonl') for line in p.read_text().splitlines()]
            for row in rows:
                self.assertEqual(row['wdl'], [1, 0, 0])
            truncated = selfplay(ENGINE, None, root / 'truncated', games=1, simulations=1, max_plies=1)
            self.assertEqual(truncated['records'], 0)
            self.assertEqual(truncated['truncated'], 1)
            initial = root / 'initial.pt'
            from zero.train import initialize
            initialize(initial, width=8, blocks=1)
            state = loop(ENGINE, initial, root / 'loop', iterations=1, games=2, simulations=16,
                         steps=3, batch_size=2, pairs=1, max_plies=8, start_fen=fen,
                         opening_fens=[fen])
            self.assertEqual(state['completed'], 1)
            self.assertEqual(state, loop(ENGINE, initial, root / 'loop', iterations=1, games=2,
                                        simulations=16, steps=3, batch_size=2, pairs=1,
                                        max_plies=8, start_fen=fen, opening_fens=[fen]))
            self.assertFalse(state['reports'][0]['promoted'])
        REPORT['pipeline'] = {'actual_result_labels': True, 'truncated_games_discarded': True,
                              'selfplay_resume': True, 'complete_generation_and_restart': True,
                              'insufficient_arena_evidence_no_promotion': True}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--engine', required=True); p.add_argument('--report')
    args = p.parse_args(); ENGINE = Path(args.engine).resolve()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Integration)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    REPORT['passed'] = result.wasSuccessful()
    if args.report:
        atomic_json(args.report, REPORT)
    sys.exit(not result.wasSuccessful())
