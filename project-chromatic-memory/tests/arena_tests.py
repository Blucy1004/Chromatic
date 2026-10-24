import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zero.match import statistics,sprt_state,arena
from zero.champion import rollback
from zero.checkpoint import digest
from zero.data import atomic_json
import chess.pgn

ENGINE=None;TEMP=Path(__file__).resolve().parents[1]/'runs/test-temp';TEMP.mkdir(parents=True,exist_ok=True)
class ArenaTests(unittest.TestCase):
    def test_statistics_distinguishes_uncertainty_from_inferiority(self):
        self.assertEqual(statistics([{'score':.5},{'score':.5}])['decision'],'insufficient_evidence')
        self.assertEqual(statistics([{'score':1},{'score':None}])['decision'],'insufficient_evidence')
        self.assertIsNone(statistics([{'score':1},{'score':None}])['elo'])
        self.assertEqual(statistics([{'score':0}]*200)['decision'],'demonstrated_inferiority')
        self.assertTrue(statistics([{'score':1}]*200)['promote'])
        self.assertEqual(sprt_state([{'score':.5}]*24)['status'],'continue')
        self.assertEqual(sprt_state([{'score':1}]*24)['status'],'accept_h1')
        repeated=[{'score':1,'opening':{'fen':'same-fixture','moves':[]}}]*200
        self.assertFalse(statistics(repeated)['promote'])
        self.assertEqual(statistics(repeated)['effective_independent_units'],1)
        self.assertEqual(sprt_state(repeated)['status'],'continue')
    def test_pgn_fixed_opening_pair_and_restart(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as folder:
            path=Path(folder)/'arena.json';fen='7k/8/5KQ1/8/8/8/8/8 w - - 99 1'
            report=arena(ENGINE,None,None,path,pairs=1,simulations=8,max_plies=2,opening_fens=[fen])
            self.assertEqual(report['wdl'],{'wins':1,'draws':0,'losses':1,'unfinished':0})
            self.assertFalse(report['promote']);self.assertEqual(report['elo'],0)
            with path.with_suffix('.pgn').open() as stream:
                games=[chess.pgn.read_game(stream),chess.pgn.read_game(stream)]
            self.assertTrue(all(game and not game.errors for game in games))
            again=arena(ENGINE,None,None,path,pairs=1,simulations=8,max_plies=2,opening_fens=[fen])
            self.assertEqual(report,again)
    def test_rollback_preserves_artifacts_and_checks_hash(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as folder:
            root=Path(folder);(root/'old.pt').write_bytes(b'old-fixture');(root/'new.pt').write_bytes(b'new-fixture')
            state={'champion':'new.pt','champion_sha256':digest(root/'new.pt'),'completed':2,
                   'history':[{'previous_champion':'old.pt','previous_sha256':digest(root/'old.pt')}],'rollbacks':[]}
            atomic_json(root/'state.json',state);result=rollback(root)
            self.assertEqual(result['champion'],'old.pt');self.assertTrue((root/'new.pt').exists())
            self.assertEqual(result['completed'],2)
            with self.assertRaisesRegex(ValueError,'not enough'):rollback(root)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--engine',required=True);p.add_argument('--report',required=True)
    args=p.parse_args();ENGINE=Path(args.engine).resolve()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ArenaTests))
    atomic_json(args.report,{'passed':result.wasSuccessful(),'tests':result.testsRun});sys.exit(not result.wasSuccessful())
