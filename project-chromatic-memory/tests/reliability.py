"""CHROMATIC replay, trace, holdout, optimizer and tiny-data learning regressions."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import chess
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zero.data import position_record,write_records,atomic_json
from zero.encoding import action
from zero.replay import Replay,prepare
from zero.train import train
from zero.selfplay import selfplay
from zero import checkpoint

ENGINE=None
TEMP=Path(__file__).resolve().parents[1]/'runs/test-temp'
TEMP.mkdir(parents=True,exist_ok=True)
REPORT={}

class Reliability(unittest.TestCase):
    def test_sparse_replay_capacity_dedup_and_group_holdout(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as folder:
            root=Path(folder);paths=[];board=chess.Board()
            for generation in range(2):
                rows=[]
                for i in range(12):
                    moves=list(board.legal_moves);move=moves[i%len(moves)]
                    rows.append(position_record(board,[[action(move),1]],'selfplay',wdl=[0,1,0],game_id=f'g{generation}-game{i//4}'))
                    board.push(move)
                path=root/str(generation)/'data.jsonl';write_records(path,rows+rows[:2]);paths.append(path)
            replay=Replay(paths,capacity=8,recent_fraction=.5,seed=9)
            self.assertLessEqual(len(replay.rows),8)
            self.assertEqual(replay.stats['older_selected'],4);self.assertEqual(replay.stats['recent_selected'],4)
            self.assertEqual(replay.stats['duplicates_in_window'],4)
            training,validation=replay.split(.25,9)
            self.assertTrue(validation)
            self.assertFalse({replay.rows[i].position for i in training}&{replay.rows[i].position for i in validation})
            replay.persist(root/'replay.json')
            self.assertTrue(all(path.exists() for path in paths))
        REPORT['bounded_replay_dedup_holdout']=True

    def test_unfinished_trajectory_is_durable_and_resumable(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as folder:
            path=Path(folder)/'games'
            first=selfplay(ENGINE,None,path,games=1,simulations=4,max_plies=1,noise=0,temperature_plies=0)
            self.assertEqual(first['records'],0)
            pending=path/'pending/game_000000.jsonl'
            rows=[json.loads(line) for line in pending.read_text().splitlines()]
            self.assertEqual(len(rows),1);self.assertNotIn('wdl',rows[0])
            with self.assertRaisesRegex(ValueError,'unfinished'):
                prepare(rows[0])
            second=selfplay(ENGINE,None,path,games=1,simulations=4,max_plies=2,noise=0,temperature_plies=0,resume_truncated=True)
            rows2=[json.loads(line) for line in pending.read_text().splitlines()]
            self.assertEqual(rows2[0],rows[0]);self.assertEqual(len(rows2),2)
            self.assertEqual(second['records'],0)
            completed=Path(folder)/'mate'
            selfplay(ENGINE,None,completed,games=1,simulations=64,max_plies=2,noise=0,temperature_plies=0,
                     start_fen='7k/8/5KQ1/8/8/8/8/8 w - - 99 1')
            row=json.loads((completed/'game_000000.jsonl').read_text().splitlines()[0]);prepared=prepare(row)
            self.assertEqual(row['schema'],2);self.assertEqual(row['wdl'],[1,0,0])
            row['legal_actions']=[]
            with self.assertRaisesRegex(ValueError,'mask mismatch'):prepare(row)
        REPORT['unfinished_durable_no_fake_outcome_resume']=True
        REPORT['versioned_trace_features_mask_action_outcome']=True

    def test_small_data_overfit_and_validation_metrics(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as folder:
            root=Path(folder);board=chess.Board();move=chess.Move.from_uci('e2e4')
            row=position_record(board,[[action(move),1]],'selfplay',wdl=[1,0,0])
            path=root/'tiny.jsonl';write_records(path,[row])
            result=train([path],root/'fit.pt',steps=160,batch_size=4,lr=.01,width=16,blocks=1,
                         save_every=80,validation_fraction=0,weight_decay=0)
            before=result['initial']['training'];after=result['final']['training']
            self.assertLess(after['policy_loss'],.03);self.assertLess(after['wdl_loss'],.03)
            self.assertLess(after['policy_loss']+after['wdl_loss'],.02*(before['policy_loss']+before['wdl_loss']))
            self.assertIn('wdl_ece',after);self.assertIn('gradient_norm_before_clip',result['last'])
            self.assertGreater(result['compute_samples_per_second'],0)
            state,_=checkpoint.load(root/'fit.pt');self.assertEqual(state['format'],2)
            # A separate optimizer path must also run with finite gradients.
            sgd=train([path],root/'sgd.pt',steps=2,batch_size=1,width=8,blocks=0,optimizer_name='sgd',validation_fraction=0)
            self.assertTrue(np.isfinite(sgd['last']['wdl_loss']))
            REPORT['overfit']={'initial':before,'final':after,'steps':160,
                               'samples_per_second':result['compute_samples_per_second']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--engine',required=True);p.add_argument('--report',required=True)
    args=p.parse_args();ENGINE=Path(args.engine).resolve()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Reliability))
    REPORT['passed']=result.wasSuccessful();atomic_json(args.report,REPORT);sys.exit(not result.wasSuccessful())
