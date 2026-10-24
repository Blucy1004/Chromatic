import argparse
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zero.native import Native
from zero.train import initialize
from zero import checkpoint
from zero.data import atomic_json
from zero.encoding import INPUTS,ACTIONS

ENGINE=None;TEMP=Path(__file__).resolve().parents[1]/'runs/test-temp';TEMP.mkdir(parents=True,exist_ok=True)
class Safety(unittest.TestCase):
    def test_missing_startup_model_stays_uci_responsive(self):
        result=subprocess.run([str(ENGINE),'--model','missing-model.zero'],input='uci\nisready\nquit\n',
                              text=True,capture_output=True,timeout=5)
        self.assertEqual(result.returncode,0);self.assertIn('uciok',result.stdout)
        self.assertIn('readyok',result.stdout);self.assertIn('cannot open network',result.stdout)
    def test_malformed_go_returns_completion_and_recovers(self):
        with Native(ENGINE) as engine:
            engine.send('go nodes nonsense')
            with self.assertRaisesRegex(RuntimeError,'error'):engine.until(lambda line:line.startswith('bestmove '))
            self.assertEqual(engine.until(lambda line:line.startswith('bestmove ')),'bestmove 0000');engine.ready()
            for fen in ('8/8/8/8/8/8/8/8 w - - 0 1','4k3/8/8/8/8/8/8/4K3 w K - 0 1'):
                engine.send('position fen '+fen)
                with self.assertRaises(RuntimeError):engine.ready()
                engine.ready()
    def test_finite_weights_that_overflow_cannot_crash_search(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as folder:
            path=Path(folder)/'overflow.zero'
            arrays=[struct.pack('<f',3e38)*INPUTS,struct.pack('<f',1.),b'\0'*(ACTIONS*8),b'\0'*24]
            path.write_bytes(b'ZERO0001'+struct.pack('<4I',INPUTS,1,0,ACTIONS)+b''.join(arrays))
            with Native(ENGINE,path) as engine:
                with self.assertRaisesRegex(RuntimeError,'nonfinite'):engine.command('zero eval')
                engine.send('go nodes 1')
                with self.assertRaisesRegex(RuntimeError,'nonfinite'):engine.until(lambda line:line.startswith('bestmove '))
                self.assertEqual(engine.until(lambda line:line.startswith('bestmove ')),'bestmove 0000')
    def test_bad_optimizer_never_replaces_last_valid_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as folder:
            path=Path(folder)/'model.pt';initialize(path,width=1,blocks=0)
            before=checkpoint.digest(path);state,_=checkpoint.load(path)
            state['optimizer']={'state':{0:{'exp_avg':torch.tensor(float('nan'))}}}
            with self.assertRaisesRegex(ValueError,'optimizer'):checkpoint.save(path,state)
            self.assertEqual(checkpoint.digest(path),before)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--engine',required=True);p.add_argument('--report',required=True)
    args=p.parse_args();ENGINE=Path(args.engine).resolve()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Safety))
    atomic_json(args.report,{'passed':result.wasSuccessful(),'tests':result.testsRun});sys.exit(not result.wasSuccessful())
