from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'vendor'))
from zero.model import ModelConfig,PolicyWDL
from shardtrain.durable import cleanup_bundles,load_bundle,read_manifest,save_bundle,verify_bundle
from shardtrain.storage import Ledger,ResourceGuard,digest


class DurableTests(unittest.TestCase):
    def setUp(self):
        root=PROJECT.parents[1]/'work'/'durable-test-temp';root.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=root);self.directory=Path(self.temp.name).resolve()
        self.assertTrue(self.directory.is_relative_to(root.resolve()))
        self.C=self.directory/'C';self.ledger=Ledger(self.directory/'ledger.sqlite')
        self.guard=ResourceGuard([self.directory],1.,4.)
        torch.set_num_threads(1);torch.manual_seed(17)
        self.model=PolicyWDL(ModelConfig(4,0));self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=.001)
        self.add_shard(1)

    def tearDown(self):
        self.ledger.close();self.temp.cleanup()

    def add_shard(self,identity):
        file=self.directory/f'shard-{identity}.jsonl';file.write_text('{"test":true}\n',encoding='utf-8')
        self.ledger.db.execute('INSERT INTO shards(id,path,sha,rows,status) VALUES (?,?,?,?,?)',
                               (identity,str(file),digest(file),1,'prepared'))
        return file

    def state(self,step=1,active=1,completed=None):
        return {'config':{'width':4,'blocks':0},'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),
                'step':step,'generator':torch.Generator().manual_seed(27).get_state(),'dataset':[],
                'scheduler':{'last_epoch':step,'learning_rate':.001},'settings':{'seed':17},
                'progress':{'completed':list(completed or []),'active':active,'epoch':0,'offset':0,
                            'permutation':['record-'+str(active)] if active is not None else [],'positions_processed':len(completed or [])}}

    def save(self,step=1,active=1,completed=None,completed_shard=None):
        return save_bundle(self.ledger,self.C,self.model,self.state(step,active,completed),self.guard,
                           {'global_step':step,'policy_loss':.2},completed_shard=completed_shard)

    def corrupt_pair(self,manifest):
        m=read_manifest(manifest)
        for key in ('checkpoint','mirror'):Path(m['files'][key]['path']).write_bytes(b'broken')

    def test_roundtrip_and_rng_and_header(self):
        before=torch.get_rng_state().clone();python_before=random.getstate()
        state,path=self.save()
        self.assertTrue(torch.equal(before,state['rng']['torch']))
        self.assertEqual(python_before,state['rng']['python'])
        loaded,actual,recovered=load_bundle(self.ledger,self.C)
        self.assertFalse(recovered);self.assertEqual(path,actual)
        self.assertEqual(loaded['step'],1);self.assertEqual(loaded['scheduler'],state['scheduler'])
        self.assertEqual(loaded['progress'],state['progress'])
        self.assertEqual(self.ledger.get('current_global_step'),1)
        self.assertEqual(read_manifest(path)['files']['checkpoint']['sha256'],read_manifest(path)['files']['mirror']['sha256'])
        self.assertEqual(verify_bundle(path)[0]['step'],1)

    def test_primary_corrupt_uses_same_commit_mirror(self):
        state,path=self.save();m=read_manifest(path)
        Path(m['files']['checkpoint']['path']).write_bytes(b'broken')
        loaded,actual,recovered=load_bundle(self.ledger,self.C)
        self.assertTrue(recovered);self.assertEqual(actual,path);self.assertEqual(loaded['step'],1)
        self.assertTrue(loaded['recovery']['mirror_used'])
        self.assertEqual(self.ledger.get('recovered_seq'),m['commit_seq'])

    def test_older_rollback_reconciles_done_and_logs(self):
        _,a=self.save(1)
        _,b=self.save(2,None,[1],1)
        self.add_shard(2)
        _,c=self.save(3,2,[1])
        self.corrupt_pair(c)
        state,path,recovered=load_bundle(self.ledger,self.C)
        self.assertTrue(recovered);self.assertEqual(path,b);self.assertEqual(state['step'],2)
        self.assertEqual(dict(self.ledger.db.execute('SELECT id,status FROM shards')),{1:'done',2:'prepared'})
        self.assertEqual([r[0] for r in self.ledger.db.execute('SELECT seq FROM checkpoints ORDER BY seq')],[1,2])
        self.assertEqual(self.ledger.db.execute('SELECT count(*) FROM events').fetchone()[0],2)
        self.assertEqual(self.ledger.get('current_global_step'),2)

    def test_missing_rollback_shard_refuses_recovery(self):
        _,a=self.save(1)
        _,b=self.save(2,None,[1],1)
        self.corrupt_pair(b)
        (self.directory/'shard-1.jsonl').unlink()
        with self.assertRaises(RuntimeError):load_bundle(self.ledger,self.C)
        self.assertEqual(self.ledger.get('current_checkpoint_seq'),2)

    def test_export_corruption_refuses_commit_recovery(self):
        _,a=self.save(1);_,b=self.save(2)
        Path(read_manifest(b)['files']['model']['path']).write_bytes(b'broken')
        state,path,recovered=load_bundle(self.ledger,self.C)
        self.assertTrue(recovered);self.assertEqual(path,a);self.assertEqual(state['step'],1)

    def test_precommit_crash_leaves_old_authority_and_orphan_cleanup(self):
        _,a=self.save(1)
        def crash(name,*args):
            if name=='checkpoint_written':raise RuntimeError('simulated termination before commit')
        with patch('shardtrain.durable.test_hook',side_effect=crash),self.assertRaises(RuntimeError):self.save(2)
        state,path,recovered=load_bundle(self.ledger,self.C)
        self.assertEqual(state['step'],1);self.assertEqual(path,a)
        self.assertEqual(self.ledger.db.execute('SELECT count(*) FROM checkpoints').fetchone()[0],1)
        files_before=list((self.C/'checkpoints').glob('*.manifest.json'))
        self.assertEqual(len(files_before),2)
        cleanup_bundles(self.ledger,self.C)
        self.assertEqual(len(list((self.C/'checkpoints').glob('*.manifest.json'))),1)

    def test_postcommit_crash_has_new_authority(self):
        self.save(1)
        def crash(name,*args):
            if name=='ledger_committed':raise RuntimeError('simulated termination after commit')
        with patch('shardtrain.durable.test_hook',side_effect=crash),self.assertRaises(RuntimeError):self.save(2,None,[1],1)
        state,path,recovered=load_bundle(self.ledger,self.C)
        self.assertEqual(state['step'],2);self.assertEqual(state['progress']['completed'],[1])
        self.assertEqual(self.ledger.db.execute('SELECT status FROM shards WHERE id=1').fetchone()[0],'done')

    def test_cleanup_is_bounded_and_does_not_delete_unknown_files(self):
        for step in range(1,7):self.save(step)
        outsider=self.C/'checkpoints'/'keep-this.txt';outsider.write_text('user',encoding='utf-8')
        cleanup_bundles(self.ledger,self.C)
        self.assertEqual(self.ledger.db.execute('SELECT count(*) FROM checkpoints').fetchone()[0],2)
        self.assertTrue(outsider.is_file());self.assertTrue((self.directory/'shard-1.jsonl').is_file())
        self.assertEqual(len(list((self.C/'checkpoints').glob('*.manifest.json'))),2)

    def test_nan_and_double_completion_are_rejected(self):
        state=self.state();state['scheduler']['learning_rate']=float('nan')
        with self.assertRaises(ValueError):save_bundle(self.ledger,self.C,self.model,state,self.guard,{'global_step':1})
        self.assertEqual(self.ledger.db.execute('SELECT count(*) FROM checkpoints').fetchone()[0],0)
        self.save(1,None,[1],1)
        with self.assertRaises(ValueError):self.save(2,None,[1],1)

    def test_checkpoint_directory_cannot_be_shared_between_ledgers(self):
        self.save(1)
        other=Ledger(self.directory/'other.sqlite')
        try:
            with self.assertRaises(RuntimeError):load_bundle(other,self.C)
        finally:other.close()

    def test_interrupted_owner_temp_is_ignored_without_rng_changes(self):
        self.C.mkdir()
        (self.C/'checkpoint-owner.json.tmp').write_bytes(b'partial owner write')
        before=torch.get_rng_state().clone();python_before=random.getstate()
        state,path,recovered=load_bundle(self.ledger,self.C)
        self.assertIsNone(state);self.assertIsNone(path);self.assertFalse(recovered)
        self.assertTrue((self.C/'checkpoint-owner.json').is_file())
        self.assertFalse((self.C/'checkpoint-owner.json.tmp').exists())
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertEqual(python_before,random.getstate())


if __name__=='__main__':unittest.main()
