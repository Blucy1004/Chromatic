from pathlib import Path
from contextlib import closing
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from shardtrain.durable import FORMAT,read_manifest,save_bundle,verify_bundle
from shardtrain.model_adapter import ModelConfig,PolicyWDL,checkpoint
from shardtrain.storage import Ledger,ResourceGuard,atomic_json,canonical,digest
from tools.import_legacy import import_legacy,EXECUTION

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parents[1]/'work/arc-legacy-import-tests'

def tree_hashes(root):
    return {p.relative_to(root).as_posix():digest(p) for p in root.rglob('*') if p.is_file()}

class ImportLegacyTests(unittest.TestCase):
    def setUp(self):
        WORK.mkdir(parents=True,exist_ok=True);self.temp=tempfile.TemporaryDirectory(dir=WORK)
        self.root=Path(self.temp.name).resolve();self.legacy=self.root/'legacy';self.data=self.legacy/'data';self.C=self.legacy/'candidate'
        (self.data/'fixed').mkdir(parents=True);(self.data/'shards').mkdir();self.C.mkdir()
        self.ledger=Ledger(self.data/'ledger.sqlite3');torch.set_num_threads(1);torch.manual_seed(28)
        model=PolicyWDL(ModelConfig(4,0));optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        (sum(t.square().mean() for t in model(torch.ones(1,1280)))).backward();optimizer.step()
        self.generator=torch.Generator().manual_seed(7)
        basic={'config':{'width':4,'blocks':0},'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'step':1,'generator':self.generator.get_state(),'dataset':[]}
        initial=self.legacy/'initial.pt';checkpoint.save(initial,basic)
        source=self.legacy/'source.jsonl';source.write_text('{"unit_test":true}\n',encoding='utf-8')
        calibration=self.legacy/'calibration.json';calibration.write_text('{"unit_test":true}\n',encoding='utf-8')
        settings={'trainer_version':1,'data_directory':str(self.data),'checkpoint_directory':str(self.C),
            'initial_checkpoint':str(initial),'initial_sha256':digest(initial),'calibration':str(calibration),'calibration_sha256':digest(calibration),
            'source':str(source),'source_provenance':{'license':'CC0-1.0','local_sha256':digest(source)},'seed':2026,
            'positions':4,'shard_size':2,'epochs':2,'batch_size':2,'learning_rate':.001,
            'scheduler_spec':{'decay':.999,'minimum_multiplier':.05},'validation_size':3,'test_size':3,
            'checkpoint_every':2,'replay_size':3,'min_depth':12,'min_knodes':10,'similarity_cap':50}
        signature=hashlib.sha256(canonical(settings).encode()).hexdigest()
        self.ledger.put('settings',settings);self.ledger.put('settings_signature',signature)
        self.ledger.put('runtime_config',dict(settings,max_memory_gb=4,max_disk_gb=1,max_download_mb=16,max_decompressed_mb=128,keep_shards=False))
        frozen={}
        for split in ['validation','test']:
            rows=[{'identity':hashlib.sha256(f'{split}-{i}'.encode()).hexdigest(),'split':split} for i in range(3)]
            path=self.data/'fixed'/f'{split}.jsonl';path.write_text(''.join(canonical(r)+'\n' for r in rows),encoding='utf-8')
            frozen[split]={'ids':[r['identity'] for r in rows],'path':str(path),'sha256':digest(path)}
        self.ledger.put('fixed_sets',frozen)
        self.ledger.put('source_cursor',{'line':128,'prefix_sha256':'1'*64,'source_identity':{'location':str(source)}})
        self.old_receipt=self.C/'checkpoints'/('bundle-'+'a'*32+'.manifest.json')
        # Completed older receipts normally outlive the checkpoint retention
        # window. They remain owned names even when that bundle was pruned.
        self.ledger.db.execute('INSERT INTO shards(id,path,sha,rows,status,checkpoint) VALUES (?,?,?,?,?,?)',
            (1,str(self.data/'shards/shard-00000001.jsonl'),'0'*64,2,'done',str(self.old_receipt)))
        shard=self.data/'shards/shard-00000002.jsonl'
        rows=[{'identity':hashlib.sha256(f'train-{i}'.encode()).hexdigest(),'split':'train'} for i in range(2)]
        shard.write_text(''.join(canonical(r)+'\n' for r in rows),encoding='utf-8')
        self.ledger.db.execute('INSERT INTO shards(id,path,sha,rows,status) VALUES (?,?,?,?,?)',(2,str(shard),digest(shard),2,'prepared'))
        for i,r in enumerate(rows):self.ledger.db.execute('INSERT INTO records(id,family,bucket,phase,split,payload,shard) VALUES (?,?,?,?,?,?,?)',
            (r['identity'],r['identity'],'bucket','opening','train',canonical(r),2))
        state=dict(basic,trainer_schema=1,settings=settings,settings_signature=signature,seed=2026,
            scheduler={'last_epoch':1},replay=[{'identity':'past','split':'train','value_weight':.2}],
            progress={'completed':[1],'active':2,'epoch':0,'offset':1,'permutation':[r['identity'] for r in rows],'positions_processed':2})
        self.state,self.manifest=save_bundle(self.ledger,self.C,model,state,ResourceGuard([self.legacy],1,4),{'kind':'unit','global_step':1})
        self.ledger.event({'commit_seq':0,'checkpoint':str(self.old_receipt),'kind':'historical'})
        self.ledger.close();self.ledger=None
        self.destination=self.root/'new-arc-run'
    def tearDown(self):
        if self.ledger is not None:self.ledger.close()
        self.temp.cleanup()
    def connect(self):return sqlite3.connect((self.destination/'data/ledger.sqlite3').as_uri()+'?mode=ro&immutable=1',uri=True)
    def meta(self,db,key):return json.loads(db.execute('SELECT v FROM meta WHERE k=?',(key,)).fetchone()[0])
    def test_real_subprocess_import_keeps_source_bytes_and_all_training_state(self):
        before=tree_hashes(self.legacy)
        result=subprocess.run([sys.executable,str(ROOT/'tools/import_legacy.py'),
            '--source-data-directory',str(self.data),'--source-checkpoint-directory',str(self.C),
            '--destination-root',str(self.destination)],capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr);report=json.loads(result.stdout)
        self.assertTrue(report['verified_source_unchanged']);self.assertEqual(before,tree_hashes(self.legacy))
        self.assertFalse((self.destination/'import-incomplete.json').exists())
        imported,mirror=verify_bundle(report['imported_head_manifest']);self.assertFalse(mirror)
        self.assertEqual(imported['progress'],self.state['progress']);self.assertEqual(imported['replay'],self.state['replay'])
        self.assertEqual(imported['settings'],self.state['settings']);self.assertEqual(imported['scheduler'],self.state['scheduler'])
        self.assertTrue(torch.equal(imported['generator'],self.state['generator']))
        self.assertTrue(all(torch.equal(t,imported['model'][name]) for name,t in self.state['model'].items()))
        for state_id,values in self.state['optimizer']['state'].items():
            for key,value in values.items():
                if isinstance(value,torch.Tensor):self.assertTrue(torch.equal(value,imported['optimizer']['state'][state_id][key]))
        old=read_manifest(self.manifest);new=read_manifest(report['imported_head_manifest'])
        self.assertEqual({k:v['sha256'] for k,v in old['files'].items()},{k:v['sha256'] for k,v in new['files'].items()})
        with closing(self.connect()) as db:
            marker=self.meta(db,'legacy_import');self.assertEqual(marker['status'],'pending_arc_migration');self.assertTrue(marker['copy_complete'])
            self.assertEqual(self.meta(db,'settings'),self.state['settings'])
            runtime=self.meta(db,'runtime_config')
            self.assertEqual(runtime['data_directory'],str(self.destination/'data'))
            self.assertEqual(runtime['checkpoint_directory'],str(self.destination/'candidate'))
            for key,value in EXECUTION.items():self.assertEqual(runtime[key],value)
            self.assertEqual(runtime['source'],self.state['settings']['source'])
            self.assertNotEqual(self.meta(db,'durable_owner_token'),json.loads((self.C/'checkpoint-owner.json').read_text())['ledger_token'])
            receipt=db.execute('SELECT checkpoint FROM shards WHERE id=1').fetchone()[0]
            self.assertEqual(Path(receipt).parent,self.destination/'candidate/checkpoints')
            self.assertEqual(self.meta(db,'source_cursor')['line'],128)
    def test_dry_run_creates_no_destination_or_source_sidecars(self):
        before=tree_hashes(self.legacy)
        report=import_legacy(self.data,self.C,self.destination,dry_run=True)
        self.assertTrue(report['dry_run']);self.assertFalse(self.destination.exists());self.assertEqual(before,tree_hashes(self.legacy))
    def test_nonempty_wal_is_refused_without_source_checkpoint_or_destination(self):
        wal=Path(str(self.data/'ledger.sqlite3')+'-wal');wal.write_bytes(b'live-uncheckpointed-data')
        before=tree_hashes(self.legacy)
        with self.assertRaisesRegex(RuntimeError,'nonempty WAL'):import_legacy(self.data,self.C,self.destination)
        self.assertEqual(before,tree_hashes(self.legacy));self.assertFalse(self.destination.exists())
    def test_destination_must_be_new_and_disjoint_from_original(self):
        before=tree_hashes(self.legacy)
        with self.assertRaisesRegex(ValueError,'contain'):import_legacy(self.data,self.C,self.data/'copy')
        self.destination.mkdir();(self.destination/'user-file.txt').write_text('preserve')
        with self.assertRaisesRegex(ValueError,'new or empty'):import_legacy(self.data,self.C,self.destination)
        self.assertEqual((self.destination/'user-file.txt').read_text(),'preserve');self.assertEqual(before,tree_hashes(self.legacy))
    def test_corrupt_fixed_file_and_copy_budget_are_refused_before_writes(self):
        with self.assertRaisesRegex(RuntimeError,'MaxCopyGB'):import_legacy(self.data,self.C,self.destination,max_copy_gb=.000001)
        self.assertFalse(self.destination.exists())
        path=self.data/'fixed/test.jsonl';path.write_text('corrupt')
        before=tree_hashes(self.legacy)
        with self.assertRaisesRegex(ValueError,'holdout'):import_legacy(self.data,self.C,self.destination)
        self.assertEqual(before,tree_hashes(self.legacy));self.assertFalse(self.destination.exists())
    def test_copy_failure_retains_nonreusable_partial_destination_and_source(self):
        before=tree_hashes(self.legacy);real_copy=shutil.copy2;calls=0
        def fail(source,target):
            nonlocal calls
            calls+=1
            if calls==2:raise OSError('injected copy interruption')
            return real_copy(source,target)
        with patch('tools.import_legacy.shutil.copy2',side_effect=fail):
            with self.assertRaisesRegex(OSError,'copy interruption'):import_legacy(self.data,self.C,self.destination)
        self.assertEqual(before,tree_hashes(self.legacy))
        self.assertEqual(json.loads((self.destination/'import-incomplete.json').read_text())['phase'],'failed')
        with self.assertRaisesRegex(ValueError,'new or empty'):import_legacy(self.data,self.C,self.destination)
    def test_primary_corruption_uses_byte_identical_mirror_without_repairing_source(self):
        primary=Path(read_manifest(self.manifest)['files']['checkpoint']['path']);primary.write_bytes(b'corrupt-primary')
        before=tree_hashes(self.legacy)
        report=import_legacy(self.data,self.C,self.destination)
        self.assertTrue(report['source_head_used_mirror']);self.assertEqual(before,tree_hashes(self.legacy))
        imported,mirror=verify_bundle(report['imported_head_manifest']);self.assertTrue(mirror)
        self.assertEqual(imported['progress'],self.state['progress'])
    def test_late_verification_failure_never_publishes_migration_marker(self):
        before=tree_hashes(self.legacy)
        def verify(path):
            if Path(path).is_relative_to(self.destination):raise RuntimeError('injected destination verification failure')
            return verify_bundle(path)
        with patch('tools.import_legacy.verify_bundle',side_effect=verify):
            with self.assertRaisesRegex(RuntimeError,'destination verification failure'):import_legacy(self.data,self.C,self.destination)
        with closing(self.connect()) as db:
            self.assertIsNone(db.execute("SELECT v FROM meta WHERE k='legacy_import'").fetchone())
            self.assertIsNotNone(db.execute("SELECT v FROM meta WHERE k='legacy_import_failed'").fetchone())
        self.assertEqual(before,tree_hashes(self.legacy))

if __name__=='__main__':unittest.main()
