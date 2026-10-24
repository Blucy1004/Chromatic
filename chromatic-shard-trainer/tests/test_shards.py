"""Transactional rotating-shard regressions using the bounded official fixture."""
from collections import Counter
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
sys.dont_write_bytecode=True

from shardtrain.anchors import opening_anchors
from shardtrain.calibration import Calibration
from shardtrain.quality import identity_for,parse_record
from shardtrain.shards import ShardManager,InsufficientData
from shardtrain.storage import Ledger,ResourceGuard,digest
from shardtrain.streaming import SourceSpec
import chess


class ShardsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures=PROJECT/'fixtures'
        cls.source=cls.fixtures/'official-evaluations.jsonl.zst'
        if not cls.source.is_file() or not (cls.fixtures/'calibration.json').is_file():
            raise unittest.SkipTest('Run tools/prepare_sample.py and independent calibration preparation first')
        cls.anchors=opening_anchors(cls.fixtures,13,2026)
        cls.calibration=Calibration.load(cls.fixtures/'calibration.json')

    def setUp(self):
        base=Path(os.environ.get('CHROMATIC_TEST_TEMP',PROJECT.parents[1]/'work'/'shard-tests'))
        base.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base)
        self.root=Path(self.temp.name);self.ledger=Ledger(self.root/'ledger.sqlite3')
        self.config={'positions':100,'shard_size':30,'seed':2026,'validation_size':9,'test_size':9,
                     'min_depth':12,'min_knodes':10,'similarity_cap':50,'max_download_mb':16,
                     'max_decompressed_mb':128,'checkpoint_directory':str(self.root/'checkpoints')}
        self.guard=ResourceGuard([self.root],.25,2)
        self.manager=self.make_manager()

    def make_manager(self,source=None):
        return ShardManager(self.ledger,self.root/'data',self.guard,self.config,self.calibration,
                            SourceSpec(str(source or self.source),version='bounded-official-shard-test'),self.anchors)

    def tearDown(self):
        self.manager.close();self.ledger.close();self.temp.cleanup()

    def phase_counts(self,rows):return Counter(row['phase'] for row in rows)

    def test_balanced_frozen_sets_root_and_resume(self):
        valid,test=self.manager.prepare_fixed()
        self.assertEqual(self.phase_counts(valid),{'opening':3,'middlegame':3,'endgame':3})
        self.assertEqual(self.phase_counts(test),{'opening':3,'middlegame':3,'endgame':3})
        frozen=self.ledger.get('fixed_sets');first=self.manager.next_shard(100)
        records=self.ledger.load_ids(first['ids'])
        self.assertEqual(self.phase_counts(records),{'opening':10,'middlegame':10,'endgame':10})
        self.assertIn(identity_for(chess.Board()),first['ids'])
        self.assertTrue(all(row['split']=='train' for row in records))
        self.assertFalse({r['family'] for r in records}&{r['family'] for r in valid+test})
        self.assertFalse({r['family'] for r in records+valid+test}&self.calibration.excluded_families)
        for phase in ('opening','middlegame','endgame'):
            self.assertLessEqual(self.ledger.count('train',phase,unassigned=True),60)
        self.assertEqual(self.ledger.db.execute('SELECT family FROM records GROUP BY family HAVING count(DISTINCT split)>1').fetchall(),[])
        self.manager.close();self.manager=self.make_manager()
        valid2,test2=self.manager.prepare_fixed();again=self.manager.next_shard(100)
        self.assertEqual([r['identity'] for r in valid],[r['identity'] for r in valid2])
        self.assertEqual([r['identity'] for r in test],[r['identity'] for r in test2])
        self.assertEqual(self.ledger.get('fixed_sets'),frozen)
        self.assertEqual(first,again)

    def test_corrupted_prepared_file_is_retained_and_refused(self):
        self.manager.prepare_fixed();first=self.manager.next_shard(100)
        cursor=self.ledger.get('source_cursor')
        with Path(first['path']).open('ab') as stream:stream.write(b'CORRUPTED')
        with self.assertRaisesRegex(RuntimeError,'checksum mismatch'):self.manager.next_shard(100)
        self.assertTrue(Path(first['path']).exists())
        self.assertEqual(self.ledger.get('source_cursor'),cursor)
        self.assertEqual(self.ledger.db.execute('SELECT count(*) FROM shards').fetchone()[0],1)

    def test_missing_payloads_restore_exact_assigned_order(self):
        self.manager.prepare_fixed();first=self.manager.next_shard(100)
        with self.ledger.transaction():
            self.ledger.db.executemany('DELETE FROM records WHERE id=?',[(first['ids'][i],) for i in (0,7,15)])
        restored=self.manager.next_shard(100)
        self.assertEqual(restored,first)
        self.assertEqual([i for i,_ in self.ledger.rows(shard=first['id'])],first['ids'])
        self.assertTrue(all(r['split']=='train' for _,r in self.ledger.rows(shard=first['id'])))

    def mark_done(self,shard):
        with self.ledger.transaction():
            self.ledger.db.execute("UPDATE shards SET status='done',checkpoint=? WHERE id=?",('verified-boundary-'+str(shard['id']),shard['id']))

    def test_cleanup_requires_completion_and_keeps_rollback_shard(self):
        self.manager.prepare_fixed();first=self.manager.next_shard(100);self.mark_done(first)
        self.manager.cleanup([1],False);self.assertTrue(Path(first['path']).exists())
        second=self.manager.next_shard(70)
        with self.assertRaisesRegex(RuntimeError,'receipt is not committed'):self.manager.cleanup([1,2],False)
        self.assertTrue(Path(first['path']).exists())
        self.mark_done(second);seen=self.ledger.db.execute('SELECT count(*) FROM seen').fetchone()[0]
        self.manager.cleanup([1,2],False)
        self.assertFalse(Path(first['path']).exists());self.assertTrue(Path(second['path']).exists())
        self.assertEqual(list(self.ledger.rows(shard=1)),[])
        self.assertEqual(self.ledger.db.execute('SELECT count(*) FROM seen').fetchone()[0],seen)
        self.assertEqual(len(self.manager.prepare_fixed()[0]),9)
        third=self.manager.next_shard(40);self.mark_done(third)
        self.manager.cleanup([1,2,3],True)
        self.assertTrue(Path(second['path']).exists());self.assertTrue(Path(third['path']).exists())

    def test_rollback_receipt_reuses_existing_shard_id(self):
        self.manager.prepare_fixed();first=self.manager.next_shard(100);self.mark_done(first)
        second=self.manager.next_shard(70);self.mark_done(second)
        self.manager.cleanup([1],True)
        self.assertEqual(self.ledger.db.execute('SELECT status FROM shards WHERE id=2').fetchone()[0],'prepared')
        for status in ('prepared','ready','training'):
            self.ledger.db.execute('UPDATE shards SET status=? WHERE id=2',(status,))
            self.assertEqual(self.manager.next_shard(70),second)

    def test_failed_ingestion_transaction_does_not_advance_cursor(self):
        initial_records=self.ledger.db.execute('SELECT count(*) FROM records').fetchone()[0]
        original=self.manager._insert;calls=0
        def injected(row,counts):
            nonlocal calls
            calls+=1
            if calls==10:raise RuntimeError('injected ingestion failure')
            return original(row,counts)
        with patch.object(self.manager,'_insert',side_effect=injected):
            with self.assertRaisesRegex(RuntimeError,'injected'):self.manager.prepare_fixed()
        self.assertIsNone(self.ledger.get('source_cursor'))
        self.assertEqual(self.ledger.db.execute('SELECT count(*) FROM records').fetchone()[0],initial_records)
        self.manager.close();self.manager=self.make_manager()
        valid,test=self.manager.prepare_fixed()
        self.assertEqual((len(valid),len(test)),(9,9))
        self.assertEqual(self.ledger.get('source_cursor')['line'],256)

    def test_out_of_tree_shard_path_is_never_deleted(self):
        self.manager.prepare_fixed();first=self.manager.next_shard(100);self.mark_done(first)
        second=self.manager.next_shard(70);self.mark_done(second)
        outside=self.root/'important-user-file.jsonl';outside.write_text('preserve',encoding='utf-8')
        self.ledger.db.execute('UPDATE shards SET path=? WHERE id=1',(str(outside),))
        with self.assertRaisesRegex(RuntimeError,'outside'):self.manager.cleanup([1,2],False)
        self.assertEqual(outside.read_text(encoding='utf-8'),'preserve')
        inside=self.root/'data/shards/important-user-file.jsonl';inside.write_text('preserve-inside',encoding='utf-8')
        self.ledger.db.execute('UPDATE shards SET path=? WHERE id=1',(str(inside),))
        with self.assertRaisesRegex(RuntimeError,'owned shard ID'):self.manager.cleanup([1,2],False)
        self.assertEqual(inside.read_text(encoding='utf-8'),'preserve-inside')

    def test_changed_completed_shard_is_not_deleted(self):
        self.manager.prepare_fixed();first=self.manager.next_shard(100);self.mark_done(first)
        second=self.manager.next_shard(70);self.mark_done(second)
        target=Path(first['path'])
        with target.open('ab') as out:out.write(b'changed-after-completion')
        with self.assertRaisesRegex(RuntimeError,'Completed shard changed'):self.manager.cleanup([1,2],False)
        self.assertTrue(target.exists());self.assertTrue(Path(second['path']).exists())

    def test_private_orphan_temps_are_reclaimed_without_touching_user_files(self):
        self.manager.prepare_fixed();first=self.manager.next_shard(100);self.manager.close()
        data=self.root/'data'
        owned=[data/'source-spool/.source-frame-ab12_cd.jsonl.tmp',
               data/'shards/shard-00000009.jsonl.tmp',data/'fixed/validation.jsonl.tmp']
        user=data/'source-spool/user-important-file.tmp';user.write_text('preserve',encoding='utf-8')
        for path in owned:path.write_text('uncommitted',encoding='utf-8')
        self.manager=self.make_manager()
        self.assertTrue(all(not path.exists() for path in owned))
        self.assertEqual(user.read_text(encoding='utf-8'),'preserve')
        self.assertTrue(Path(first['path']).is_file())
        self.assertEqual(self.ledger.get('quality_stats')['orphan_temps_removed'],3)


if __name__=='__main__':unittest.main()
