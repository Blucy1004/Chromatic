import copy
from pathlib import Path
import random
import tempfile
import time
import unittest

import chess
import numpy as np
import torch

from shardtrain.fastdata import EncodedCache,BatchBuffer,Prefetch,ROW_DTYPE
from shardtrain.model_adapter import batch
from shardtrain.quality import parse_record

WORK=Path(__file__).resolve().parents[3]/'work'/'arc-data-tests'

def records(count=12):
    result=[];board=chess.Board();rng=random.Random(197)
    while len(result)<count:
        legal=list(board.legal_moves)
        record=parse_record({'fen':board.fen(),'evals':[{'depth':24,'knodes':1000,
            'pvs':[{'cp':20,'line':legal[0].uci()}]}]})
        record['split']='train';result.append(record)
        board.push(rng.choice(legal))
    return result

class FastDataTests(unittest.TestCase):
    def setUp(self):
        WORK.mkdir(parents=True,exist_ok=True);self.temp=tempfile.TemporaryDirectory(prefix='data-',dir=WORK)
        self.root=Path(self.temp.name);self.records=records()
    def tearDown(self):self.temp.cleanup()
    def equal(self,actual,expected):
        for a,b in zip(actual,expected):self.assertTrue(torch.equal(a,b),f'{a.dtype}/{a.shape}')
    def test_binary_modes_match_reference_exactly_and_preserve_order(self):
        chosen=[7,0,9,1]
        for mode in ['memory','mmap','auto','none']:
            with self.subTest(mode=mode),EncodedCache(self.root/mode,mode=mode,memory_budget_bytes=1024**2) as cache:
                data=cache.prepare('shard-1',self.records,expected_source_sha='source-a')
                self.equal(data.batch(chosen),batch([self.records[i] for i in chosen]))
                self.equal(cache.get_batch([self.records[i] for i in chosen]),batch([self.records[i] for i in chosen]))
                if mode!='none':self.assertEqual(cache.stats['get_batch_misses'],0)
                data.close();self.assertEqual(cache.resident_bytes,0)
    def test_auto_uses_mmap_when_ram_budget_is_small_and_reuses_verified_file(self):
        with EncodedCache(self.root,mode='auto',memory_budget_bytes=1) as cache:
            first=cache.prepare('a',self.records);self.assertEqual(first.mode,'mmap');first.close()
            second=cache.prepare('a',self.records)
            self.assertEqual(cache.stats['builds'],1);self.assertEqual(cache.stats['disk_hits'],1)
            with self.assertRaises(IndexError):second.batch([-1])
    def test_binary_corruption_fails_without_reusing_or_deleting_it(self):
        with EncodedCache(self.root,mode='mmap') as cache:
            data=cache.prepare('a',self.records);data.close();path,_=cache._paths('a')
            with path.open('r+b') as f:f.seek(-1,2);b=f.read(1);f.seek(-1,2);f.write(bytes([b[0]^1]))
            with self.assertRaisesRegex(RuntimeError,'checksum'):cache.prepare('a',self.records)
            with self.assertRaisesRegex(RuntimeError,'retained'):cache.remove('a')
            self.assertTrue(path.exists())
    def test_label_change_invalidates_cache_even_when_identity_is_the_same(self):
        with EncodedCache(self.root,mode='mmap') as cache:
            cache.prepare('a',self.records).close()
            changed=copy.deepcopy(self.records);changed[0]['policy_weight']=.0123
            dataset=cache.prepare('a',changed)
            self.equal(dataset.batch([0]),batch(changed[:1]));self.assertEqual(cache.stats['builds'],2)
    def test_memory_budget_and_hot_cache_are_bounded(self):
        budget=ROW_DTYPE.itemsize*2
        with EncodedCache(self.root/'auto',mode='auto',memory_budget_bytes=budget,hot_cache_bytes=budget) as cache:
            cache.get_batch(self.records);cache.get_batch(self.records[-2:])
            self.assertLessEqual(cache.hot_bytes,budget);self.assertGreater(cache.stats['hot_evictions'],0)
            self.assertEqual(cache.stats['get_batch_hits'],2)
        with EncodedCache(self.root/'memory',mode='memory',memory_budget_bytes=budget) as cache:
            with self.assertRaisesRegex(RuntimeError,'budget'):cache.prepare('a',self.records)
    def test_reusable_batch_buffer_matches_and_has_stable_storage(self):
        with EncodedCache(self.root,mode='auto') as cache:
            dataset=cache.prepare('a',self.records);buffer=BatchBuffer(8)
            first=dataset.batch([0,1,2],out=buffer);address=first[0].data_ptr()
            self.equal(first,batch(self.records[:3]))
            second=dataset.batch([9,4],out=buffer);self.assertEqual(second[0].data_ptr(),address)
            self.equal(second,batch([self.records[9],self.records[4]]))
    def test_prefetch_fifo_bound_error_and_early_shutdown_without_rng_use(self):
        before=torch.get_rng_state().clone();before_random=random.getstate()
        with Prefetch(range(40),lambda x:x*x,depth=2) as producer:
            self.assertEqual(list(producer),[i*i for i in range(40)])
            self.assertLessEqual(producer.stats['max_queue'],2)
        self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertEqual(before_random,random.getstate())
        def broken(x):
            if x==4:raise ValueError('position 4')
            return x
        with Prefetch(range(8),broken,depth=2) as producer:
            self.assertEqual([next(producer) for _ in range(4)],list(range(4)))
            with self.assertRaisesRegex(ValueError,'position 4'):next(producer)
        with Prefetch(range(100000),depth=1) as producer:
            self.assertEqual(next(producer),0)
        self.assertFalse(producer.thread.is_alive())
    def test_cache_remove_requires_closed_mmap_and_never_uses_input_as_path(self):
        with EncodedCache(self.root,mode='mmap') as cache:
            dataset=cache.prepare('../../outside',self.records)
            path,_=cache._paths('../../outside');self.assertEqual(path.parent,self.root)
            with self.assertRaisesRegex(RuntimeError,'Close'):cache.remove('../../outside')
            dataset.close();cache.remove('../../outside');self.assertFalse(path.exists())

if __name__=='__main__':unittest.main()
