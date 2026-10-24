import json
from pathlib import Path
import tempfile
import time
import unittest
import ctypes
from ctypes import wintypes
from types import SimpleNamespace

from shardtrain.profiling import Profile,ResourceSampler,install_profile,span,profiled,_WindowsGpuCounters
from shardtrain.quality import parse_record

WORK=Path(__file__).resolve().parents[3]/'work'/'arc-profile-tests'

class ProfileTests(unittest.TestCase):
    def setUp(self):
        WORK.mkdir(parents=True,exist_ok=True);self.temp=tempfile.TemporaryDirectory(prefix='profile-',dir=WORK);self.root=Path(self.temp.name)
    def tearDown(self):install_profile(None);self.temp.cleanup()
    def test_nested_accounting_and_sync_are_explicit(self):
        calls=[];profile=Profile(True,self.root,synchronize=lambda:calls.append('sync'))
        with profile.span('parent',items=3):
            with profile.span('child',sync=True):time.sleep(.003)
        report=profile.write();profile.close()
        self.assertEqual(len(calls),2);self.assertEqual(report['stages']['parent']['items'],3)
        self.assertGreaterEqual(report['stages']['parent']['inclusive_seconds'],report['stages']['child']['inclusive_seconds'])
        self.assertLess(report['stages']['parent']['seconds'],report['stages']['parent']['inclusive_seconds'])
        self.assertEqual(len((self.root/'stages.jsonl').read_text(encoding='utf-8').splitlines()),2)
    def test_disabled_hooks_never_synchronize_or_change_results(self):
        def forbidden():raise AssertionError('disabled profiling synchronized')
        profile=Profile(False,synchronize=forbidden);install_profile(profile)
        @profiled('a')
        def value(x):return x+1
        with profile.span('sync',sync=True):self.assertEqual(value(2),3)
        self.assertEqual(profile.snapshot()['events'],0)
    def test_quality_hooks_do_not_change_normalized_records(self):
        raw={'fen':'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -',
             'evals':[{'depth':24,'knodes':1000,'pvs':[{'cp':12,'line':'e2e4 e7e5'}]}]}
        expected=parse_record(raw);profile=Profile(True);install_profile(profile);actual=parse_record(raw)
        self.assertEqual(expected,actual)
        self.assertTrue({'fen_validation','pv_validation','label_conversion'}.issubset(profile.snapshot()['stages']))
    def test_resource_sampler_streams_and_bounds_retained_tail(self):
        sampler=ResourceSampler(.02,'cpu',self.root,max_retained_samples=2).start()
        time.sleep(.09);summary=sampler.stop()
        self.assertGreaterEqual(summary['samples'],3);self.assertLessEqual(len(summary['retained_tail_samples']),2)
        self.assertGreater(summary['peaks']['rss_bytes'],0);self.assertFalse(summary['gpu_utilization_available'])
        self.assertIsNotNone(summary['gpu_unavailable_reason'])
        rows=[json.loads(line) for line in (self.root/'resources.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(rows),summary['samples']);self.assertGreaterEqual(rows[-1]['io_write_bytes'],0)
    def test_gpu_counter_warmup_is_missing_data_and_not_a_permanent_error(self):
        counter=_WindowsGpuCounters.__new__(_WindowsGpuCounters)
        counter.c=ctypes;counter.w=wintypes;counter.query=None
        counter.dll=SimpleNamespace(PdhCollectQueryData=lambda query:0x800007D5)
        self.assertEqual(counter.read(),(None,0));self.assertIn('not yet available',counter.last_status)

if __name__=='__main__':unittest.main()
