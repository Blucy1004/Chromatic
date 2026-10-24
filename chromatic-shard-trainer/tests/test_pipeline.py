"""Opt-in real-process kill/resume tests on the official small CC0 fixture.

CHROMATIC_RUN_PIPELINE_TESTS=1 enables the bounded integration suite. Every
training run uses 100 positions, two epochs, and CPU execution. No network
request is made and no champion path is written.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
sys.dont_write_bytecode=True


def atomic_report(path,obj):
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w',encoding='utf-8') as stream:
        json.dump(obj,stream,ensure_ascii=False,indent=2);stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


def assert_tree_equal(test,left,right,path='state'):
    import numpy as np
    import torch
    if isinstance(left,torch.Tensor):
        test.assertIsInstance(right,torch.Tensor,path)
        test.assertEqual((left.dtype,left.shape),(right.dtype,right.shape),path)
        test.assertTrue(torch.equal(left,right),path)
    elif isinstance(left,np.ndarray):
        test.assertIsInstance(right,np.ndarray,path)
        test.assertEqual((left.dtype,left.shape),(right.dtype,right.shape),path)
        test.assertTrue(np.array_equal(left,right),path)
    elif isinstance(left,dict):
        test.assertEqual(set(left),set(right),path)
        for key in left:assert_tree_equal(test,left[key],right[key],path+'.'+str(key))
    elif isinstance(left,(tuple,list)):
        test.assertEqual(type(left),type(right),path);test.assertEqual(len(left),len(right),path)
        for i,(a,b) in enumerate(zip(left,right)):assert_tree_equal(test,a,b,path+f'[{i}]')
    else:test.assertEqual(left,right,path)


@unittest.skipUnless(os.environ.get('CHROMATIC_RUN_PIPELINE_TESTS')=='1','opt-in bounded process integration suite')
class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import psutil
        process=psutil.Process();allowed=process.cpu_affinity()
        process.cpu_affinity([3 if 3 in allowed else allowed[-1]])
        fixture=PROJECT/'fixtures/official-evaluations.jsonl.zst'
        if not fixture.is_file():raise unittest.SkipTest('Run official small-sample preparation first')
        base=PROJECT.parents[1]/'work'/'pipeline-tests';base.mkdir(parents=True,exist_ok=True)
        cls.directory=Path(tempfile.mkdtemp(prefix='run-',dir=base))
        cls.report_path=PROJECT/'reports/pipeline-tests.json';cls.report_path.parent.mkdir(exist_ok=True)
        previous=json.loads(cls.report_path.read_text(encoding='utf-8')) if os.environ.get('CHROMATIC_PIPELINE_REPORT_APPEND')=='1' and cls.report_path.exists() else {}
        prior_directories=previous.get('work_directories',[previous['work_directory']] if previous.get('work_directory') else [])
        cls.report={'format':1,'scope':'100 official CC0 positions per run; real process termination; CPU',
                    'work_directory':str(cls.directory),'work_directories':prior_directories+[str(cls.directory)],
                    'cpu_affinity':process.cpu_affinity(),'tests':previous.get('tests',[]),'status':'running'}
        atomic_report(cls.report_path,cls.report)
        cls.baseline_case=cls.directory/'uninterrupted'
        cls.run_cli(cls.baseline_case)
        cls.baseline=cls.snapshot(cls.baseline_case)
        cls.report['baseline']={'step':cls.baseline['state']['step'],'progress':cls.baseline['state']['progress'],
                                'model_export_sha256':cls.baseline['export_sha256']}
        atomic_report(cls.report_path,cls.report)

    @classmethod
    def tearDownClass(cls):
        cls.report['status']='passed' if all(t.get('passed') for t in cls.report['tests']) else 'failed'
        atomic_report(cls.report_path,cls.report)

    @classmethod
    def command(cls,case,resume=False,extra=()):
        cmd=[sys.executable,'-m','shardtrain.run','--data-directory',str(case/'data')]
        if resume:cmd+=['--resume']
        else:cmd+=['--positions','100','--shard-size','25','--epochs','2','--batch-size','16',
                   '--source',str(PROJECT/'fixtures/official-evaluations.jsonl.zst'),
                   '--checkpoint-directory',str(case/'checkpoints'),'--max-download-mb','16',
                   '--max-disk-gb','.5','--max-memory-gb','2','--checkpoint-every','2']
        return cmd+list(extra)

    @classmethod
    def env(cls,hook=None):
        environment=dict(os.environ,PYTHONPATH=str(PROJECT),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
        for name in ['CHROMATIC_TEST_PAUSE_AT','CHROMATIC_TEST_PAUSE_STEP','CHROMATIC_TEST_PAUSE_SHARD']:
            environment.pop(name,None)
        if hook:
            environment.update(CHROMATIC_TEST_PAUSE_AT=hook,CHROMATIC_TEST_PAUSE_STEP='2',
                               CHROMATIC_TEST_PAUSE_SHARD='2' if hook=='before_delete' else '0')
        return environment

    @classmethod
    def run_cli(cls,case,resume=False,extra=(),expected=0):
        case.mkdir(parents=True,exist_ok=True)
        stamp=time.time_ns();log=case/f'process-{stamp}.log';start=time.perf_counter()
        with log.open('w',encoding='utf-8') as out:
            process=subprocess.run(cls.command(case,resume,extra),cwd=PROJECT,env=cls.env(),
                                   stdout=out,stderr=subprocess.STDOUT,timeout=180)
        if process.returncode!=expected:
            raise AssertionError(f'CLI return {process.returncode}, expected {expected}: '+log.read_text(encoding='utf-8',errors='replace')[-6000:])
        return {'seconds':time.perf_counter()-start,'returncode':process.returncode,'log':str(log)}

    @classmethod
    def snapshot(cls,case):
        import torch
        torch.set_num_threads(1)
        from shardtrain.durable import read_manifest,verify_bundle
        from shardtrain.storage import digest
        with sqlite3.connect(case/'data/ledger.sqlite3') as db:
            manifest=json.loads(db.execute("SELECT v FROM meta WHERE k='current_manifest'").fetchone()[0])
            fixed=json.loads(db.execute("SELECT v FROM meta WHERE k='fixed_sets'").fetchone()[0])
            checkpoints=db.execute('SELECT count(*) FROM checkpoints').fetchone()[0]
            source_cursor=json.loads(db.execute("SELECT v FROM meta WHERE k='source_cursor'").fetchone()[0])
            events=[json.loads(p) for (p,) in db.execute('SELECT payload FROM events ORDER BY id')]
        state,mirror=verify_bundle(manifest);metadata=read_manifest(manifest)
        return {'state':state,'manifest':manifest,'metadata':metadata,'mirror':mirror,
                'export_sha256':digest(Path(metadata['files']['model']['path'])),
                'fixed':fixed,'checkpoints':checkpoints,'source_cursor':source_cursor,'events':events,
                'status':json.loads((case/'checkpoints/status.json').read_text(encoding='utf-8'))}

    def compare(self,actual):
        expected=self.baseline
        for key in ['config','model','optimizer','scheduler','generator','rng','progress','replay','holdout_digest','step','source_cursor']:
            assert_tree_equal(self,expected['state'][key],actual['state'][key],key)
        self.assertEqual(expected['export_sha256'],actual['export_sha256'])
        self.assertEqual(expected['source_cursor'],actual['source_cursor'])
        for split in ['validation','test']:
            self.assertEqual(expected['fixed'][split]['ids'],actual['fixed'][split]['ids'])
            self.assertEqual(expected['fixed'][split]['sha256'],actual['fixed'][split]['sha256'])
        self.assertEqual(actual['status']['status'],'completed')
        completed=[e for e in actual['events'] if e.get('kind')=='shard_completed']
        self.assertEqual(len(completed),4)
        self.assertEqual(len({e['commit_seq'] for e in completed}),4)

    def record(self,name,operation):
        self.report['tests']=[entry for entry in self.report['tests'] if entry['name']!=name]
        entry={'name':name,'passed':False};self.report['tests'].append(entry)
        try:
            details=operation();entry.update(details or {});entry['passed']=True
        except BaseException as exc:
            entry['error']=f'{type(exc).__name__}: {exc}';raise
        finally:atomic_report(self.report_path,self.report)

    def test_01_completed_resume_has_no_new_training(self):
        def operation():
            before=self.snapshot(self.baseline_case);run=self.run_cli(self.baseline_case,True);after=self.snapshot(self.baseline_case)
            self.compare(after);self.assertEqual(before['manifest'],after['manifest'])
            self.assertEqual(before['checkpoints'],after['checkpoints'])
            return dict(run,global_step=after['state']['step'],no_new_commit=True)
        self.record('completed_resume_no_duplicate_training',operation)

    def killed_resume(self,hook):
        case=self.directory/('kill-'+hook);case.mkdir()
        log=case/'killed-process.log';marker=case/'checkpoints/test-marker.json';start=time.perf_counter()
        with log.open('w',encoding='utf-8') as out:
            process=subprocess.Popen(self.command(case),cwd=PROJECT,env=self.env(hook),stdout=out,stderr=subprocess.STDOUT)
            try:
                while not marker.is_file():
                    if process.poll() is not None:raise AssertionError('Process ended before kill marker: '+log.read_text(encoding='utf-8',errors='replace')[-4000:])
                    if time.perf_counter()-start>90:raise AssertionError('Timed out waiting for stable kill point')
                    time.sleep(.05)
                paused=json.loads(marker.read_text(encoding='utf-8'))
                self.assertEqual(paused['hook'],hook)
                # A Windows venv launcher may have a different PID from the
                # interpreter. The fresh private marker identifies our child.
                os.kill(int(paused['pid']),signal.SIGTERM)
                process.wait(timeout=15)
            finally:
                if process.poll() is None:process.kill();process.wait(timeout=15)
        resumed=self.run_cli(case,True);snapshot=self.snapshot(case);self.compare(snapshot)
        self.assertFalse(any((case/'data/source-spool').glob('.source-frame-*.jsonl.tmp')))
        return {'hook':hook,'marker':paused,'forced_exit':process.returncode,'resume':resumed,
                'global_step':snapshot['state']['step'],'exact_model_optimizer_scheduler_rng_progress_replay':True}

    def test_02_kill_after_checkpoint_before_ledger(self):
        self.record('kill_checkpoint_written',lambda:self.killed_resume('checkpoint_written'))

    def test_03_kill_after_authoritative_commit(self):
        self.record('kill_ledger_committed',lambda:self.killed_resume('ledger_committed'))

    def test_04_kill_before_old_shard_delete(self):
        self.record('kill_before_delete',lambda:self.killed_resume('before_delete'))

    def test_05_corruption_mirror_then_previous_checkpoint(self):
        def operation():
            case=self.directory/'corruption';self.run_cli(case);initial=self.snapshot(case)
            metadata=initial['metadata'];Path(metadata['files']['checkpoint']['path']).write_bytes(b'CORRUPTED PRIMARY')
            mirror_resume=self.run_cli(case,True);mirror=self.snapshot(case);self.compare(mirror)
            self.assertTrue(mirror['mirror'])
            self.assertTrue(mirror['status']['recovered'])
            Path(metadata['files']['checkpoint']['path']).write_bytes(b'CORRUPTED PRIMARY AGAIN')
            Path(metadata['files']['mirror']['path']).write_bytes(b'CORRUPTED MIRROR')
            previous_resume=self.run_cli(case,True);previous=self.snapshot(case);self.compare(previous)
            self.assertNotEqual(previous['manifest'],initial['manifest'])
            return {'mirror_resume':mirror_resume,'previous_checkpoint_resume':previous_resume,
                    'exact_model_optimizer_scheduler_rng_progress_replay':True,
                    'corrupt_latest_bundle_recovered_without_deleted_shard_dependency':True}
        self.record('corruption_mirror_and_previous_checkpoint',operation)

    def test_06_resume_config_mismatch_preserves_head(self):
        def operation():
            before=self.snapshot(self.baseline_case)
            run=self.run_cli(self.baseline_case,True,extra=['--learning-rate','.002'],expected=1)
            after=self.snapshot(self.baseline_case)
            self.assertEqual(before['manifest'],after['manifest'])
            self.assertEqual(before['export_sha256'],after['export_sha256'])
            self.assertIn('settings/source differ',Path(run['log']).read_text(encoding='utf-8'))
            return dict(run,checkpoint_head_unchanged=True)
        self.record('resume_config_mismatch_rejected',operation)


if __name__=='__main__':unittest.main(verbosity=2)
