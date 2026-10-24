from pathlib import Path
import importlib.util
import sys
import json
import os
import subprocess
import time

path=Path(__file__).resolve().parents[1]/'tools/benchmark_pipeline.py'
spec=importlib.util.spec_from_file_location('arc_benchmark_pipeline',path)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)

def test_plan_covers_devices_before_repeats_and_never_100k():
    all_cases=module.make_cases('all',3)
    assert all(c.positions<=10000 for c in all_cases)
    assert len({c.key for c in all_cases})==len(all_cases)
    first_repeat=[c for c in all_cases if c.positions==10000 and c.implementation=='arc' and not c.profile and c.repeat==1]
    assert len(first_repeat)==7
    assert max(all_cases.index(c) for c in first_repeat)<min(i for i,c in enumerate(all_cases) if c.repeat>1)
    assert len([c for c in all_cases if c.implementation=='baseline'])==3

def test_baseline_command_read_only_and_new_outputs(tmp_path):
    case=module.Case('baseline',100,implementation='baseline')
    old=tmp_path/'old';python=tmp_path/'python.exe';source=old/'fixtures/source.zst'
    argv,cwd,env=module.build_command(case,python,old,source,tmp_path/'arc/data',tmp_path/'arc/candidate')
    assert cwd==old and env['PYTHONDONTWRITEBYTECODE']=='1'
    assert '--device' not in argv and argv[argv.index('--shard-size')+1]=='25'
    assert argv[argv.index('--data-directory')+1]==str(tmp_path/'arc/data')
    newer=module.Case('new',10000,device='xpu',profile=True)
    argv,cwd,env=module.build_command(newer,python,old,source,tmp_path/'arc/data',tmp_path/'arc/candidate')
    assert '--device' in argv and '--profile' in argv and cwd==module.ROOT

def test_summary_does_not_invent_unprofiled_stage_times():
    base=dict(name='case',positions=100,implementation='arc',device='cpu',precision='fp32',threads=1,batch_size=32,
              cache_mode='auto',profile=False,repeat=1,status='ok',wall_seconds=10.,holdout_digest='same',global_step=12,
              pure_training_seconds=2.,data_processing_seconds=None,validation_seconds=None,checkpoint_seconds=None)
    summary=module.summarize([base,{**base,'repeat':2,'wall_seconds':12.}])['case']
    assert summary['wall_seconds_median']==11.
    assert summary['data_processing_seconds_median'] is None
    assert summary['pure_training_seconds_median']==2.

def test_process_tree_follows_allocating_child_and_retains_its_io(tmp_path):
    """The launcher stays alive after its real worker exits, like a redirector."""
    marker=tmp_path/'child-ready.json';release=tmp_path/'release-worker';finished=tmp_path/'child-finished'
    worker=tmp_path/'worker.py';launcher=tmp_path/'launcher.py';payload=tmp_path/'written.bin'
    worker.write_text('''from pathlib import Path
import json,os,sys,time
marker,release,payload=map(Path,sys.argv[1:])
allocation=bytearray(48*1024*1024)
for i in range(0,len(allocation),4096):allocation[i]=1
with payload.open('wb') as f:
    f.write(bytes(2*1024*1024));f.flush();os.fsync(f.fileno())
marker.write_text(json.dumps({'pid':os.getpid()}))
while not release.exists():time.sleep(.01)
''',encoding='utf-8')
    launcher.write_text('''from pathlib import Path
import subprocess,sys,time
worker,marker,release,payload,finished=sys.argv[1:]
child=subprocess.Popen([sys.executable,worker,marker,release,payload])
child.wait();Path(finished).touch()
time.sleep(1)
''',encoding='utf-8')
    process=subprocess.Popen([sys.executable,str(launcher),str(worker),str(marker),str(release),str(payload),str(finished)])
    sampler=module.ProcessTreeSampler(process.pid);seen=[]
    try:
        deadline=time.monotonic()+8
        while not marker.exists() and time.monotonic()<deadline:
            seen.append(sampler.sample());time.sleep(.03)
        assert marker.exists()
        child_pid=json.loads(marker.read_text())['pid']
        for _ in range(3):seen.append(sampler.sample());time.sleep(.03)
        assert any(child_pid in row['live_pids'] for row in seen)
        assert max(row['rss_bytes'] for row in seen)>48*1024*1024
        assert sampler.snapshot()['tracked_processes']>=2
        before=max(row['io_write_bytes'] for row in seen)
        assert before>=2*1024*1024
        release.touch()
        deadline=time.monotonic()+5
        while not finished.exists() and time.monotonic()<deadline:
            sampler.sample();time.sleep(.03)
        assert finished.exists()
        after=sampler.sample()
        assert child_pid not in after['live_pids']
        assert after['io_write_bytes']>=before
        assert sampler.snapshot()['invalid_parent_launcher_sample'] is False
    finally:
        release.touch(exist_ok=True);module.terminate_tree(process)

def test_internal_training_log_rss_is_readable_without_launcher_metrics(tmp_path):
    logs=tmp_path/'logs';logs.mkdir()
    (logs/'training.csv').write_text('memory_rss_bytes,kind\n500000000,checkpoint\n650000000,complete\n',encoding='utf-8')
    assert module.training_log_rss_peak(tmp_path)==650000000

def test_selected_flag_is_arc_only_and_memory_override_reaches_both(tmp_path):
    common=(tmp_path/'python.exe',tmp_path/'old',tmp_path/'source.zst',tmp_path/'data',tmp_path/'candidate')
    arc=module.Case('selected-arc',1000,device='xpu')
    argv,_,_=module.build_command(arc,*common,validation_mode='selected',max_memory_gb=8)
    assert argv[argv.index('--validation-mode')+1]=='selected'
    assert argv[argv.index('--max-memory-gb')+1]=='8'
    baseline=module.Case('old-baseline',1000,implementation='baseline')
    argv,_,_=module.build_command(baseline,*common,validation_mode='selected',max_memory_gb=8)
    assert '--validation-mode' not in argv
    assert argv[argv.index('--max-memory-gb')+1]=='8'
    default,_,_=module.build_command(arc,*common)
    assert default[default.index('--validation-mode')+1]=='full'
    assert default[default.index('--max-memory-gb')+1]=='4'
