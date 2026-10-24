"""Sequential, bounded end-to-end baseline/ARC experiments on a local corpus.

Fresh per-case ledgers and encoded caches; the OS file cache is not flushed.
No downloads, 100k run, parallel trainers, deletion, or champion replacement.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass,asdict
from datetime import datetime,timezone
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
_ACTIVE_PROCESS=None


@dataclass(frozen=True)
class Case:
    name:str
    positions:int
    implementation:str='arc'
    device:str='cpu'
    precision:str='fp32'
    threads:int=1
    batch_size:int=32
    cache_mode:str='auto'
    profile:bool=False
    repeat:int=1

    @property
    def key(self):return f'{self.name}-rep{self.repeat}'


def make_cases(suite='all',repeats=3):
    if suite not in ['small','main','profile','all']:raise ValueError('Unknown benchmark suite')
    if not 1<=repeats<=3:raise ValueError('Repeats must be between 1 and 3')
    small=[Case(f'baseline-cpu1-{n}',n,implementation='baseline') for n in [100,1000,10000]]
    small += [Case('arc-cpu1-100',100),Case('arc-xpu-fp32-100',100,device='xpu')]
    small += [Case('arc-cpu1-1000',1000),Case('arc-xpu-fp32-1000',1000,device='xpu'),
              Case('arc-xpu-bf16-1000',1000,device='xpu',precision='bf16')]
    for batch in [256,512]:
        small += [Case(f'arc-cpu8-batch{batch}-1000',1000,threads=8,batch_size=batch),
                  Case(f'arc-xpu-fp32-batch{batch}-1000',1000,device='xpu',batch_size=batch)]
    main=[Case(f'arc-cpu{threads}-10000',10000,threads=threads) for threads in [1,2,4,8]]
    main += [Case('arc-xpu-fp32-10000',10000,device='xpu'),Case('arc-xpu-bf16-10000',10000,device='xpu',precision='bf16'),
             Case('arc-xpu-resident-10000',10000,device='xpu',cache_mode='device')]
    profile=[Case('profile-cpu1-10000',10000,profile=True),Case('profile-xpu-fp32-10000',10000,device='xpu',profile=True)]
    # Cover all devices once before spending remaining time on repeats.
    if suite=='small':return small
    if suite=='profile':return profile
    repeated=[Case(**{**asdict(case),'repeat':rep}) for rep in range(1,repeats+1) for case in main]
    if suite=='main':return repeated
    return small+[Case(**{**asdict(c),'repeat':1}) for c in main]+profile+[c for c in repeated if c.repeat>1]


def read_json(path,default=None):
    path=Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else default


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk:=stream.read(1024*1024):h.update(chunk)
    return h.hexdigest()


def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


class ProcessTreeSampler:
    """Follow venv redirectors into the actual interpreter, keyed by PID+birth.

    CPU/I/O are sums of each observed process's lifetime counters. Last known
    counters remain after a child exits. RSS sums only processes still alive;
    shared pages across the launcher and interpreter can be counted twice.
    """
    IO_KEYS=('io_read_bytes','io_write_bytes','io_read_operations','io_write_operations')

    def __init__(self,pid):
        import psutil
        self.psutil=psutil;self.pid=int(pid);self.started=time.perf_counter();self.last_time=self.started
        self.cache={};self.samples=0;self.peak_rss=0;self.previous_cpu=0.;self.last_row=None;self.errors=[]
        self.root_create_time=None
        try:
            root_process=psutil.Process(self.pid);self.root_create_time=root_process.create_time();self._remember(root_process)
        except psutil.Error:pass

    def _remember(self,process):
        born=process.create_time();key=(process.pid,born)
        self.cache.setdefault(key,{'process':process,'cpu_seconds':0.,**dict.fromkeys(self.IO_KEYS,0)})

    def sample(self):
        p=self.psutil;now=time.perf_counter()
        # Walk known live ancestors as well as the original launcher. A launcher
        # can exit before a descendant, and a descendant may create new workers.
        try:
            root_process=p.Process(self.pid)
            if self.root_create_time is not None and root_process.create_time()==self.root_create_time:self._remember(root_process)
        except p.Error:pass
        for state in list(self.cache.values()):
            process=state['process']
            try:
                if not process.is_running():continue
                for child in process.children(recursive=True):self._remember(child)
            except p.NoSuchProcess:continue
            except p.Error as exc:
                if len(self.errors)<8:self.errors.append(f'{type(exc).__name__}: {exc}')
        rss=0;live=[];root_rss=0
        for (pid,born),state in list(self.cache.items()):
            process=state['process']
            try:
                if not process.is_running() or process.create_time()!=born:continue
                with process.oneshot():
                    cpu=process.cpu_times();memory=process.memory_info();io=process.io_counters()
                state['cpu_seconds']=max(state['cpu_seconds'],float(cpu.user+cpu.system))
                observed={'io_read_bytes':io.read_bytes,'io_write_bytes':io.write_bytes,
                          'io_read_operations':io.read_count,'io_write_operations':io.write_count}
                for key,value in observed.items():state[key]=max(state[key],int(value))
                rss+=memory.rss;live.append(pid)
                if pid==self.pid:root_rss=memory.rss
            except p.NoSuchProcess:continue
            except p.Error as exc:
                if len(self.errors)<8:self.errors.append(f'{type(exc).__name__}: {exc}')
        cpu_total=sum(state['cpu_seconds'] for state in self.cache.values())
        cpu_percent=max(0.,cpu_total-self.previous_cpu)*100/max(now-self.last_time,1e-6)
        row={'elapsed_seconds':now-self.started,'rss_bytes':rss,'cpu_process_percent':cpu_percent,
             'cpu_seconds_total_last_known':cpu_total,'root_launcher_rss_bytes':root_rss,
             **{key:sum(state[key] for state in self.cache.values()) for key in self.IO_KEYS},
             'live_pids':sorted(live),'tracked_processes':len(self.cache),'live_processes':len(live),
             'sampling_scope':'process_tree_v2'}
        self.samples+=1;self.peak_rss=max(self.peak_rss,rss);self.last_time=now;self.previous_cpu=cpu_total;self.last_row=row
        return row

    def snapshot(self):
        row=self.last_row or {};elapsed=time.perf_counter()-self.started
        return {'samples':self.samples,'rss_peak_bytes':self.peak_rss,
            'cpu_process_percent_mean':row.get('cpu_seconds_total_last_known',0.)*100/max(elapsed,1e-6),
            'cpu_seconds_total_last_known':row.get('cpu_seconds_total_last_known',0.),
            **{key:row.get(key,0) for key in self.IO_KEYS},'tracked_processes':len(self.cache),
            'process_identities':[{'pid':pid,'create_time':born} for pid,born in self.cache],
            'sampling_scope':'process_tree_v2','invalid_parent_launcher_sample':False,
            'memory_accounting':'Sum of live process RSS; launcher/interpreter shared pages may be counted twice. This is not unique physical memory.',
            'counter_accounting':'Observed per-PID lifetime CPU/I/O counters; exited children retain last known values. Activity between the last sample and exit can be missed.',
            'errors':list(self.errors)}


def training_log_rss_peak(candidate):
    path=Path(candidate)/'logs/training.csv'
    if not path.is_file():return None
    values=[]
    with path.open(encoding='utf-8',newline='') as stream:
        for row in csv.DictReader(stream):
            try:values.append(int(row['memory_rss_bytes']))
            except (KeyError,ValueError,TypeError):pass
    return max(values) if values else None


def build_command(case,python,baseline_root,source,data,candidate,prefetch=0,validation_mode='full',max_memory_gb=4):
    code=baseline_root if case.implementation=='baseline' else ROOT
    shard={100:25,1000:250,10000:1000}[case.positions]
    argv=[str(python),'-m','shardtrain.run','--positions',str(case.positions),'--shard-size',str(shard),
          '--epochs','3','--batch-size',str(case.batch_size),'--learning-rate','.001','--source',str(source),
          '--data-directory',str(data),'--checkpoint-directory',str(candidate),'--max-memory-gb',format(max_memory_gb,'g'),'--max-disk-gb','1',
          '--max-download-mb','16','--seed','2026','--checkpoint-every','32','--replay-size','256']
    if case.implementation!='baseline':
        argv += ['--device',case.device,'--cpu-threads',str(case.threads),'--precision',case.precision,
                 '--prefetch',str(prefetch),'--cache-mode',case.cache_mode,'--optimizer','default',
                 '--gradient-accumulation','1','--transfer-mode','copy','--validation-mode',validation_mode]
        if case.profile:argv.append('--profile')
    env=os.environ.copy();env['PYTHONPATH']=str(code);env['PYTHONDONTWRITEBYTECODE']='1'
    # The launcher changes no global environment or device settings.
    return argv,code,env


def terminate_tree(process):
    if process.poll() is not None:return
    if os.name=='nt':subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,timeout=10)
    else:process.kill()
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)


def run_case(case,*,python,baseline_python,baseline_root,source,run_root,report_root,timeout,prefetch,validation_mode='full',max_memory_gb=4):
    global _ACTIVE_PROCESS
    directory=run_root/case.key
    if directory.exists():raise FileExistsError('Fresh benchmark directory already exists: '+str(directory))
    directory.mkdir(parents=True);data=directory/'data';candidate=directory/'candidate'
    argv,cwd,env=build_command(case,baseline_python if case.implementation=='baseline' else python,baseline_root,source,data,candidate,
        prefetch,validation_mode=validation_mode,max_memory_gb=max_memory_gb)
    command={'argv':argv,'cwd':str(cwd),'environment':{'PYTHONPATH':env['PYTHONPATH'],'PYTHONDONTWRITEBYTECODE':'1'},
             'case':asdict(case),'validation_mode':'full' if case.implementation=='baseline' else validation_mode,
             'max_memory_gb':max_memory_gb,'cold_scope':'fresh data ledger, spool, and binary cache; OS file cache remains unmanaged'}
    atomic_json(directory/'command.json',command)
    stdout_path=directory/'training.stdout.log';sample_path=directory/'external-resources.jsonl'
    started=time.perf_counter();timed_out=False
    with stdout_path.open('w',encoding='utf-8') as stdout,sample_path.open('w',encoding='utf-8') as external:
        process=subprocess.Popen(argv,cwd=cwd,env=env,stdout=stdout,stderr=subprocess.STDOUT,text=True)
        _ACTIVE_PROCESS=process
        monitor=ProcessTreeSampler(process.pid)
        while True:
            elapsed=time.perf_counter()-started
            observed=monitor.sample();external.write(json.dumps(observed,separators=(',',':'))+'\n')
            if process.poll() is not None:break
            if elapsed>=timeout:
                timed_out=True;terminate_tree(process);break
            time.sleep(min(.2,max(.01,timeout-elapsed)))
        returncode=process.wait();wall=time.perf_counter()-started
        _ACTIVE_PROCESS=None
        tree_resources=monitor.snapshot()
        atomic_json(directory/'external-resource-summary.json',tree_resources)
    final=read_json(candidate/'training-report.json',{})
    stage=read_json(candidate/'profile/profile.json',{})
    resources=read_json(candidate/'resources/resource-summary.json',final.get('resources') or {})
    internal_peaks=resources.get('peaks',{});internal_means=resources.get('means',{})
    use_internal=case.implementation=='arc' and internal_peaks.get('rss_bytes',0)>0
    log_rss=training_log_rss_peak(candidate)
    selected_rss=internal_peaks['rss_bytes'] if use_internal else tree_resources['rss_peak_bytes'] or log_rss
    selected_cpu=internal_means.get('cpu_process_percent') if use_internal else tree_resources['cpu_process_percent_mean']
    selected_io={key:internal_peaks.get(key) if use_internal else tree_resources.get(key) for key in ProcessTreeSampler.IO_KEYS}
    execution=final.get('execution') or {};device=final.get('device') or {}
    profile_enabled=bool(stage.get('enabled'))
    stages=stage.get('stages',{}) if profile_enabled else {}
    def stage_seconds(names):
        if not profile_enabled:return None
        return sum(stages.get(name,{}).get('seconds',0.) for name in names) if any(name in stages for name in names) else None
    validation=final.get('validation') or {};test=final.get('test') or {}
    positions=final.get('positions',0);ok=returncode==0 and positions==case.positions
    row={**asdict(case),'key':case.key,'status':'timeout' if timed_out else 'ok' if ok else 'error','returncode':returncode,
         'validation_mode':command['validation_mode'],'max_memory_gb':max_memory_gb,
         'wall_seconds':wall,'positions_per_second':positions/wall if ok else None,'processed_positions':positions,
         'pure_training_seconds':execution.get('pure_training_seconds'),'training_samples':execution.get('training_samples'),
         'training_samples_per_second':execution.get('training_samples_per_second'),'transfer_seconds':execution.get('host_to_device_seconds'),
         'data_processing_seconds':stage_seconds(['source_read','zstd_decompress','source_spool_write','source_spool_read','json_parse',
             'fen_validation','pv_validation','label_conversion','dedup_sampling','shard_serialize','shard_deserialize',
             'binary_cache_prepare','binary_encode','data_loading','data_preparation','source_prefetch_wait']),
         'validation_seconds':stage_seconds(['validation']),'checkpoint_seconds':stage_seconds(['checkpoint_save']),
         'recovery_seconds':stage_seconds(['checkpoint_restore','checkpoint_recovery']),
         'profile_enabled':profile_enabled,'stage_timing_status':'measured exclusive spans; concurrent thread sums may exceed wall time' if profile_enabled else 'NA: profile disabled; no stage split inferred',
         'rss_peak_bytes':selected_rss,'cpu_process_percent_mean':selected_cpu,
         **selected_io,'external_resource_samples':tree_resources['samples'],'monitor_errors':tree_resources['errors'],
         'external_process_tree':tree_resources,'training_log_rss_peak_bytes':log_rss,
         'resource_measurement_source':'arc_internal_since_sampler_start' if use_internal else 'external_process_tree_lifetime_last_known',
         'invalid_parent_launcher_sample':False,
         'gpu_peak_allocated_bytes':(final.get('gpu_memory') or {}).get('peak_allocated_bytes'),
         'gpu_peak_reserved_bytes':(final.get('gpu_memory') or {}).get('peak_reserved_bytes'),
         'gpu_utilization_mean':resources.get('means',{}).get('gpu_busiest_engine_percent'),
         'gpu_utilization_available':resources.get('gpu_utilization_available',False),
         'gpu_utilization_unavailable_reason':resources.get('gpu_unavailable_reason','not an attributed XPU run'),
         'policy_validation_loss':validation.get('policy_loss'),'wdl_validation_loss':validation.get('wdl_loss'),
         'policy_test_loss':test.get('policy_loss'),'wdl_test_loss':test.get('wdl_loss'),
         'holdout_digest':final.get('holdout_digest'),'global_step':final.get('global_step'),
         'resolved_device':device.get('device',case.device),'effective_precision':execution.get('effective_precision',case.precision),
         'precision_fallbacks':execution.get('precision_fallbacks'),'directory':str(directory),'stdout':str(stdout_path),
         'command':str(directory/'command.json'),'source_cursor_line':(final.get('source_cursor') or {}).get('line'),
         'quality_stats':final.get('quality'),'cold_scope':command['cold_scope']}
    atomic_json(directory/'measurement.json',row)
    return row


CSV_FIELDS=['key','name','repeat','implementation','positions','threads','device','precision','batch_size','cache_mode','profile',
    'validation_mode','max_memory_gb',
    'status','returncode','wall_seconds','positions_per_second','pure_training_seconds','training_samples','training_samples_per_second',
    'data_processing_seconds','transfer_seconds','validation_seconds','checkpoint_seconds','recovery_seconds','rss_peak_bytes',
    'gpu_peak_allocated_bytes','gpu_peak_reserved_bytes','gpu_utilization_mean','gpu_utilization_available','cpu_process_percent_mean',
    'io_read_bytes','io_write_bytes','resource_measurement_source','invalid_parent_launcher_sample',
    'policy_validation_loss','wdl_validation_loss','policy_test_loss','wdl_test_loss','global_step',
    'holdout_digest','source_cursor_line','stage_timing_status','directory','stdout']


def summarize(rows):
    groups=defaultdict(list)
    for row in rows:
        if row.get('status')=='ok':groups[row['name']].append(row)
    medians={}
    metrics=[k for k in CSV_FIELDS if k.endswith(('_seconds','_loss','_bytes')) or k in ['positions_per_second','training_samples_per_second','gpu_utilization_mean','cpu_process_percent_mean']]
    baselines={r['positions']:r['wall_seconds'] for r in rows if r.get('status')=='ok' and r['implementation']=='baseline'}
    for name,parts in groups.items():
        item={k:parts[0][k] for k in ['positions','implementation','device','precision','threads','batch_size','cache_mode','profile']}
        item.update(validation_mode=parts[0].get('validation_mode','full'),max_memory_gb=parts[0].get('max_memory_gb',4))
        item.update(repetitions=len(parts),requested_repetitions=max(r['repeat'] for r in parts))
        for metric in metrics:
            values=[r[metric] for r in parts if isinstance(r.get(metric),(int,float)) and math.isfinite(r[metric])]
            item[metric+'_median']=statistics.median(values) if values else None
        baseline=baselines.get(item['positions']);item['speedup_vs_reproduced_baseline']=baseline/item['wall_seconds_median'] if baseline else None
        item['all_holdout_digests_agree']=len({r.get('holdout_digest') for r in parts})==1
        item['all_global_steps_agree']=len({r.get('global_step') for r in parts})==1
        medians[name]=item
    return medians


def write_progress(report_root,rows,plan,metadata):
    atomic_json(report_root/'pipeline-progress.json',{'metadata':metadata,'completed_cases':len(rows),'planned_cases':len(plan),'results':rows,'summary':summarize(rows)})
    # Regenerate complete JSONL/CSV instead of appending a partial trailing row.
    temp=report_root/'pipeline-runs.jsonl.tmp'
    temp.write_text(''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n' for row in rows),encoding='utf-8');os.replace(temp,report_root/'pipeline-runs.jsonl')
    with (report_root/'pipeline-runs.csv.tmp').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=CSV_FIELDS,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    os.replace(report_root/'pipeline-runs.csv.tmp',report_root/'pipeline-runs.csv')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=['small','main','profile','all'],default='all');p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--deadline-seconds',type=float,default=900);p.add_argument('--case-timeout-seconds',type=float,default=240)
    p.add_argument('--python',type=Path,default=Path(r'C:\Users\fly_t\chromatic-xpu\Scripts\python.exe'))
    p.add_argument('--baseline-python',type=Path);p.add_argument('--baseline-root',type=Path,default=ROOT.parent/'chromatic-shard-trainer')
    p.add_argument('--source',type=Path);p.add_argument('--run-group',default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    p.add_argument('--validation-mode',choices=['full','selected'],default='full')
    p.add_argument('--max-memory-gb',type=float,default=4)
    p.add_argument('--prefetch',type=int,default=0);p.add_argument('--case-filter',action='append',help='Run only named cases; repeat this option for multiple names')
    p.add_argument('--plan-only',action='store_true');args=p.parse_args(argv)
    if not 1<=args.deadline_seconds<=1800 or not 1<=args.case_timeout_seconds<=300:raise ValueError('Invalid bounded benchmark duration')
    if not 0<=args.prefetch<=16:raise ValueError('Prefetch must be between 0 and 16')
    if not math.isfinite(args.max_memory_gb) or args.max_memory_gb<.5:raise ValueError('MaxMemoryGB must be finite and at least 0.5')
    if not args.run_group or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.run_group):raise ValueError('Run group must be a simple alphanumeric identifier')
    baseline_root=args.baseline_root.resolve();source=(args.source or baseline_root/'fixtures/official-evaluations.jsonl.zst').resolve()
    python=args.python.resolve();baseline_python=(args.baseline_python or python).resolve()
    for path in [python,baseline_python,baseline_root/'shardtrain/run.py',source,Path(str(source)+'.source.json')]:
        if not path.is_file():raise FileNotFoundError(path)
    provenance=read_json(Path(str(source)+'.source.json'))
    if provenance.get('license')!='CC0-1.0' or provenance.get('local_sha256')!=sha(source):raise ValueError('Local official source integrity/provenance mismatch')
    plan=make_cases(args.suite,args.repeats)
    if args.case_filter:
        chosen=set(args.case_filter);available={case.name for case in plan}
        if not chosen<=available:raise ValueError('Case filters not in this suite: '+str(chosen-available))
        plan=[case for case in plan if case.name in chosen]
    report_root=ROOT/'reports/benchmarks'/('pipeline-'+args.run_group)
    run_root=ROOT/'runs'/('benchmark-'+args.run_group)
    metadata={'format':'chromatic-arc-pipeline-benchmark-v1','suite':args.suite,'requested_repeats':args.repeats,
        'created_utc':datetime.now(timezone.utc).isoformat(),'python':str(python),'baseline_python':str(baseline_python),
        'baseline_root':str(baseline_root),'source':str(source),'source_sha256':sha(source),
        'champion_sha256':sha(ROOT/'champion/model.pt'),'baseline_champion_sha256':sha(baseline_root/'champion/model.pt'),
        'settings':{'epochs':3,'seed':2026,'replay_size':256,'checkpoint_every':32,'max_memory_gb':args.max_memory_gb,
                    'validation_mode':args.validation_mode,'baseline_validation_mode':'full','max_disk_gb':1,'max_download_mb':16,'prefetch':args.prefetch},
        'deadline_seconds':args.deadline_seconds,'case_timeout_seconds':args.case_timeout_seconds,
        'sequential_only':True,'downloads_started':False,'cold_scope':'Fresh per-run ledger and encoded cache. OS file cache is not flushed or controlled.',
        'historical_baseline_seconds':{'100':15.4,'1000':24.8,'10000':132.6},'plan':[asdict(case) for case in plan]}
    if args.plan_only:print(json.dumps(metadata,ensure_ascii=False,indent=2));return 0
    if report_root.exists() or run_root.exists():raise FileExistsError('Run group already exists; choose a new --run-group')
    report_root.mkdir(parents=True);run_root.mkdir(parents=True)
    atomic_json(report_root/'plan.json',metadata);results=[];started=time.perf_counter();reason='completed'
    try:
        for index,case in enumerate(plan):
            remaining=args.deadline_seconds-(time.perf_counter()-started)
            if remaining<1:reason='total_deadline';break
            print(json.dumps({'event':'case_started','index':index+1,'of':len(plan),'case':case.key,'remaining_seconds':remaining}),flush=True)
            row=run_case(case,python=python,baseline_python=baseline_python,baseline_root=baseline_root,source=source,
                run_root=run_root,report_root=report_root,timeout=min(args.case_timeout_seconds,remaining),prefetch=args.prefetch,
                validation_mode=args.validation_mode,max_memory_gb=args.max_memory_gb)
            results.append(row);write_progress(report_root,results,plan,metadata)
            print(json.dumps({'event':'case_completed','case':case.key,'status':row['status'],'wall_seconds':row['wall_seconds'],
                'policy_validation_loss':row['policy_validation_loss'],'wdl_validation_loss':row['wdl_validation_loss']}),flush=True)
    except KeyboardInterrupt:reason='interrupted'
    except Exception as exc:
        reason='driver_error';metadata['driver_error']=f'{type(exc).__name__}: {exc}'
    finally:
        if _ACTIVE_PROCESS is not None:terminate_tree(_ACTIVE_PROCESS)
        final={'metadata':metadata,'status':reason,'elapsed_seconds':time.perf_counter()-started,'finished_cases':len(results),
            'planned_cases':len(plan),'all_requested_completed':len(results)==len(plan),'results':results,'summary':summarize(results)}
        write_progress(report_root,results,plan,metadata);atomic_json(report_root/'pipeline-summary.json',final)
    print(json.dumps({'summary':str(report_root/'pipeline-summary.json'),'status':reason,'finished':len(results),'planned':len(plan)}))
    return 130 if reason=='interrupted' else 1 if reason=='driver_error' else 0

if __name__=='__main__':raise SystemExit(main())
