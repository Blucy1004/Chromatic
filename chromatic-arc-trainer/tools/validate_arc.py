"""Bounded numerical, native-export and real-process recovery validation.

Only the separate ARC output directory is writable. Existing engines, models
and the previous experiment corpus are read-only inputs. GPU runs are explicit.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while data:=stream.read(1024*1024):h.update(data)
    return h.hexdigest()


def atomic_report(path,data):
    from shardtrain.storage import atomic_json
    atomic_json(path,data)


def read_records(path,maximum=32):
    rows=[]
    with Path(path).open(encoding='utf-8') as source:
        for raw in source:
            if raw.strip():rows.append(json.loads(raw))
            if len(rows)>=maximum:break
    if not rows:raise ValueError('An existing fixed normalized JSONL set is required')
    return rows


def tensor_tree_cpu(value):
    import torch
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {key:tensor_tree_cpu(item) for key,item in value.items()}
    if isinstance(value,list):return [tensor_tree_cpu(item) for item in value]
    if isinstance(value,tuple):return tuple(tensor_tree_cpu(item) for item in value)
    return copy.deepcopy(value)


def assert_cpu_tensors(value,path='state'):
    import torch
    if isinstance(value,torch.Tensor):
        if value.device.type!='cpu':raise AssertionError(path+' was not saved as a CPU tensor')
    elif isinstance(value,dict):
        for key,item in value.items():assert_cpu_tensors(item,path+'.'+str(key))
    elif isinstance(value,(tuple,list)):
        for index,item in enumerate(value):assert_cpu_tensors(item,path+f'[{index}]')


def tree_compare(expected,actual,*,atol=0.,rtol=0.,path='state'):
    """Compare training semantics; callers deliberately exclude wall time/paths."""
    import numpy as np
    import torch
    if isinstance(expected,torch.Tensor):
        if not isinstance(actual,torch.Tensor) or expected.shape!=actual.shape or expected.dtype!=actual.dtype:
            raise AssertionError(path+' tensor schema differs')
        expected=expected.cpu();actual=actual.cpu()
        equal=torch.equal(expected,actual) if not (atol or rtol) or not expected.is_floating_point() else torch.allclose(expected,actual,atol=atol,rtol=rtol)
        if not equal:
            error=float((expected.to(torch.float64)-actual.to(torch.float64)).abs().max()) if expected.numel() else 0.
            raise AssertionError(path+f' tensor differs (maximum absolute error {error})')
    elif isinstance(expected,np.ndarray):
        if not isinstance(actual,np.ndarray) or expected.dtype!=actual.dtype or expected.shape!=actual.shape or not np.array_equal(expected,actual):
            raise AssertionError(path+' array differs')
    elif isinstance(expected,dict):
        if not isinstance(actual,dict) or expected.keys()!=actual.keys():raise AssertionError(path+' dictionary schema differs')
        for key in expected:tree_compare(expected[key],actual[key],atol=atol,rtol=rtol,path=path+'.'+str(key))
    elif isinstance(expected,(list,tuple)):
        if type(expected)!=type(actual) or len(expected)!=len(actual):raise AssertionError(path+' sequence differs')
        for index,(left,right) in enumerate(zip(expected,actual)):tree_compare(left,right,atol=atol,rtol=rtol,path=path+f'[{index}]')
    elif isinstance(expected,float) and (atol or rtol):
        if not math.isclose(expected,actual,abs_tol=atol,rel_tol=rtol):raise AssertionError(path+' differs')
    elif expected!=actual:raise AssertionError(path+' differs')


def _sync(device):
    import torch
    if device=='xpu':torch.xpu.synchronize()


def _loss_outputs(model,tensors,precision='fp32',device='cpu'):
    import torch
    x,p,v,mask,pw,vw=tensors
    context=torch.autocast('xpu',dtype=torch.bfloat16) if precision=='bf16' else nullcontext()
    with context:policy,wdl=model(x)
    # Compute probabilities and the two weighted cross-entropies in FP32.
    policy=policy.float();wdl=wdl.float()
    pl=-(p*torch.log_softmax(policy.masked_fill(~mask,-1e9),-1)).sum(-1)
    vl=-(v*torch.log_softmax(wdl,-1)).sum(-1)
    return (pl*pw+vl*vw).mean(),policy,torch.softmax(wdl,-1)


def _train_one(state,records,device,precision='fp32',restore_optimizer=False):
    import torch
    from shardtrain.model_adapter import PolicyWDL,ModelConfig,batch
    from shardtrain.devices import make_optimizer,clip_gradients
    model=PolicyWDL(ModelConfig(**state['config']));model.load_state_dict(state['model']);model.to(device)
    optimizer=make_optimizer(model,lr=.001,weight_decay=1e-4,mode='default')
    if restore_optimizer and state.get('optimizer'):
        optimizer.load_state_dict(state['optimizer'])
        for values in optimizer.state.values():
            for key,value in values.items():
                if isinstance(value,torch.Tensor) and key!='step':values[key]=value.to(device)
    tensors=tuple(t.to(device) for t in batch(records));optimizer.zero_grad(set_to_none=True)
    _sync(device);start=time.perf_counter()
    loss,policy,wdl=_loss_outputs(model,tensors,precision,device)
    loss.backward()
    gradients={name:p.grad.detach().cpu().clone() for name,p in model.named_parameters() if p.grad is not None}
    raw_norm64=math.sqrt(sum(float(g.double().square().sum()) for g in gradients.values()))
    # Arc 140T / PyTorch 2.14.1+xpu foreach norm has a measured reduction
    # error. The non-foreach reduction agrees with copied FP64 gradients.
    grad_norm=clip_gradients(model.parameters(),1.)
    clipped_norm64=math.sqrt(sum(float(p.grad.detach().cpu().double().square().sum()) for p in model.parameters() if p.grad is not None))
    optimizer.step();_sync(device)
    elapsed=time.perf_counter()-start
    if not math.isfinite(float(loss.detach().cpu())):raise AssertionError('nonfinite loss')
    for name,p in model.named_parameters():
        if not bool(torch.isfinite(p).all().cpu()):raise AssertionError('nonfinite updated weight '+name)
    result={'loss':float(loss.detach().cpu()),'policy':policy.detach().cpu(),'wdl':wdl.detach().cpu(),
            'gradients':gradients,'updated':tensor_tree_cpu(model.state_dict()),'optimizer':tensor_tree_cpu(optimizer.state_dict()),
            'gradient_norm':float(grad_norm.detach().cpu()),'gradient_norm_reference_float64':raw_norm64,
            'clipped_gradient_norm_float64':clipped_norm64,'clip_foreach':False if device=='xpu' else None,
            'optimizer_mode':'default','seconds':elapsed}
    result['head_raw_gradient_norm_float64']={head:math.sqrt(sum(float(g.double().square().sum()) for name,g in gradients.items()
        if name.startswith(head+'.'))) for head in ['policy','wdl']}
    result['head_update_norm_float64']={head:math.sqrt(sum(float((result['updated'][name].double()-state['model'][name].double()).square().sum())
        for name in result['updated'] if name.startswith(head+'.'))) for head in ['policy','wdl']}
    return result


def _max_error(left,right):
    import torch
    if isinstance(left,dict):return max((_max_error(left[key],right[key]) for key in left),default=0.)
    return float((left.to(torch.float64)-right.to(torch.float64)).abs().max()) if left.numel() else 0.


def numeric_validation(records,device='xpu',precision='fp32',initial_checkpoint=None,output=None):
    import torch
    from shardtrain.model_adapter import checkpoint
    from shardtrain.devices import cpu_state,capture_xpu_rng,restore_xpu_rng
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(2026)
    if device=='xpu' and not torch.xpu.is_available():raise RuntimeError('XPU validation explicitly requested but torch.xpu is unavailable')
    if precision=='bf16' and device!='xpu':raise ValueError('This validation treats BF16 as an XPU experiment')
    initial_checkpoint=Path(initial_checkpoint or ROOT/'champion/model.pt')
    before=digest(initial_checkpoint);state,actual=checkpoint.load(initial_checkpoint)
    cpu=_train_one(state,records,'cpu','fp32')
    accelerated=_train_one(state,records,device,precision)
    errors={'policy_logits':_max_error(cpu['policy'],accelerated['policy']),
            'wdl_probability':_max_error(cpu['wdl'],accelerated['wdl']),
            'gradient':_max_error(cpu['gradients'],accelerated['gradients']),
            'updated_weights':_max_error(cpu['updated'],accelerated['updated']),
            'relative_loss':abs(cpu['loss']-accelerated['loss'])/max(abs(cpu['loss']),1e-8)}
    limits={'policy_logits':1e-4,'wdl_probability':1e-5,'gradient':1e-4,'updated_weights':1e-4,'relative_loss':1e-5} if precision=='fp32' else {
        'policy_logits':.25,'wdl_probability':.02,'gradient':.02,'updated_weights':.0021,'relative_loss':.05}
    passed=all(math.isfinite(value) and value<=limits[key] for key,value in errors.items())
    norm_checks={}
    for label,result in [('cpu_fp32',cpu),('tested',accelerated)]:
        reference=result['gradient_norm_reference_float64']
        returned=result['gradient_norm'];after=result['clipped_gradient_norm_float64']
        expected_after=min(reference,reference/(reference+1e-6))
        allowed=max(1e-5,reference*1e-5)
        norm_checks[label]={'reference_cpu_float64':reference,'returned_norm':returned,'absolute_error':abs(returned-reference),
            'allowed_absolute_error':allowed,'clipped_cpu_float64_norm':after,'expected_clipped_norm':expected_after,
            'passed':abs(returned-reference)<=allowed and abs(after-expected_after)<=1e-5,'foreach':result['clip_foreach']}
        passed=passed and norm_checks[label]['passed']
    head_coverage={label:result['head_raw_gradient_norm_float64'] for label,result in [('cpu_fp32',cpu),('tested',accelerated)]}
    both_heads=all(math.isfinite(norm) and norm>0 for values in head_coverage.values() for norm in values.values())
    passed=passed and both_heads
    old_optimizer=_train_one(state,records,device,precision,restore_optimizer=True)
    # GPU-created tensors are recursively detached/copied; the file is consumed
    # by torch.load(map_location=cpu) and a CPU Optimizer performs another step.
    portable=dict(state,model=cpu_state(accelerated['updated']),optimizer=cpu_state(accelerated['optimizer']))
    assert_cpu_tensors(portable)
    scratch=ROOT/'reports/.numeric-work';scratch.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='portable-',dir=scratch) as temporary:
        saved_path=Path(temporary)/'portable.pt';torch.save(portable,saved_path)
        portable_bytes=saved_path.stat().st_size
        reloaded=torch.load(saved_path,map_location='cpu',weights_only=False);assert_cpu_tensors(reloaded)
        resumed_cpu=_train_one(reloaded,records,'cpu','fp32',restore_optimizer=True)
    rng=None
    if device=='xpu':
        saved=capture_xpu_rng();first=torch.rand(16,device='xpu').cpu();restore_xpu_rng(saved,strict=True)
        second=torch.rand(16,device='xpu').cpu();_sync(device)
        rng={'saved_cpu_uint8':all(t.device.type=='cpu' and t.dtype==torch.uint8 for t in saved),'exact_restore':bool(torch.equal(first,second))}
        if not all(rng.values()):passed=False
    report={'format':'chromatic-arc-numeric-validation-v1','passed':passed,'device':device,'precision':precision,
        'records':len(records),'optimizer_mode':'default','initial_checkpoint':str(actual),'initial_sha256':before,'initial_unchanged':digest(initial_checkpoint)==before,
        'errors':errors,'predeclared_absolute_limits':limits,'loss':{'cpu_fp32':cpu['loss'],'tested':accelerated['loss']},
        'gradient_norm':{'cpu_fp32':cpu['gradient_norm'],'tested':accelerated['gradient_norm']},'gradient_norm_checks':norm_checks,
        'head_raw_gradient_norm_float64':head_coverage,
        'head_update_norm_float64':{label:result['head_update_norm_float64'] for label,result in [('cpu_fp32',cpu),('tested',accelerated)]},
        'both_heads_gradients_exercised':both_heads,'validation_coverage_weak':not both_heads,
        'input_coverage':{'positive_policy_weight_rows':sum(float(row['policy_weight'])>0 for row in records),
            'positive_value_weight_rows':sum(float(row['value_weight'])>0 for row in records),
            'policy_weight_sum':sum(float(row['policy_weight']) for row in records),'value_weight_sum':sum(float(row['value_weight']) for row in records),
            'identities':[row.get('identity') for row in records]},
        'old_optimizer_compatible':math.isfinite(old_optimizer['loss']),'cpu_portable_optimizer_step_finite':math.isfinite(resumed_cpu['loss']),
        'portable_tensors_all_cpu':True,'portable_serialized_bytes':portable_bytes,'portable_file_loaded_on_cpu':True,'xpu_rng':rng,
        'scope':'one identical real-data batch; numerical fidelity check, not convergence or playing-strength evidence',
        'seconds':{'cpu':cpu['seconds'],'tested':accelerated['seconds']}}
    if output:atomic_report(output,report)
    if not both_heads:raise AssertionError('Validation coverage is insufficient: Policy and WDL raw gradients must both be nonzero; include positive-value-weight records')
    if not passed:raise AssertionError('Numerical limits exceeded: '+json.dumps(errors))
    return report


def native_validation(records,checkpoint_path,engine,output):
    import numpy as np
    import torch
    from shardtrain.model_adapter import checkpoint,PolicyWDL,ModelConfig,features,restore_board,action
    from zero.native import Native
    before={str(p):digest(p) for p in [Path(checkpoint_path),Path(engine)]}
    state,actual=checkpoint.load(checkpoint_path);assert_cpu_tensors(state)
    model=PolicyWDL(ModelConfig(**state['config']));model.load_state_dict(state['model']);model.eval();torch.set_num_threads(1)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    exported=output.with_suffix('.zero');model.export(exported)
    maximum={'features':0.,'legal_policy_logits':0.,'wdl_probability':0.};cases=[]
    with Native(Path(engine).resolve(),exported.resolve(),timeout=30) as native:
        native.configure({'InferenceBackend':'SIMD','SparsePolicyHead':True,'SearchTelemetry':False})
        for record in records:
            board=restore_board(record);native.position(board);result=native.command('zero evallegal')
            legal={move.uci():action(move) for move in board.legal_moves}
            if {r['uci']:r['action'] for r in result['legal']}!=legal:raise AssertionError('native legal move/action mismatch')
            x=features(board)
            with torch.no_grad():p,v=model(torch.from_numpy(x[None,:]));p=p[0].numpy();v=torch.softmax(v,-1)[0].numpy()
            ids=list(legal.values())
            errors={'features':float(np.max(np.abs(np.asarray(result['features'])-x))),
                    'legal_policy_logits':float(np.max(np.abs(np.asarray(result['policy'])[ids]-p[ids]))),
                    'wdl_probability':float(np.max(np.abs(np.asarray(result['wdl'])-v)))}
            if errors['features']>1e-7 or max(errors['legal_policy_logits'],errors['wdl_probability'])>2e-5:raise AssertionError('native inference tolerance exceeded')
            for key,value in errors.items():maximum[key]=max(maximum[key],value)
            cases.append({'identity':record.get('identity'),'fen':board.fen(),'max_abs_error':errors})
    after={p:digest(p) for p in before}
    if before!=after:raise AssertionError('read-only checkpoint or engine changed')
    report={'format':'chromatic-arc-native-validation-v1','passed':True,'cases':cases,'maximum_absolute_error':maximum,
            'tolerance':{'features':1e-7,'policy_logits':2e-5,'wdl_probability':2e-5},'checkpoint':str(actual),
            'exported_model':str(exported),'export_sha256':digest(exported),'preservation':{'before':before,'after':after,'unchanged':True}}
    atomic_report(output,report);return report


def _snapshot(case):
    from shardtrain.durable import verify_bundle,read_manifest
    case=Path(case)
    with sqlite3.connect((case/'data/ledger.sqlite3').resolve().as_uri()+'?mode=ro',uri=True) as db:
        def get(key):return json.loads(db.execute('SELECT v FROM meta WHERE k=?',(key,)).fetchone()[0])
        head=get('current_manifest');cursor=get('source_cursor');fixed=get('fixed_sets')
        events=[json.loads(row[0]) for row in db.execute('SELECT payload FROM events ORDER BY id')]
    state,mirror=verify_bundle(head);assert_cpu_tensors(state);manifest=read_manifest(head)
    return {'state':state,'manifest':head,'metadata':manifest,'mirror':mirror,'fixed':fixed,'source_cursor':cursor,
            'events':events,'export_sha256':digest(manifest['files']['model']['path']),
            'status':json.loads((case/'candidate/status.json').read_text(encoding='utf-8'))}


def compare_snapshots(expected,actual):
    keys=['config','model','optimizer','scheduler','generator','rng','progress','replay','holdout_digest','step','source_cursor']
    if 'rng_xpu' in expected['state']:keys.append('rng_xpu')
    for key in keys:tree_compare(expected['state'][key],actual['state'][key],path=key)
    if expected['export_sha256']!=actual['export_sha256']:raise AssertionError('native export differs after resume')
    for split in ['validation','test']:
        for field in ['ids','sha256']:
            if expected['fixed'][split][field]!=actual['fixed'][split][field]:raise AssertionError('fixed holdout changed')
    if expected['source_cursor']!=actual['source_cursor']:raise AssertionError('source cursor changed')
    if actual['status']['status']!='completed':raise AssertionError('resumed run incomplete')
    completed=[e for e in actual['events'] if e.get('kind')=='shard_completed']
    if len(completed)!=4 or len({e['commit_seq'] for e in completed})!=4:raise AssertionError('duplicate/missing completed shard commit')
    return {'exact_model_optimizer_scheduler_rng_progress_replay':True,'native_export_equal':True,'fixed_holdout_equal':True,
            'source_cursor_equal':True,'checkpoint_tensors_cpu':True,'global_step':actual['state']['step']}


def recovery_validation(directory,device='cpu',precision='fp32',cpu_threads=1,cache_mode='auto',prefetch=2,output=None,timeout=240,max_memory_gb=4,validation_mode='full',profile_resume=False):
    """Run bounded actual process termination. Never uses external network."""
    if validation_mode not in ['full','selected']:raise ValueError('Validation mode must be full or selected')
    directory=Path(directory).resolve()
    if directory.exists() and any(directory.iterdir()):raise ValueError('Recovery output must be a new empty directory')
    directory.mkdir(parents=True,exist_ok=True)
    output=Path(output or directory/'recovery-report.json');source=ROOT/'fixtures/official-evaluations.jsonl.zst'
    report={'format':'chromatic-arc-recovery-validation-v1','device':device,'precision':precision,'cpu_threads':cpu_threads,
        'positions_per_case':100,'epochs':2,'cache_mode':cache_mode,'prefetch':prefetch,'max_memory_gb':max_memory_gb,
        'validation_mode':validation_mode,'profile_resume':bool(profile_resume),'status':'running','tests':[]}
    atomic_report(output,report)
    def command(case,resume=False):
        args=[sys.executable,'-m','shardtrain.run','--data-directory',str(case/'data')]
        if resume:return args+['--resume']+(['--profile'] if profile_resume else [])
        return args+['--positions','100','--shard-size','25','--epochs','2','--batch-size','32',
            '--source',str(source),'--checkpoint-directory',str(case/'candidate'),'--max-download-mb','16',
            '--max-disk-gb','1','--max-memory-gb',str(max_memory_gb),'--checkpoint-every','2','--replay-size','256',
            '--device',device,'--cpu-threads',str(cpu_threads),'--precision',precision,'--cache-mode',cache_mode,'--prefetch',str(prefetch),
            '--validation-mode',validation_mode]
    def env(hook=None):
        values=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
        for name in ['CHROMATIC_TEST_PAUSE_AT','CHROMATIC_TEST_PAUSE_STEP','CHROMATIC_TEST_PAUSE_SHARD']:values.pop(name,None)
        if hook:values.update(CHROMATIC_TEST_PAUSE_AT=hook,CHROMATIC_TEST_PAUSE_STEP='2',CHROMATIC_TEST_PAUSE_SHARD='2' if hook=='before_delete' else '0')
        return values
    def run(case,resume=False):
        case.mkdir(parents=True,exist_ok=True);stamp=time.time_ns();log=case/f'process-{stamp}.log';start=time.perf_counter()
        with log.open('w',encoding='utf-8') as stream:
            result=subprocess.run(command(case,resume),cwd=ROOT,env=env(),stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
        if result.returncode:raise AssertionError(f'Child returned {result.returncode}: '+log.read_text(encoding='utf-8',errors='replace')[-6000:])
        entry={'seconds':time.perf_counter()-start,'returncode':result.returncode,'log':str(log)}
        if resume and profile_resume:
            profile_file=case/'candidate/profile/profile.json'
            profile=json.loads(profile_file.read_text(encoding='utf-8'))
            stage=profile.get('stages',{}).get('checkpoint_recovery',{})
            seconds=stage.get('seconds');calls=stage.get('calls')
            if not profile.get('enabled') or not isinstance(seconds,(float,int)) or not math.isfinite(seconds) or seconds<0 or not calls:
                raise AssertionError('Resume checkpoint_recovery exclusive span was not measured')
            snapshot=case/f'resume-profile-{stamp}.json';snapshot.write_bytes(profile_file.read_bytes())
            timing={'case':str(case),'exclusive_seconds':seconds,'calls':calls,'profile':str(snapshot),'profile_sha256':digest(snapshot)}
            report.setdefault('checkpoint_recovery_measurements',[]).append(timing)
            entry.update(checkpoint_recovery_seconds=seconds,checkpoint_recovery_calls=calls,checkpoint_recovery_profile=str(snapshot))
        return entry
    try:
        baseline_case=directory/'uninterrupted';baseline_run=run(baseline_case);baseline=_snapshot(baseline_case)
        resumed_run=run(baseline_case,True);after=_snapshot(baseline_case);detail=compare_snapshots(baseline,after)
        if baseline['manifest']!=after['manifest']:raise AssertionError('completed resume created a new model commit')
        report['baseline']={'step':baseline['state']['step'],'export_sha256':baseline['export_sha256'],'run':baseline_run}
        report['tests'].append({'name':'completed_resume','passed':True,'resume':resumed_run,'no_new_commit':True,**detail});atomic_report(output,report)
        for hook in ['checkpoint_written','ledger_committed','before_delete']:
            case=directory/('kill-'+hook);case.mkdir();marker=case/'candidate/test-marker.json';log=case/'killed-process.log';start=time.perf_counter()
            with log.open('w',encoding='utf-8') as stream:
                process=subprocess.Popen(command(case),cwd=ROOT,env=env(hook),stdout=stream,stderr=subprocess.STDOUT)
                try:
                    while not marker.is_file():
                        if process.poll() is not None:raise AssertionError('Child ended before kill marker: '+log.read_text(encoding='utf-8',errors='replace')[-5000:])
                        if time.perf_counter()-start>timeout:raise AssertionError('Timed out waiting for kill marker')
                        time.sleep(.05)
                    paused=json.loads(marker.read_text(encoding='utf-8'));os.kill(int(paused['pid']),signal.SIGTERM);process.wait(timeout=20)
                finally:
                    if process.poll() is None:process.kill();process.wait(timeout=20)
            resumed=run(case,True);actual=_snapshot(case);detail=compare_snapshots(baseline,actual)
            report['tests'].append({'name':'kill_'+hook,'passed':True,'marker':paused,'forced_exit':process.returncode,'resume':resumed,**detail});atomic_report(output,report)
        corrupt=directory/'corruption';run(corrupt);snapshot=_snapshot(corrupt);metadata=snapshot['metadata']
        Path(metadata['files']['checkpoint']['path']).write_bytes(b'PRIMARY CORRUPTION IN PRIVATE TEST RUN')
        mirror_run=run(corrupt,True);mirrored=_snapshot(corrupt);compare_snapshots(baseline,mirrored)
        if not mirrored['mirror']:raise AssertionError('mirror fallback was not used')
        Path(metadata['files']['mirror']['path']).write_bytes(b'MIRROR CORRUPTION IN PRIVATE TEST RUN')
        prior_run=run(corrupt,True);prior=_snapshot(corrupt);detail=compare_snapshots(baseline,prior)
        if prior['manifest']==snapshot['manifest']:raise AssertionError('previous checkpoint fallback was not used')
        report['tests'].append({'name':'mirror_and_previous_checkpoint','passed':True,'mirror_resume':mirror_run,'previous_resume':prior_run,**detail})
        report['status']='passed';report['passed']=True;atomic_report(output,report);return report
    except BaseException as exc:
        report.update(status='failed',passed=False,error=f'{type(exc).__name__}: {exc}');atomic_report(output,report);raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['numeric','native','recovery'],required=True)
    p.add_argument('--device',choices=['cpu','xpu'],default='cpu');p.add_argument('--precision',choices=['fp32','bf16'],default='fp32')
    p.add_argument('--records',type=Path);p.add_argument('--maximum-records',type=int,default=32)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'champion/model.pt');p.add_argument('--engine',type=Path)
    p.add_argument('--directory',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cpu-threads',type=int,default=1);p.add_argument('--prefetch',type=int,default=2)
    p.add_argument('--cache-mode',default='auto');p.add_argument('--timeout',type=int,default=240)
    p.add_argument('--max-memory-gb',type=float,default=4)
    p.add_argument('--validation-mode',choices=['full','selected'],default='full')
    p.add_argument('--profile-resume',action='store_true',help='Profile recovery spans in resumed children only; initial baseline stays unprofiled')
    a=p.parse_args(argv)
    if not 1<=a.maximum_records<=100:raise ValueError('Use 1..100 bounded validation records')
    if a.phase=='recovery':
        if not a.directory:p.error('--directory must be a new empty private test directory')
        report=recovery_validation(a.directory,a.device,a.precision,a.cpu_threads,a.cache_mode,a.prefetch,a.output,a.timeout,a.max_memory_gb,a.validation_mode,a.profile_resume)
    else:
        if not a.records:p.error('--records must name an existing fixed normalized JSONL file')
        records=read_records(a.records,a.maximum_records)
        if a.phase=='numeric':report=numeric_validation(records,a.device,a.precision,a.checkpoint,a.output)
        else:
            if not a.engine:p.error('--engine is required for native export validation')
            report=native_validation(records,a.checkpoint,a.engine,a.output)
    print(json.dumps({'phase':a.phase,'passed':report.get('passed',True),'output':str(a.output)},ensure_ascii=False));return 0


if __name__=='__main__':raise SystemExit(main())
