"""Bounded device/batch experiments; recommendations are tied to the model.

This is a microbenchmark of already encoded real records. It does not replace
the full streaming/shard/checkpoint benchmark or independent chess validation.
"""
from pathlib import Path
from dataclasses import asdict
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import time

from .devices import (DeviceRuntime,DeviceUnavailable,NumericalError,make_optimizer,cpu_state,
                      hardware_signature,available_xpu,assert_finite,BatchTransfer)
from .model_adapter import torch,PolicyWDL,ModelConfig,batch as encode_batch,checkpoint
from .storage import atomic_json,digest

BATCHES=(32,64,128,256,512)
THREADS=(1,2,4,8)

def model_implementation():
    root=Path(__file__).resolve().parents[1]
    return {name:digest(root/name) for name in ['vendor/zero/model.py','vendor/zero/encoding.py']}

def _bound_implementation(path,report):
    expected=report.get('model_implementation_sha256')
    if expected is None:
        binding=Path(path).parent/'model-implementation-binding.json'
        if not binding.is_file():return False
        document=json.loads(binding.read_text(encoding='utf-8'))
        if document.get('matrix_sha256')!=digest(path):return False
        expected=document.get('model_implementation_sha256')
    return expected==model_implementation()


def tensor_loss(model,tensors,runtime):
    x,p,v,mask,pw,vw=tensors
    with runtime.autocast():policy,wdl=model(x)
    # Reductions remain FP32 under BF16 autocast; no BF16 softmax/CE reduction.
    pl=-(p*torch.log_softmax(policy.float().masked_fill(~mask,-1e9),-1)).sum(-1)
    vl=-(v*torch.log_softmax(wdl.float(),-1)).sum(-1)
    return (pl*pw+vl*vw).mean()


def load_records(path,limit=512):
    rows=[]
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():rows.append(json.loads(line))
            if len(rows)>=limit:break
    if not rows:raise ValueError('Benchmark records are empty')
    return rows


def _repeat_batch(tensors,n):
    indices=torch.arange(n)%len(tensors[0]);return tuple(t[indices].contiguous() for t in tensors)


def _train_once(model,optimizer,tensors,runtime,accumulation=1,finite_mode='auto'):
    optimizer.zero_grad(set_to_none=True);size=len(tensors[0]);losses=[]
    chunk=max(1,math.ceil(size/accumulation))
    for start in range(0,size,chunk):
        part=tuple(t[start:start+chunk] for t in tensors)
        loss=tensor_loss(model,part,runtime);(loss*(len(part[0])/size)).backward();losses.append(loss.detach())
    # Gradient clipping checks the shared device reduction, with one host result.
    from .devices import clip_gradients
    norm=clip_gradients(model.parameters(),1.)
    optimizer.step()
    assert_finite(model,optimizer,mode=finite_mode)
    return torch.stack(losses).mean(),norm


def benchmark_case(initial,tensors,*,device='cpu',precision='fp32',cpu_threads=1,batch_size=32,
                   mode='copy',optimizer_mode='default',accumulation=1,finite_mode='auto',repeats=3,steps=15,warmup=2,seed=2026):
    """Each repeat restarts the same model and rows; synchronized phase timings."""
    if mode not in ['copy','preallocated','resident']:raise ValueError('Unknown transfer/cache mode')
    if not 1<=repeats<=5 or not 1<=steps<=100 or not 0<=warmup<=10:raise ValueError('Microbenchmark must remain bounded')
    runtime=DeviceRuntime(device,precision,cpu_threads)
    source=_repeat_batch(tensors,batch_size);config=ModelConfig(**initial['config'])
    trials=[];last_state=None
    for repeat in range(repeats):
        runtime.seed(seed);model=PolicyWDL(config).to(runtime.device);model.load_state_dict(initial['model']);model.train()
        optimizer=make_optimizer(model,mode=optimizer_mode)
        transfer=BatchTransfer(runtime,preallocate=mode=='preallocated')
        runtime.synchronize();setup_start=time.perf_counter()
        resident=runtime.transfer(source) if mode=='resident' else None
        runtime.synchronize();resident_setup=time.perf_counter()-setup_start
        def get_batch():return resident if resident is not None else transfer(source)
        warm_start=time.perf_counter()
        for _ in range(warmup):_train_once(model,optimizer,get_batch(),runtime,accumulation,finite_mode)
        runtime.synchronize();warm_seconds=time.perf_counter()-warm_start
        # Warm kernels, then restore exactly the initial weights and fresh state.
        model.load_state_dict(initial['model']);optimizer=make_optimizer(model,mode=optimizer_mode)
        runtime.reset_peak_memory();runtime.synchronize();start=time.perf_counter()
        transfer_seconds=0.;training_seconds=0.;last_loss=None
        for _ in range(steps):
            runtime.synchronize();stamp=time.perf_counter();current=get_batch();runtime.synchronize()
            transfer_seconds+=time.perf_counter()-stamp
            stamp=time.perf_counter();last_loss,_=_train_once(model,optimizer,current,runtime,accumulation,finite_mode)
            runtime.synchronize();training_seconds+=time.perf_counter()-stamp
        runtime.synchronize();total=time.perf_counter()-start
        assert_finite(model,optimizer,True)
        model.eval()
        with torch.no_grad():post_loss=float(tensor_loss(model,get_batch(),runtime))
        if not math.isfinite(post_loss):raise NumericalError('Nonfinite benchmark loss')
        trials.append({'repeat':repeat,'seconds':total,'training_seconds':training_seconds,'transfer_seconds':transfer_seconds,
            'resident_setup_seconds':resident_setup,'warmup_seconds':warm_seconds,'samples':steps*batch_size,
            'samples_per_second':steps*batch_size/total,'training_samples_per_second':steps*batch_size/training_seconds,
            'post_batch_loss':post_loss,'memory':runtime.memory()})
        last_state=cpu_state(model.state_dict())
        del model,optimizer,resident
        if runtime.device.type=='xpu':torch.xpu.empty_cache()
    median=lambda key:statistics.median(r[key] for r in trials)
    result={'status':'ok','device':device,'precision':precision,'cpu_threads':cpu_threads,'batch_size':batch_size,
        'mode':mode,'optimizer':optimizer_mode,'accumulation':accumulation,'finite_mode':finite_mode,'repeats':repeats,'steps':steps,'warmup':warmup,
        'seconds_median':median('seconds'),'training_seconds_median':median('training_seconds'),
        'transfer_seconds_median':median('transfer_seconds'),'samples_per_second_median':median('samples_per_second'),
        'training_samples_per_second_median':median('training_samples_per_second'),'post_batch_loss_median':median('post_batch_loss'),
        'all_losses_finite':True,'trials':trials,'synchronization':'torch.xpu.synchronize before/after transfer and complete training iteration'}
    return result,last_state


def _max_state_error(a,b):return max(float((a[k]-b[k]).abs().max()) for k in a)


def select_recommendation(results,signature,quality_tolerance=.02):
    """Only successful, finite, numerically acceptable measured rows can win."""
    eligible=[r for r in results if r.get('status')=='ok' and r.get('all_losses_finite') and
              (r.get('quality_relative_loss_delta') is not None and r['quality_relative_loss_delta']<=quality_tolerance) and
              r.get('mode') in ['copy','preallocated','resident']]
    if not eligible:raise RuntimeError('No finite compatible benchmark configuration is eligible')
    winner=max(eligible,key=lambda r:r['samples_per_second_median'])
    return {'format':'chromatic-arc-recommendation-v1','signature':signature,'device':winner['device'],'precision':winner['precision'],
        'cpu_threads':winner['cpu_threads'],'batch_size':winner['batch_size'],'transfer_mode':winner['mode'],
        'optimizer':winner['optimizer'],'accumulation':winner['accumulation'],
        'measurement':{k:winner[k] for k in ['seconds_median','samples_per_second_median','training_samples_per_second_median','repeats','steps']},
        'scope':'encoded-batch microbenchmark; full-pipeline results may select a different default',
        'model_size_specific':True,'model_implementation_sha256':model_implementation(),'quality_tolerance_relative_loss':quality_tolerance}


def read_recommendation(path,model_config,*,precision=None,cpu_threads=None):
    path=Path(path)
    if path.is_dir():path=path/'by-signature'/hardware_signature(model_config)['key']/'recommendation.json'
    if not path.is_file():return None
    saved=json.loads(path.read_text(encoding='utf-8'));current=hardware_signature(model_config)
    if saved.get('format')!='chromatic-arc-recommendation-v1' or saved.get('signature',{}).get('key')!=current['key']:return None
    if saved.get('model_implementation_sha256')!=model_implementation():return None
    if saved.get('device') not in ['cpu','xpu'] or saved.get('precision') not in ['fp32','bf16']:return None
    if precision is not None and saved['precision']!=precision:return None
    if cpu_threads is not None and saved.get('cpu_threads')!=cpu_threads:return None
    if saved.get('batch_size') not in BATCHES or not 1<=saved.get('cpu_threads',0)<=max(1,os.cpu_count() or 1):return None
    return saved


def recommend_for_constraints(matrix_path,model_config,*,cpu_threads,batch_size,precision='fp32'):
    """Respect user batch/thread choices; choose only the device from evidence."""
    path=Path(matrix_path)
    if not path.is_file():return None
    report=json.loads(path.read_text(encoding='utf-8'));signature=hardware_signature(model_config)
    if report.get('signature',{}).get('key')!=signature['key']:return None
    if not _bound_implementation(path,report):return None
    rows=[r for r in report.get('results',[]) if r.get('batch_size')==batch_size and r.get('cpu_threads')==cpu_threads
          and r.get('mode')=='copy' and r.get('optimizer')=='default' and r.get('accumulation')==1
          and r.get('finite_mode')=='auto' and (r.get('device')=='cpu' or r.get('precision')==precision)]
    if not rows:return None
    if available_xpu()[0] and not {'cpu','xpu'}<={r.get('device') for r in rows}:return None
    result=select_recommendation(rows,signature);result['scope']='same measured batch/thread constraints; device-only selection'
    return result


def run_matrix(initial_checkpoint,records_path,output,*,repeats=3,steps=15,warmup=2,include_xpu=True,ablations=True,deadline_seconds=240):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    initial,actual=checkpoint.load(initial_checkpoint);records=load_records(records_path);tensors=encode_batch(records)
    signature=hardware_signature(initial['config']);results=[];started=time.perf_counter();reference={}
    tasks=[]
    for threads in THREADS:
        if threads>max(1,os.cpu_count() or 1):continue
        for size in BATCHES:tasks.append({'device':'cpu','cpu_threads':threads,'batch_size':size})
    xpu_ok,xpu_error=available_xpu()
    if include_xpu and xpu_ok:
        for precision in ['fp32','bf16']:
            for size in BATCHES:tasks.append({'device':'xpu','precision':precision,'cpu_threads':1,'batch_size':size})
    elif include_xpu:results.append({'status':'unavailable','device':'xpu','error':xpu_error})
    if ablations:
        devices=[('cpu','fp32')]+([('xpu','fp32'),('xpu','bf16')] if include_xpu and xpu_ok else [])
        for device,precision in devices:
            base={'device':device,'precision':precision,'cpu_threads':1,'batch_size':256}
            tasks.extend([{**base,'mode':mode} for mode in ['preallocated','resident']])
            tasks.extend([{**base,'optimizer_mode':mode} for mode in ['loop','foreach','fused']])
            tasks.extend([{**base,'accumulation':n} for n in [2,4]])
            tasks.extend([{**base,'finite_mode':mode} for mode in ['reduce','concat']])
    for index,options in enumerate(tasks):
        if time.perf_counter()-started>deadline_seconds:
            results.append({'status':'not_run_time_budget','remaining_cases':len(tasks)-index});break
        try:
            row,state=benchmark_case(initial,tensors,repeats=repeats,steps=steps,warmup=warmup,**options)
            key=row['batch_size']
            if row['device']=='cpu' and row['cpu_threads']==1 and row['accumulation']==1 and row['optimizer']=='default' and row['mode']=='copy':
                reference[key]=(state,row['post_batch_loss_median'])
            if key in reference:
                ref,loss=reference[key];row['max_absolute_parameter_delta_cpu_reference']=_max_state_error(state,ref)
                row['quality_relative_loss_delta']=abs(row['post_batch_loss_median']-loss)/max(abs(loss),1e-8)
            else:row['quality_relative_loss_delta']=None
            results.append(row)
            print(json.dumps({'case':index+1,'of':len(tasks),**{k:row[k] for k in ['device','precision','cpu_threads','batch_size','mode','optimizer','accumulation','samples_per_second_median']}}),flush=True)
        except Exception as exc:
            row={'status':'error',**options,'error':f'{type(exc).__name__}: {exc}'};results.append(row)
            print(json.dumps({'case':index+1,**row}),flush=True)
        # Durable partial progress makes an interrupted experiment reviewable.
        atomic_json(output/'microbenchmark-progress.json',{'signature':signature,'results':results,'finished_cases':index+1,'planned_cases':len(tasks)})
    recommendation=select_recommendation(results,signature)
    report={'format':'chromatic-arc-microbenchmark-v1','signature':signature,'model_implementation_sha256':model_implementation(),'initial_checkpoint':str(actual),
        'initial_sha256':digest(actual),'records':str(Path(records_path).resolve()),'records_sha256':digest(records_path),
        'encoded_records':len(records),'repeats':repeats,'steps':steps,'warmup':warmup,'elapsed_seconds':time.perf_counter()-started,
        'results':results,'recommendation':recommendation,
        'meaning':'already encoded actual records; repeated identical mini-batch, no model persisted; pure step benchmark includes finite checks and clipping; not full-pipeline/chess-quality evidence'}
    atomic_json(output/'microbenchmark.json',report);atomic_json(output/'recommendation.json',recommendation)
    (output/'microbenchmark.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results),encoding='utf-8')
    keys=['status','device','precision','cpu_threads','batch_size','mode','optimizer','accumulation','finite_mode','repeats','steps',
          'seconds_median','training_seconds_median','transfer_seconds_median','samples_per_second_median','training_samples_per_second_median',
          'post_batch_loss_median','quality_relative_loss_delta','max_absolute_parameter_delta_cpu_reference','error']
    with (output/'microbenchmark.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(results)
    historical=output/'by-signature'/signature['key'];historical.mkdir(parents=True,exist_ok=True)
    for name in ['microbenchmark.json','microbenchmark.csv','microbenchmark.jsonl','recommendation.json']:
        shutil.copy2(output/name,historical/name)
    return report


def quick_recommend(initial,tensors,*,precision='fp32',cpu_threads=1,batch_size=256):
    """Small measured probe for auto when no current architecture result exists."""
    rows=[];reference_loss=None
    devices=['cpu']+(['xpu'] if available_xpu()[0] else [])
    for device in devices:
        actual_precision=precision if device=='xpu' else 'fp32'
        try:
            result,_=benchmark_case(initial,tensors,device=device,precision=actual_precision,cpu_threads=cpu_threads,
                batch_size=batch_size,repeats=2,steps=3,warmup=1,mode='copy')
            if device=='cpu':reference_loss=result['post_batch_loss_median']
            result['quality_relative_loss_delta']=abs(result['post_batch_loss_median']-reference_loss)/max(abs(reference_loss),1e-8) if reference_loss is not None else None
            rows.append(result)
        except Exception as exc:rows.append({'device':device,'status':'error','error':f'{type(exc).__name__}: {exc}'})
    result=select_recommendation(rows,hardware_signature(initial['config']))
    result['quick_probe_results']=rows
    return result


def compile_environment():
    """Detect missing prerequisites without initiating an expensive compile."""
    import importlib.util
    return {'official_windows_xpu_support_since':'PyTorch 2.7','torch':torch.__version__,
        'torch_compile_api':hasattr(torch,'compile'),'triton_importable':importlib.util.find_spec('triton') is not None,
        'cl_on_path':shutil.which('cl'),'clang_cl_on_path':shutil.which('clang-cl'),'cxx_environment':os.environ.get('CXX'),
        'default_enabled':False,'meaning':'environment inspection only; compile success/speed must be measured separately',
        'sources':['https://docs.pytorch.org/docs/main/notes/get_start_xpu.html',
                   'https://docs.pytorch.org/tutorials/unstable/inductor_windows.html']}
