"""Explicit CPU/XPU execution, portable states, and bounded device buffers.

No automatic install, compiler invocation, or silent mid-step device fallback.
An explicit XPU request fails unless the caller opted into CPU fallback.
"""
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import os
import platform
import time

import torch


class DeviceUnavailable(RuntimeError):
    pass


class NumericalError(RuntimeError):
    pass


def clip_gradients(parameters,max_norm=1.):
    """Avoid the measured XPU foreach-norm defect on this Arc/PyTorch stack.

    Keep the historical CPU path. The single-tensor XPU reductions were checked
    against a CPU float64 reference and the actual post-clipping norm.
    """
    parameters=list(parameters)
    xpu=any(p.grad is not None and p.grad.device.type=='xpu' for p in parameters)
    return torch.nn.utils.clip_grad_norm_(parameters,max_norm,error_if_nonfinite=True,foreach=False if xpu else None)


def cpu_state(value):
    """Snapshot all tensor leaves to independent CPU storage before saving."""
    if isinstance(value,torch.Tensor):return value.detach().to('cpu').clone()
    if isinstance(value,dict):return {k:cpu_state(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_state(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_state(v) for v in value)
    return value


def capture_xpu_rng():
    if hasattr(torch,'xpu') and torch.xpu.is_available():
        return [state.cpu().clone() for state in torch.xpu.get_rng_state_all()]
    return None


def restore_xpu_rng(state,strict=False):
    if state is None:return
    if hasattr(torch,'xpu') and torch.xpu.is_available():
        if len(state)!=torch.xpu.device_count():
            if strict:raise DeviceUnavailable('Saved XPU RNG device count differs from this machine')
            return
        torch.xpu.set_rng_state_all([s.cpu() for s in state]);return
    if strict:raise DeviceUnavailable('Saved XPU RNG requires an available XPU')


def optimizer_to_device(optimizer,device):
    """Move momentum tensors, retaining host scalar steps where AdamW requires."""
    device=torch.device(device)
    for group in optimizer.param_groups:
        device_step=bool(group.get('capturable',False) or group.get('fused',False))
        for parameter in group['params']:
            state=optimizer.state.get(parameter,{})
            for key,value in state.items():
                if isinstance(value,torch.Tensor):
                    destination=device if key!='step' or device_step else torch.device('cpu')
                    state[key]=value.to(destination)


def make_optimizer(model,lr=.001,mode='default',weight_decay=1e-4):
    if mode not in ['default','loop','foreach','fused']:raise ValueError('Unknown optimizer mode: '+str(mode))
    options={'loop':{'foreach':False},'foreach':{'foreach':True},'fused':{'fused':True},'default':{}}[mode]
    return torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay,**options)


def assert_finite(model,optimizer=None,include_gradients=False,mode='auto'):
    """One reduction per occupied device, avoiding a CPU sync for every tensor."""
    tensors=list(model.parameters())
    if include_gradients:tensors.extend(p.grad for p in model.parameters() if p.grad is not None)
    if optimizer is not None:
        tensors.extend(v for values in optimizer.state.values() for v in values.values() if isinstance(v,torch.Tensor))
    if mode not in ['auto','reduce','concat']:raise ValueError('Unknown finite-check mode')
    grouped={}
    for value in tensors:
        grouped.setdefault(str(value.device),[]).append(value)
    for device,values in grouped.items():
        concatenate=mode=='concat' or (mode=='auto' and device.startswith('xpu'))
        if concatenate:
            ok=torch.isfinite(torch.cat([v.detach().reshape(-1) for v in values])).all()
        else:ok=torch.stack([torch.isfinite(v).all() for v in values]).all()
        if not bool(ok):raise NumericalError('NaN/Inf model, gradient, or optimizer state on '+device)


def available_xpu():
    try:
        ok=bool(hasattr(torch,'xpu') and torch.xpu.is_available())
        return ok,None if ok else 'torch.xpu.is_available() is false'
    except Exception as exc:return False,f'{type(exc).__name__}: {exc}'


def hardware_signature(model_config=None):
    ok,error=available_xpu()
    gpu=[]
    if ok:
        for index in range(torch.xpu.device_count()):
            p=torch.xpu.get_device_properties(index)
            gpu.append({'name':torch.xpu.get_device_name(index),'total_memory':getattr(p,'total_memory',None),
                        'platform_name':getattr(p,'platform_name',None),'driver_version':getattr(p,'driver_version',None)})
    if hasattr(model_config,'__dict__'):model_config=vars(model_config)
    info={'schema':1,'platform':platform.platform(),'processor':platform.processor(),'logical_cpus':os.cpu_count(),
          'torch':torch.__version__,'model_config':model_config or {},'inputs':1280,'policy_actions':4672,'xpu':gpu,
          'xpu_error':None if ok else error}
    encoded=json.dumps(info,sort_keys=True,separators=(',',':')).encode()
    return {'key':hashlib.sha256(encoded).hexdigest(),'details':info}


@dataclass
class DeviceRuntime:
    requested:str='cpu'
    precision:str='fp32'
    cpu_threads:int=1
    fallback:str='error'
    recommendation:dict|None=None

    def __post_init__(self):
        if self.requested not in ['auto','cpu','xpu']:raise ValueError('Device must be auto, cpu, or xpu')
        if self.precision not in ['fp32','bf16']:raise ValueError('Precision must be fp32 or bf16')
        if not isinstance(self.cpu_threads,int) or not 1<=self.cpu_threads<=max(1,os.cpu_count() or 1):
            raise ValueError('CpuThreads must be between 1 and the logical CPU count')
        if self.fallback not in ['error','cpu']:raise ValueError('Fallback must be error or cpu')
        torch.set_num_threads(self.cpu_threads)
        self.decision={'requested':self.requested,'requested_precision':self.precision,'cpu_threads':self.cpu_threads,
                       'fallback_policy':self.fallback,'reason':None,'probe':None}
        selected=self.requested
        if selected=='auto':
            # The owner validates architecture+hardware signature before passing
            # a recommendation. No unmeasured GPU-speed assertion is made here.
            if self.recommendation:
                selected=self.recommendation.get('device','cpu')
                self.decision['reason']='measured matching recommendation supplied by caller'
            else:
                selected='cpu';self.decision['reason']='no matching measured recommendation; conservative CPU'
        if selected=='xpu':
            ok,error=available_xpu()
            if ok:
                try:
                    start=time.perf_counter();p=torch.tensor([1.,2.],device='xpu',requires_grad=True)
                    loss=p.square().sum();loss.backward();torch.xpu.synchronize()
                    if not bool(torch.isfinite(p.grad).all()):raise NumericalError('XPU smoke gradient nonfinite')
                    self.decision['probe']={'forward_backward_ok':True,'seconds':time.perf_counter()-start}
                except Exception as exc:ok=False;error=f'{type(exc).__name__}: {exc}'
            if not ok:
                if self.fallback!='cpu':raise DeviceUnavailable('XPU execution unavailable: '+str(error)+'; request CPU or opt into --fallback cpu')
                selected='cpu';self.precision='fp32';self.decision['reason']='explicit CPU fallback: '+str(error)
        if selected=='cpu' and self.precision!='fp32':
            if self.fallback=='cpu':self.precision='fp32';self.decision['reason']='CPU fallback uses FP32'
            else:raise ValueError('BF16 mode is validated for XPU only; CPU runs require fp32')
        self.device=torch.device(selected)
        self.decision.update(device=str(self.device),precision=self.precision)
        if self.decision['reason'] is None:self.decision['reason']='explicit device request'

    def synchronize(self):
        if self.device.type=='xpu':torch.xpu.synchronize(self.device)

    def autocast(self):
        return torch.autocast('xpu',dtype=torch.bfloat16) if self.device.type=='xpu' and self.precision=='bf16' else nullcontext()

    def transfer(self,batch,non_blocking=False):
        return tuple(t.to(self.device,non_blocking=non_blocking) for t in batch)

    def seed(self,seed):
        torch.manual_seed(seed)
        if self.device.type=='xpu':torch.xpu.manual_seed_all(seed)

    def capture_rng(self):
        return {'torch':torch.get_rng_state().cpu().clone(),'xpu':capture_xpu_rng() if self.device.type=='xpu' else None}

    def restore_rng(self,state,strict=False):
        torch.set_rng_state(state['torch'].cpu());restore_xpu_rng(state.get('xpu'),strict=strict)

    def memory(self):
        if self.device.type!='xpu':return {'allocated_bytes':0,'reserved_bytes':0,'peak_allocated_bytes':0,'peak_reserved_bytes':0}
        return {'allocated_bytes':torch.xpu.memory_allocated(self.device),'reserved_bytes':torch.xpu.memory_reserved(self.device),
                'peak_allocated_bytes':torch.xpu.max_memory_allocated(self.device),'peak_reserved_bytes':torch.xpu.max_memory_reserved(self.device)}

    def reset_peak_memory(self):
        if self.device.type=='xpu':torch.xpu.reset_peak_memory_stats(self.device)

    def fallback_fp32(self,reason):
        """Caller must roll back the failed batch before retrying, never keep NaNs."""
        self.precision='fp32';self.decision.update(precision='fp32',precision_fallback_reason=str(reason))


class BatchTransfer:
    """Reusable device buffers for one batch; no unsafe overlap or hidden cache."""
    def __init__(self,runtime,preallocate=True):
        self.runtime=runtime;self.preallocate=preallocate;self.buffers=None

    def __call__(self,batch):
        if self.runtime.device.type=='cpu':return batch
        if not self.preallocate:return self.runtime.transfer(batch)
        if self.buffers is None or any(b.shape!=a.shape or b.dtype!=a.dtype for a,b in zip(batch,self.buffers)):
            self.buffers=tuple(torch.empty_like(t,device=self.runtime.device) for t in batch)
        for source,destination in zip(batch,self.buffers):destination.copy_(source,non_blocking=False)
        return self.buffers
