"""Optional, exclusive wall-time stage accounting and bounded resource sampling.

XPU time spans may request an explicit synchronize callback. Resource counters
are observations, not device execution timers. Concurrent stage sums can exceed
elapsed wall time. Disabled hooks never synchronize or sample hardware.
"""
from __future__ import annotations

from contextlib import ContextDecorator
from collections import defaultdict
import csv
import functools
import json
import os
from pathlib import Path
import threading
import time

_active=None

class _Nothing:
    def __enter__(self):return self
    def __exit__(self,*args):return False
_NOTHING=_Nothing()

def install_profile(profile):
    global _active
    previous=_active
    _active=profile if profile is not None and profile.enabled else None
    return previous

def get_profile():return _active

def span(stage,**kwargs):
    profile=_active
    return profile.span(stage,**kwargs) if profile is not None else _NOTHING

def profiled(stage):
    def decorate(function):
        @functools.wraps(function)
        def wrapped(*args,**kwargs):
            profile=_active
            if profile is None:return function(*args,**kwargs)
            with profile.span(stage):return function(*args,**kwargs)
        return wrapped
    return decorate

class _Span:
    def __init__(self,profile,stage,items,bytes_count,sync):
        self.profile=profile;self.stage=str(stage);self.items=items;self.bytes_count=bytes_count;self.sync=sync
    def __enter__(self):
        if self.sync and self.profile.synchronize is not None:self.profile.synchronize()
        self.started=time.perf_counter();self.child=0.
        stack=getattr(self.profile.local,'stack',None)
        if stack is None:stack=[];self.profile.local.stack=stack
        stack.append(self)
        return self
    def __exit__(self,typ,value,tb):
        if self.sync and self.profile.synchronize is not None:self.profile.synchronize()
        elapsed=time.perf_counter()-self.started
        stack=self.profile.local.stack
        if not stack or stack[-1] is not self:raise RuntimeError('Profile spans must close in stack order')
        stack.pop()
        if stack:stack[-1].child+=elapsed
        self.profile.record(self.stage,max(0.,elapsed-self.child),items=self.items,
                            bytes_count=self.bytes_count,inclusive_seconds=elapsed,error=typ is not None)
        return False

class Profile:
    def __init__(self,enabled=False,output_directory=None,synchronize=None):
        self.enabled=bool(enabled);self.output_directory=Path(output_directory) if output_directory else None
        self.synchronize=synchronize;self.started=time.perf_counter();self.local=threading.local();self.lock=threading.Lock()
        self.stages={};self.events=0;self.stream=None
        if self.enabled and self.output_directory:
            self.output_directory.mkdir(parents=True,exist_ok=True)
            self.stream=(self.output_directory/'stages.jsonl').open('a',encoding='utf-8',buffering=65536)
    def span(self,stage,*,items=0,bytes_count=0,sync=False):
        return _Span(self,stage,items,bytes_count,sync) if self.enabled else _NOTHING
    def record(self,stage,seconds,*,items=0,bytes_count=0,inclusive_seconds=None,error=False):
        if not self.enabled:return
        event={'stage':str(stage),'seconds':float(seconds),'inclusive_seconds':float(seconds if inclusive_seconds is None else inclusive_seconds),
               'items':int(items or 0),'bytes':int(bytes_count or 0),'error':bool(error),
               'elapsed_seconds':time.perf_counter()-self.started,'thread':threading.current_thread().name}
        with self.lock:
            totals=self.stages.setdefault(str(stage),{'seconds':0.,'inclusive_seconds':0.,'calls':0,'items':0,'bytes':0,'errors':0})
            for key in ['seconds','inclusive_seconds','items','bytes']:totals[key]+=event[key]
            totals['calls']+=1;totals['errors']+=int(error);self.events+=1
            if self.stream:self.stream.write(json.dumps(event,separators=(',',':'))+'\n')
    def snapshot(self):
        with self.lock:
            return {'enabled':self.enabled,'elapsed_seconds':time.perf_counter()-self.started,'events':self.events,
                'stages':{k:dict(v) for k,v in sorted(self.stages.items())},
                'timing_contract':'seconds are exclusive wall time; inclusive_seconds include nested spans; concurrent thread sums may exceed elapsed wall time',
                'xpu_synchronized_spans':self.synchronize is not None}
    def write(self,path=None):
        from .storage import atomic_json
        result=self.snapshot()
        if self.stream:self.stream.flush()
        target=Path(path) if path else self.output_directory/'profile.json' if self.output_directory else None
        if target:atomic_json(target,result)
        return result
    def close(self):
        if self.stream:self.stream.flush();self.stream.close();self.stream=None

class _WindowsGpuCounters:
    """PDH process busiest-engine %, matching Task Manager's process meaning.

    Primary API: https://learn.microsoft.com/en-us/windows/win32/api/pdh/nf-pdh-pdhgetformattedcounterarrayw
    Interpretation: https://devblogs.microsoft.com/directx/gpus-in-the-task-manager/
    """
    def __init__(self,pid):
        if os.name!='nt':raise RuntimeError('Windows PDH GPU Engine counters are unavailable on this platform')
        import ctypes as c
        from ctypes import wintypes as w
        self.c=c;self.w=w;self.pid=pid;self.dll=c.WinDLL('pdh.dll');self.query=w.HANDLE();self.counter=w.HANDLE();self.last_status=None
        class Value(c.Structure):_fields_=[('status',w.DWORD),('value',c.c_double)]
        class Item(c.Structure):_fields_=[('name',w.LPWSTR),('value',Value)]
        self.Item=Item
        signatures={'PdhOpenQueryW':([w.LPCWSTR,c.c_size_t,c.POINTER(w.HANDLE)],w.LONG),
            'PdhAddEnglishCounterW':([w.HANDLE,w.LPCWSTR,c.c_size_t,c.POINTER(w.HANDLE)],w.LONG),
            'PdhCollectQueryData':([w.HANDLE],w.LONG),
            'PdhGetFormattedCounterArrayW':([w.HANDLE,w.DWORD,c.POINTER(w.DWORD),c.POINTER(w.DWORD),c.c_void_p],w.LONG),
            'PdhCloseQuery':([w.HANDLE],w.LONG)}
        for name,(args,result) in signatures.items():getattr(self.dll,name).argtypes=args;getattr(self.dll,name).restype=result
        status=self.dll.PdhOpenQueryW(None,0,c.byref(self.query))
        if status:raise RuntimeError(f'PdhOpenQuery failed 0x{status & 0xffffffff:08x}')
        status=self.dll.PdhAddEnglishCounterW(self.query,r'\GPU Engine(*)\Utilization Percentage',0,c.byref(self.counter))
        if status:
            self.close();raise RuntimeError(f'GPU Engine counter unavailable 0x{status & 0xffffffff:08x}')
        self.dll.PdhCollectQueryData(self.query)
    def read(self):
        c=self.c;w=self.w;status=self.dll.PdhCollectQueryData(self.query)
        transient=(0x800007D1,0x800007D3,0x800007D5,0xC0000BBA,0xC0000BC6)
        if status & 0xffffffff in transient:
            self.last_status=f'GPU counter rate sample is not yet available (PDH 0x{status & 0xffffffff:08x})';return None,0
        if status:raise RuntimeError(f'PDH collection failed 0x{status & 0xffffffff:08x}')
        size=w.DWORD();count=w.DWORD()
        status=self.dll.PdhGetFormattedCounterArrayW(self.counter,0x200,c.byref(size),c.byref(count),None)
        if status & 0xffffffff in transient:
            self.last_status=f'GPU counter rate sample is not yet available (PDH 0x{status & 0xffffffff:08x})';return None,0
        if status & 0xffffffff not in (0,0x800007D2):raise RuntimeError(f'GPU Engine query failed 0x{status & 0xffffffff:08x}')
        if not size.value:return None,0
        buffer=c.create_string_buffer(size.value)
        status=self.dll.PdhGetFormattedCounterArrayW(self.counter,0x200,c.byref(size),c.byref(count),buffer)
        if status & 0xffffffff in transient:
            self.last_status=f'GPU counter rate sample is not yet available (PDH 0x{status & 0xffffffff:08x})';return None,0
        if status:raise RuntimeError(f'GPU Engine values failed 0x{status & 0xffffffff:08x}')
        values=[];prefix=f'pid_{self.pid}_'
        for item in c.cast(buffer,c.POINTER(self.Item*count.value)).contents:
            if item.name and item.name.startswith(prefix) and item.value.status in (0,1):values.append(max(0.,min(100.,item.value.value)))
        self.last_status=None if values else 'GPU Engine counters are readable, but no current-PID engine instance has a valid sample'
        return max(values) if values else None,len(values)
    def close(self):
        if self.query:self.dll.PdhCloseQuery(self.query);self.query=self.w.HANDLE()

class ResourceSampler:
    def __init__(self,interval=.2,device='cpu',output_directory=None,max_retained_samples=64):
        if not .02<=float(interval)<=60:raise ValueError('Resource sampling interval must be between .02 and 60 seconds')
        if device not in ('cpu','xpu'):raise ValueError('Resource device must be cpu or xpu')
        self.interval=float(interval);self.device=device;self.output_directory=Path(output_directory) if output_directory else None
        self.max_retained_samples=max(0,int(max_retained_samples));self.samples=[];self.sample_count=0;self.sums=defaultdict(float);self.peaks=defaultdict(float);self.valid_counts=defaultdict(int)
        self.stop_event=threading.Event();self.thread=None;self.lock=threading.Lock();self.gpu=None;self.gpu_reason=None;self.errors=[]
        self.stream=None;self.csv_stream=None;self.writer=None;self.started=None
    def start(self):
        import psutil
        if self.thread is not None:raise RuntimeError('ResourceSampler is already started')
        self.process=psutil.Process();self.psutil=psutil;self.started=time.perf_counter();self.initial_io=self.process.io_counters()
        self.process.cpu_percent(None);psutil.cpu_percent(None)
        if self.device=='xpu':
            try:self.gpu=_WindowsGpuCounters(self.process.pid)
            except Exception as exc:self.gpu_reason=f'{type(exc).__name__}: {exc}'
        else:self.gpu_reason='CPU run; GPU utilization is not attributed to this process'
        if self.output_directory:
            self.output_directory.mkdir(parents=True,exist_ok=True)
            self.stream=(self.output_directory/'resources.jsonl').open('w',encoding='utf-8',buffering=65536)
            self.csv_stream=(self.output_directory/'resources.csv').open('w',encoding='utf-8',newline='',buffering=65536)
        self._sample()
        self.thread=threading.Thread(target=self._loop,name='arc-resource-sampler',daemon=True);self.thread.start()
        return self
    def _sample(self):
        io=self.process.io_counters();row={'elapsed_seconds':time.perf_counter()-self.started,
            'cpu_process_percent':self.process.cpu_percent(None),'cpu_system_percent':self.psutil.cpu_percent(None),
            'rss_bytes':self.process.memory_info().rss,
            'io_read_bytes':max(0,io.read_bytes-self.initial_io.read_bytes),'io_write_bytes':max(0,io.write_bytes-self.initial_io.write_bytes),
            'io_read_operations':max(0,io.read_count-self.initial_io.read_count),'io_write_operations':max(0,io.write_count-self.initial_io.write_count),
            'gpu_busiest_engine_percent':None,'gpu_matching_engine_count':0,'xpu_allocated_bytes':None,'xpu_reserved_bytes':None,
            'xpu_peak_allocated_bytes':None,'xpu_peak_reserved_bytes':None}
        if self.gpu is not None:
            try:
                row['gpu_busiest_engine_percent'],row['gpu_matching_engine_count']=self.gpu.read()
                self.gpu_reason=self.gpu.last_status
            except Exception as exc:self.gpu_reason=f'{type(exc).__name__}: {exc}';self.gpu.close();self.gpu=None
        if self.device=='xpu':
            try:
                import torch
                for key,fn in [('xpu_allocated_bytes','memory_allocated'),('xpu_reserved_bytes','memory_reserved'),
                               ('xpu_peak_allocated_bytes','max_memory_allocated'),('xpu_peak_reserved_bytes','max_memory_reserved')]:
                    row[key]=getattr(torch.xpu,fn)()
            except Exception as exc:
                if not self.errors:self.errors.append(f'XPU allocator metrics unavailable: {exc}')
        with self.lock:
            self.sample_count+=1
            for key,value in row.items():
                if isinstance(value,(int,float)):
                    self.sums[key]+=value;self.peaks[key]=max(self.peaks[key],value);self.valid_counts[key]+=1
            if self.max_retained_samples:
                self.samples.append(row)
                if len(self.samples)>self.max_retained_samples:del self.samples[0]
            if self.stream:self.stream.write(json.dumps(row,separators=(',',':'))+'\n')
            if self.csv_stream:
                if self.writer is None:self.writer=csv.DictWriter(self.csv_stream,fieldnames=list(row));self.writer.writeheader()
                self.writer.writerow(row)
        return row
    def _loop(self):
        while not self.stop_event.wait(self.interval):
            try:self._sample()
            except Exception as exc:
                with self.lock:self.errors.append(f'{type(exc).__name__}: {exc}')
                break
    def stop(self):
        if self.thread is not None:
            self.stop_event.set();self.thread.join(timeout=max(2.,self.interval+1))
            self._sample();self.thread=None
        if self.gpu:self.gpu.close();self.gpu=None
        for file in [self.stream,self.csv_stream]:
            if file:file.flush();file.close()
        self.stream=self.csv_stream=None
        result=self.snapshot()
        if self.output_directory:
            from .storage import atomic_json
            atomic_json(self.output_directory/'resource-summary.json',result)
        return result
    def snapshot(self):
        with self.lock:
            gpu_samples=self.valid_counts['gpu_busiest_engine_percent']
            return {'samples':self.sample_count,'interval_seconds':self.interval,'device':self.device,
                'elapsed_seconds':time.perf_counter()-self.started if self.started is not None else 0.,
                'means':{k:v/self.valid_counts[k] for k,v in self.sums.items()} if self.sample_count else {},'peaks':dict(self.peaks),
                'gpu_utilization_available':gpu_samples>0,'gpu_unavailable_reason':None if gpu_samples else self.gpu_reason or 'No matching GPU engine instance observed for this process',
                'gpu_utilization_meaning':'Windows PDH busiest engine for current PID; not EU occupancy; unavailable is not zero',
                'cpu_process_percent_meaning':'psutil process percentage may exceed 100 when several CPU cores execute',
                'shared_memory_note':'XPU allocator bytes and process RSS may describe overlapping physical memory on integrated GPUs; do not add them',
                'io_meaning':'process I/O byte counters, not physical drive traffic',
                'errors':list(self.errors),'retained_tail_samples':list(self.samples)}
