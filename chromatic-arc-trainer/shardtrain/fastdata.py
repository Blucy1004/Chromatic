"""Verified, bounded binary encoding cache and ordered CPU prefetch.

The cache stores compact legal masks and sparse policy labels. Dense policy and
legal tensors exist only for requested batches. Model architecture/encoding and
all normalized record bytes contribute to the immutable cache identity.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import threading
import uuid

from .model_adapter import np,torch,INPUTS,ACTIONS,batch as reference_batch
from .storage import atomic_json,canonical,digest
from .profiling import span

SCHEMA=1
POLICY_SUPPORT=256
MASK_BYTES=(ACTIONS+7)//8
ROW_DTYPE=np.dtype([('x','<f4',(INPUTS,)),('legal','u1',(MASK_BYTES,)),
    ('policy_actions','<i2',(POLICY_SUPPORT,)),('policy_prob','<f4',(POLICY_SUPPORT,)),
    ('wdl','<f4',(3,)),('pw','<f4'),('vw','<f4')])
FIELDS=('x','p','v','mask','pw','vw')

def encoding_signature():
    root=Path(__file__).resolve().parents[1]
    return hashlib.sha256((digest(root/'vendor/zero/encoding.py')+f':{INPUTS}:{ACTIONS}:{SCHEMA}:{ROW_DTYPE.descr}').encode()).hexdigest()

def record_fingerprint(record):return hashlib.sha256(canonical(record).encode('utf-8')).hexdigest()

def _checked_indices(indices,length):
    indexes=np.asarray(indices,dtype=np.int64)
    if indexes.ndim!=1 or (indexes<0).any() or (indexes>=length).any():raise IndexError('Encoded cache index is outside the dataset')
    return indexes

class BatchBuffer:
    """Reusable dense CPU arrays. Returned views expire on the next fill.

    Callers must not overwrite a buffer while an asynchronous transfer still
    reads it. Prefetch defaults to independent ownership instead of reusing it.
    """
    def __init__(self,capacity):
        if int(capacity)<1:raise ValueError('Batch buffer capacity must be positive')
        self.capacity=int(capacity)
        self.arrays=(np.empty((capacity,INPUTS),np.float32),np.zeros((capacity,ACTIONS),np.float32),
            np.empty((capacity,3),np.float32),np.empty((capacity,ACTIONS),np.bool_),
            np.empty(capacity,np.float32),np.empty(capacity,np.float32))
    @property
    def nbytes(self):return sum(a.nbytes for a in self.arrays)
    def views(self,size):
        if not 0<=size<=self.capacity:raise ValueError('Requested batch exceeds preallocated capacity')
        return tuple(a[:size] for a in self.arrays)

def _encode(records):
    """Use the preserved encoder as the validation authority exactly once."""
    if any(len(r.get('policy',[]))>POLICY_SUPPORT for r in records):raise ValueError('Binary policy support exceeds the bounded 256-move schema')
    x,p,v,mask,pw,vw=reference_batch(records)
    out=np.empty(len(records),dtype=ROW_DTYPE);out['x']=x.numpy();out['legal']=np.packbits(mask.numpy(),axis=1,bitorder='little')
    out['policy_actions'].fill(-1);out['policy_prob'].fill(0)
    for i,row in enumerate(records):
        # Reference encoding adds duplicate action entries; retain that exact
        # float32 result rather than rebuilding a different reduction order.
        nonzero=np.flatnonzero(p[i].numpy())
        if len(nonzero)>POLICY_SUPPORT:raise ValueError('Too many nonzero policy labels')
        out['policy_actions'][i,:len(nonzero)]=nonzero
        out['policy_prob'][i,:len(nonzero)]=p[i].numpy()[nonzero]
    out['wdl']=v.numpy();out['pw']=pw.numpy();out['vw']=vw.numpy()
    return out

def _expand(encoded,out=None):
    n=len(encoded)
    arrays=out.views(n) if out is not None else (np.empty((n,INPUTS),np.float32),np.zeros((n,ACTIONS),np.float32),
        np.empty((n,3),np.float32),np.empty((n,ACTIONS),np.bool_),np.empty(n,np.float32),np.empty(n,np.float32))
    x,p,v,mask,pw,vw=arrays
    x[:]=encoded['x'];p.fill(0);v[:]=encoded['wdl'];pw[:]=encoded['pw'];vw[:]=encoded['vw']
    mask[:]=np.unpackbits(encoded['legal'],axis=1,count=ACTIONS,bitorder='little')
    actions=encoded['policy_actions'];valid=actions>=0
    rows=np.broadcast_to(np.arange(n)[:,None],actions.shape)
    p[rows[valid],actions[valid]]=encoded['policy_prob'][valid]
    return tuple(torch.from_numpy(a) for a in arrays)

class EncodedSet:
    def __init__(self,identities,array,mode,*,fingerprints=None,release=None,raw_records=None):
        self.identities=tuple(identities);self.id_to_index={identity:i for i,identity in enumerate(self.identities)}
        if len(self.id_to_index)!=len(self.identities):raise ValueError('Encoded sets require unique record identities')
        self.array=array;self.mode=mode;self.fingerprints=tuple(fingerprints or ())
        self.raw_records=raw_records;self.release=release;self.closed=False
    def __len__(self):return len(self.identities)
    @property
    def nbytes(self):return int(self.array.nbytes) if self.array is not None else 0
    def batch(self,indices,out=None):
        if self.closed:raise RuntimeError('Encoded dataset is closed')
        indexes=_checked_indices(indices,len(self))
        with span('data_loading',items=len(indexes)):
            if self.mode=='none':
                tensors=reference_batch([self.raw_records[int(i)] for i in indexes])
                if out is None:return tensors
                arrays=out.views(len(indexes))
                for a,t in zip(arrays,tensors):np.copyto(a,t.numpy())
                return tuple(torch.from_numpy(a) for a in arrays)
            return _expand(self.array[indexes],out)
    take=batch
    def batch_by_ids(self,identities,out=None):return self.batch([self.id_to_index[i] for i in identities],out)
    def close(self):
        if self.closed:return
        size=self.nbytes
        if isinstance(self.array,np.memmap):self.array._mmap.close()
        self.array=None;self.raw_records=None;self.closed=True
        self.id_to_index.clear();self.identities=();self.fingerprints=()
        if self.release:self.release(size,self.mode);self.release=None
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

class EncodedCache:
    def __init__(self,directory,mode='auto',memory_budget_bytes=64*1024**2,guard=None,profile=None,
                 chunk_size=128,hot_cache_bytes=8*1024**2):
        if mode not in ('none','memory','mmap','auto'):raise ValueError('CacheMode must be none, memory, mmap or auto')
        if int(memory_budget_bytes)<0 or not 1<=int(chunk_size)<=4096:raise ValueError('Invalid binary cache resource budget')
        self.directory=Path(directory).resolve();self.directory.mkdir(parents=True,exist_ok=True)
        self.mode=mode;self.memory_budget_bytes=int(memory_budget_bytes);self.guard=guard;self.profile=profile
        self.chunk_size=int(chunk_size);self.resident_bytes=0;self.sets=[];self.signature=encoding_signature()
        self.hot=OrderedDict();self.hot_bytes_limit=max(0,min(int(hot_cache_bytes),self.memory_budget_bytes));self.hot_bytes=0
        self.stats={'builds':0,'disk_hits':0,'memory_sets':0,'mmap_sets':0,'get_batch_hits':0,'get_batch_misses':0,'hot_evictions':0}
        self.lock=threading.RLock()
    def _paths(self,key):
        token=hashlib.sha256(str(key).encode('utf-8')).hexdigest()
        return self.directory/f'encoded-{token}.npy',self.directory/f'encoded-{token}.json'
    def _reserve(self,extra_disk=0):
        if self.guard:self.guard.check(extra_disk=extra_disk)
    def _release(self,nbytes,mode):
        if mode=='memory':self.resident_bytes=max(0,self.resident_bytes-nbytes)
    def prepare(self,key,records,expected_source_sha=None):
        self.sets=[dataset for dataset in self.sets if not dataset.closed]
        records=records if isinstance(records,(list,tuple)) else list(records)
        identities=[r['identity'] for r in records]
        if len(set(identities))!=len(identities):raise ValueError('Duplicate identities in an encoded set')
        if self.mode=='none':
            result=EncodedSet(identities,None,'none',raw_records=records);self.sets.append(result);return result
        fingerprints=[record_fingerprint(r) for r in records]
        contract={'schema':SCHEMA,'encoding_signature':self.signature,'identities':identities,'fingerprints':fingerprints,
                  'source_sha256':expected_source_sha,'rows':len(records),'row_bytes':ROW_DTYPE.itemsize}
        signature=hashlib.sha256(canonical(contract).encode()).hexdigest()
        path,manifest=self._paths(key)
        with span('binary_cache_prepare',items=len(records)):
            valid=False
            if manifest.is_file() and path.is_file():
                meta=json.loads(manifest.read_text(encoding='utf-8'))
                if meta.get('contract_sha256')==signature:
                    if digest(path)!=meta.get('file_sha256'):raise RuntimeError('Encoded binary cache checksum mismatch; refusing stale/corrupt tensors')
                    valid=True;self.stats['disk_hits']+=1
            if not valid:
                # A complete new binary and manifest are committed in that
                # order; an interrupted build is never reused from a temp file.
                temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
                estimate=len(records)*ROW_DTYPE.itemsize+4096
                self._reserve(estimate)
                mapped=None
                try:
                    mapped=np.lib.format.open_memmap(temp,mode='w+',dtype=ROW_DTYPE,shape=(len(records),))
                    for start in range(0,len(records),self.chunk_size):
                        self._reserve()
                        with span('binary_encode',items=min(self.chunk_size,len(records)-start)):
                            mapped[start:start+self.chunk_size]=_encode(records[start:start+self.chunk_size])
                    mapped.flush();mapped._mmap.close();mapped=None
                    with temp.open('r+b') as f:os.fsync(f.fileno())
                    os.replace(temp,path)
                    metadata={'contract':contract,'contract_sha256':signature,'file_sha256':digest(path),'bytes':path.stat().st_size}
                    self._reserve(len(canonical(metadata).encode('utf-8'))+4096)
                    atomic_json(manifest,metadata)
                    self.stats['builds']+=1
                finally:
                    if mapped is not None:mapped._mmap.close()
                    if temp.exists():temp.unlink()
            loaded=np.load(path,mmap_mode='r',allow_pickle=False)
            if loaded.dtype!=ROW_DTYPE or loaded.shape!=(len(records),):
                loaded._mmap.close();raise RuntimeError('Encoded binary schema or row count mismatch')
            available=max(0,self.memory_budget_bytes-self.resident_bytes-self.hot_bytes)
            memory=self.mode=='memory' or (self.mode=='auto' and loaded.nbytes<=available)
            if memory:
                if loaded.nbytes>available:
                    loaded._mmap.close();raise RuntimeError('Explicit memory CacheMode exceeds its bounded cache budget')
                self._reserve();array=np.array(loaded,copy=True);loaded._mmap.close()
                self.resident_bytes+=array.nbytes;mode='memory';self.stats['memory_sets']+=1
            else:array=loaded;mode='mmap';self.stats['mmap_sets']+=1
            result=EncodedSet(identities,array,mode,fingerprints=fingerprints,release=self._release)
            self.sets.append(result);return result
    def get_batch(self,records,out=None):
        """Resolve current shard + replay rows by ID, encoding misses once.

        Only normalized metadata are hashed on a cache hit. The expensive FEN,
        legal-move, history and feature work has already passed the reference
        encoder. FIFO/prefetch workers must receive a precomputed record list;
        this method never touches a training RNG or SQLite connection.
        """
        if self.mode=='none':return reference_batch(records)
        if not records:return _expand(np.empty(0,dtype=ROW_DTYPE),out)
        with self.lock,span('data_loading',items=len(records)):
            rows=[]
            for record in records:
                identity=record['identity'];fingerprint=record_fingerprint(record);found=None
                for dataset in reversed(self.sets):
                    if dataset.closed or dataset.mode=='none':continue
                    index=dataset.id_to_index.get(identity)
                    if index is not None and dataset.fingerprints[index]==fingerprint:
                        found=dataset.array[index];break
                if found is None:
                    hot_key=(identity,fingerprint)
                    if hot_key in self.hot:
                        found=self.hot.pop(hot_key);self.hot[hot_key]=found
                    else:
                        self.stats['get_batch_misses']+=1
                        found=_encode([record])[0].copy()
                        while self.hot and self.hot_bytes+ROW_DTYPE.itemsize>min(self.hot_bytes_limit,max(0,self.memory_budget_bytes-self.resident_bytes)):
                            self.hot.popitem(last=False);self.hot_bytes-=ROW_DTYPE.itemsize;self.stats['hot_evictions']+=1
                        if self.hot_bytes+ROW_DTYPE.itemsize<=min(self.hot_bytes_limit,max(0,self.memory_budget_bytes-self.resident_bytes)):
                            self.hot[hot_key]=found;self.hot_bytes+=ROW_DTYPE.itemsize
                        rows.append(found);continue
                self.stats['get_batch_hits']+=1;rows.append(found)
            return _expand(np.asarray(rows,dtype=ROW_DTYPE),out)
    def remove(self,key):
        """Delete only this key's disposable verified binary; close views first."""
        path,manifest=self._paths(key)
        if path.exists():
            for dataset in self.sets:
                if not dataset.closed and isinstance(dataset.array,np.memmap) and Path(dataset.array.filename).resolve()==path:
                    raise RuntimeError('Close the mmap encoded set before removing its cache')
        if manifest.is_file():
            meta=json.loads(manifest.read_text(encoding='utf-8'))
            if path.is_file() and digest(path)!=meta['file_sha256']:raise RuntimeError('Changed cache retained for inspection')
        for target in [path,manifest]:
            if target.exists():target.unlink()
    def snapshot(self):return {**self.stats,'resident_bytes':self.resident_bytes,'hot_bytes':self.hot_bytes,'budget_bytes':self.memory_budget_bytes}
    def close(self):
        for dataset in self.sets:dataset.close()
        self.sets.clear();self.hot.clear();self.hot_bytes=0
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

class Prefetch:
    """Bounded, ordered producer/consumer wrapper; worker owns no training RNG.

    The consumer must close this wrapper on early exit. Producer errors are
    rethrown at the original sequence position. Depth zero is synchronous.
    """
    def __init__(self,iterable,prepare_fn=lambda x:x,depth=0):
        if not 0<=int(depth)<=16:raise ValueError('Prefetch depth must be between 0 and 16')
        self.source=iter(iterable);self.prepare=prepare_fn;self.depth=int(depth);self.closed=False
        self.stop_event=threading.Event();self.thread=None;self.stats={'prepared':0,'yielded':0,'max_queue':0}
        self.queue=queue.Queue(maxsize=self.depth) if self.depth else None
    def _put(self,item):
        while not self.stop_event.is_set():
            try:self.queue.put(item,timeout=.05);self.stats['max_queue']=max(self.stats['max_queue'],self.queue.qsize());return True
            except queue.Full:pass
        return False
    def _run(self):
        try:
            while not self.stop_event.is_set():
                try:item=next(self.source)
                except StopIteration:self._put(('end',None));return
                prepared=self.prepare(item);self.stats['prepared']+=1
                if not self._put(('item',prepared)):return
        except BaseException as exc:self._put(('error',exc))
    def __iter__(self):return self
    def __next__(self):
        if self.closed:raise StopIteration
        if not self.depth:
            try:item=self.prepare(next(self.source));self.stats['prepared']+=1;self.stats['yielded']+=1;return item
            except StopIteration:self.close();raise
        if self.thread is None:self.thread=threading.Thread(target=self._run,name='arc-data-prefetch',daemon=True);self.thread.start()
        kind,value=self.queue.get()
        if kind=='item':self.stats['yielded']+=1;return value
        self.close()
        if kind=='error':raise value
        raise StopIteration
    def close(self):
        if self.closed:return
        self.closed=True;self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
            if self.thread.is_alive():raise RuntimeError('Prefetch producer did not stop; its input must be finite and bounded')
        if hasattr(self.source,'close'):self.source.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

def get_batch(records):return reference_batch(records)
