"""Immutable, redundant checkpoint bundles with a single SQLite commit point.

Only locally created trusted torch checkpoints are read. Callers hold their run
lock. No shard is deleted here and no champion path is ever written.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import uuid

from .storage import atomic_json,canonical,digest,test_hook

VENDOR=Path(__file__).resolve().parents[1]/'vendor'
if str(VENDOR) not in sys.path:sys.path.insert(0,str(VENDOR))
from zero import checkpoint as native_checkpoint

FORMAT='chromatic-shard-bundle-v1'
OWNED=re.compile(r'^bundle-([0-9a-f]{32})\.(?:pt|mirror\.pt|zero|manifest\.json)(?:\.tmp)?$')


def _root(directory):
    root=Path(directory).resolve();root.mkdir(parents=True,exist_ok=True)
    child=root/'checkpoints';child.mkdir(exist_ok=True)
    if child.resolve()!=child:raise RuntimeError('checkpoint storage cannot be a symlink')
    return child


def _owned_path(path,root):
    path=Path(path)
    if not path.is_absolute():path=root/path
    resolved=path.resolve()
    if not resolved.is_relative_to(root) or resolved.parent!=root or not OWNED.fullmatch(path.name):
        raise ValueError('checkpoint manifest points outside owned bundle files')
    return resolved


def _owner(ledger,directory):
    """Prevent one data ledger from cleaning another run's checkpoint folder."""
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    token=ledger.get('durable_owner_token')
    if token is None:
        token=uuid.uuid4().hex
        with ledger.transaction():ledger.put('durable_owner_token',token)
    path=directory/'checkpoint-owner.json'
    expected={'format':FORMAT,'ledger_token':token}
    # A separate OS lock also serializes binding from different data ledgers.
    # The owner marker itself is atomically replaced, so termination before its
    # initial commit leaves an ignorable .tmp rather than a partial JSON owner.
    lock=(directory/'checkpoint-owner.lock').open('a+b')
    if lock.tell()==0:lock.write(b'0');lock.flush()
    lock.seek(0)
    acquired=False
    try:
        try:
            if os.name=='nt':
                import msvcrt;msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl;fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            acquired=True
        except OSError as exc:raise RuntimeError('CheckpointDirectory binding is locked by another process') from exc
        if path.exists():
            if json.loads(path.read_text(encoding='utf-8'))!=expected:
                raise RuntimeError('CheckpointDirectory belongs to a different data ledger')
        else:atomic_json(path,expected)
    finally:
        if acquired:
            lock.seek(0)
            if os.name=='nt':
                import msvcrt;msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl;fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
        lock.close()


def _finite(value):
    import numpy as np
    import torch
    if isinstance(value,torch.Tensor):
        if not bool(torch.isfinite(value).all()):raise ValueError('nonfinite checkpoint tensor')
    elif isinstance(value,np.ndarray):
        if not bool(np.isfinite(value).all()):raise ValueError('nonfinite checkpoint array')
    elif isinstance(value,float):
        if not math.isfinite(value):raise ValueError('nonfinite checkpoint metadata')
    elif isinstance(value,dict):
        for item in value.values():_finite(item)
    elif isinstance(value,(tuple,list)):
        for item in value:_finite(item)


def _progress(state):
    p=state.get('progress')
    if not isinstance(p,dict):raise ValueError('missing durable shard progress')
    keys={'completed','active','epoch','offset','permutation','positions_processed'}
    if not keys<=p.keys():raise ValueError('incomplete durable shard progress')
    completed=p['completed']
    if not isinstance(completed,list) or any(type(i) is not int or i<0 for i in completed) or len(set(completed))!=len(completed):
        raise ValueError('invalid completed shard IDs')
    if p['active'] is not None and (type(p['active']) is not int or p['active']<0 or p['active'] in completed):
        raise ValueError('invalid active shard ID')
    for key in ('epoch','offset','positions_processed'):
        if type(p[key]) is not int or p[key]<0:raise ValueError('invalid progress '+key)
    if not isinstance(p['permutation'],list) or any(not isinstance(i,str) for i in p['permutation']) or p['offset']>len(p['permutation']):
        raise ValueError('invalid sampler permutation/cursor')
    if type(state.get('step')) is not int or state['step']<0:raise ValueError('invalid global step')
    return p


def _export_shape(state):
    c=state['config'];width=int(c['width']);blocks=int(c['blocks'])
    if not 1<=width<=1024 or not 0<=blocks<=16:raise ValueError('invalid network configuration')
    layers=[('input',1280,width)]+[(f'residual.{i}',width,width) for i in range(blocks)]+[('policy',width,4672),('wdl',width,3)]
    expected=24+sum((a*b+b)*4 for _,a,b in layers)
    return width,blocks,layers,expected


def _export_digest(state):
    """Validate ZERO0001 directly from tensor bytes, without initializing an NN."""
    width,blocks,layers,size=_export_shape(state)
    h=hashlib.sha256();h.update(b'ZERO0001');h.update(struct.pack('<4I',1280,width,blocks,4672))
    for name,inputs,outputs in layers:
        for suffix,shape in [('weight',(outputs,inputs)),('bias',(outputs,))]:
            tensor=state['model'][name+'.'+suffix]
            if tuple(tensor.shape)!=shape:raise ValueError('checkpoint tensor shape mismatch: '+name+'.'+suffix)
            h.update(tensor.detach().cpu().numpy().astype('<f4').tobytes())
    return h.hexdigest(),size


def _estimated_checkpoint_bytes(value):
    import numpy as np
    import torch
    if isinstance(value,torch.Tensor):return value.numel()*value.element_size()+1024
    if isinstance(value,np.ndarray):return value.nbytes+1024
    if isinstance(value,str):return len(value.encode('utf-8'))+128
    if isinstance(value,dict):return 512+sum(_estimated_checkpoint_bytes(k)+_estimated_checkpoint_bytes(v) for k,v in value.items())
    if isinstance(value,(list,tuple)):return 256+sum(_estimated_checkpoint_bytes(v) for v in value)
    return 128


def _next_seq(ledger):
    row=ledger.db.execute("SELECT seq FROM sqlite_sequence WHERE name='checkpoints'").fetchone()
    maximum=ledger.db.execute('SELECT coalesce(max(seq),0) FROM checkpoints').fetchone()[0]
    return max(int(row[0]) if row else 0,int(maximum))+1


def _file(path):return {'path':str(path),'sha256':digest(path),'bytes':path.stat().st_size}


def read_manifest(path):
    path=Path(path).resolve();data=json.loads(path.read_text(encoding='utf-8'))
    if data.get('format')!=FORMAT or not re.fullmatch('[0-9a-f]{32}',str(data.get('id',''))):raise ValueError('invalid checkpoint bundle manifest')
    expected='bundle-'+data['id']+'.manifest.json'
    if path.name!=expected:raise ValueError('manifest ID/name mismatch')
    if type(data.get('commit_seq')) is not int or data['commit_seq']<1:raise ValueError('invalid commit sequence')
    return data


def verify_bundle(manifest_path):
    """Return trusted state and whether its redundant mirror was needed."""
    manifest_path=Path(manifest_path).resolve();root=manifest_path.parent
    manifest=read_manifest(manifest_path);base='bundle-'+manifest['id'];files=manifest['files']
    expected={'checkpoint':base+'.pt','mirror':base+'.mirror.pt','model':base+'.zero'}
    for key,name in expected.items():
        spec=files[key];path=_owned_path(spec['path'],root)
        if path.name!=name:raise ValueError('bundle file ID/name mismatch')
    failures=[];state=None;mirror=False
    for key in ('checkpoint','mirror'):
        spec=files[key];path=_owned_path(spec['path'],root)
        try:
            if path.stat().st_size!=spec['bytes'] or digest(path)!=spec['sha256']:raise ValueError('checkpoint size/hash mismatch')
            candidate=native_checkpoint._read(path);_finite(candidate);progress=_progress(candidate)
            if candidate.get('durable_commit_seq')!=manifest['commit_seq'] or candidate['step']!=manifest['step']:
                raise ValueError('manifest/state commit mismatch')
            if progress['completed']!=manifest['completed'] or progress['active']!=manifest['active']:
                raise ValueError('manifest/state progress mismatch')
            state=candidate;mirror=key=='mirror';break
        except Exception as exc:failures.append(f'{key}: {type(exc).__name__}: {exc}')
    if state is None:raise RuntimeError('no readable redundant checkpoint: '+'; '.join(failures))
    spec=files['model'];path=_owned_path(spec['path'],root)
    expected_hash,size=_export_digest(state)
    if path.stat().st_size!=size or spec['bytes']!=size or digest(path)!=spec['sha256'] or spec['sha256']!=expected_hash:
        raise ValueError('native export does not match checkpoint weights')
    with path.open('rb') as stream:
        width,blocks,_,_=_export_shape(state)
        if stream.read(24)!=b'ZERO0001'+struct.pack('<4I',1280,width,blocks,4672):raise ValueError('invalid native export header')
    return state,mirror


def save_bundle(ledger,checkpoint_dir,model,state,guard,event,completed_shard=None):
    import torch
    _owner(ledger,checkpoint_dir);root=_root(checkpoint_dir);ledger._durable_guard=guard
    progress=_progress(state);_finite(state);_finite(event)
    done={r[0] for r in ledger.db.execute("SELECT id FROM shards WHERE status='done'")}
    expected=done|({completed_shard} if completed_shard is not None else set())
    if set(progress['completed'])!=expected:raise ValueError('checkpoint completion set disagrees with committed shard ledger')
    if completed_shard is not None:
        row=ledger.db.execute('SELECT status FROM shards WHERE id=?',(completed_shard,)).fetchone()
        if row is None or completed_shard in done or progress['active'] is not None:
            raise ValueError('invalid/duplicate shard completion commit')
    if progress['active'] is not None and ledger.db.execute('SELECT 1 FROM shards WHERE id=?',(progress['active'],)).fetchone() is None:
        raise ValueError('active shard is absent from ledger')
    model_state=model.state_dict()
    if set(model_state)!=set(state['model']) or any(not torch.equal(v.detach().cpu(),state['model'][k].detach().cpu()) for k,v in model_state.items()):
        raise ValueError('checkpoint state does not match model being exported')
    cleanup_bundles(ledger,checkpoint_dir)
    _,_,_,export_bytes=_export_shape(state)
    estimate=_estimated_checkpoint_bytes(state)+256*1024
    guard.check(extra_disk=estimate*2+export_bytes+65536,force=True)
    identifier=uuid.uuid4().hex;base=root/('bundle-'+identifier);seq=_next_seq(ledger)
    primary=Path(str(base)+'.pt');mirror=Path(str(base)+'.mirror.pt');export=Path(str(base)+'.zero');manifest_path=Path(str(base)+'.manifest.json')
    payload=dict(state,durable_commit_seq=seq)
    native_checkpoint.save(primary,payload)
    actual=native_checkpoint._read(primary);_finite(actual);_progress(actual)
    guard.check(extra_disk=primary.stat().st_size+export_bytes+65536,force=True)
    mirror_temp=Path(str(mirror)+'.tmp')
    with primary.open('rb') as source,mirror_temp.open('wb') as target:
        shutil.copyfileobj(source,target,1024*1024);target.flush();os.fsync(target.fileno())
    os.replace(mirror_temp,mirror)
    if digest(primary)!=digest(mirror):raise RuntimeError('checkpoint mirror verification failed')
    native_checkpoint._read(mirror)
    guard.check(extra_disk=export_bytes+65536,force=True);model.export(export)
    summary={k:actual['progress'][k] for k in ('completed','active','epoch','offset','positions_processed')}
    summary['permutation_sha256']=hashlib.sha256(canonical(actual['progress']['permutation']).encode()).hexdigest()
    manifest={'format':FORMAT,'id':identifier,'commit_seq':seq,'step':actual['step'],
              'active':progress['active'],'completed':list(progress['completed']),'config':actual['config'],'progress':summary,
              'files':{'checkpoint':_file(primary),'mirror':_file(mirror),'model':_file(export)}}
    guard.check(extra_disk=len(canonical(manifest).encode())*2+1024,force=True);atomic_json(manifest_path,manifest)
    verified,used_mirror=verify_bundle(manifest_path)
    if used_mirror:raise RuntimeError('new primary checkpoint did not verify')
    shard=completed_shard if completed_shard is not None else progress['active']
    test_hook('checkpoint_written',checkpoint_dir,actual['step'],shard if shard is not None else 0)
    with ledger.transaction():
        if _next_seq(ledger)!=seq:raise RuntimeError('concurrent checkpoint writer detected')
        ledger.db.execute('INSERT INTO checkpoints(seq,manifest,step,completed,active) VALUES (?,?,?,?,?)',
                          (seq,str(manifest_path),actual['step'],len(progress['completed']),progress['active']))
        ledger.put('current_manifest',str(manifest_path));ledger.put('current_checkpoint_seq',seq)
        ledger.put('current_global_step',actual['step']);ledger.put('recovered_seq',None)
        if completed_shard is not None:
            ledger.db.execute("UPDATE shards SET status='done',checkpoint=? WHERE id=?",(str(manifest_path),completed_shard))
        ledger.event(dict(event or {},commit_seq=seq,checkpoint=str(manifest_path)))
    test_hook('ledger_committed',checkpoint_dir,actual['step'],shard if shard is not None else 0)
    if completed_shard is not None:test_hook('shard_completed',checkpoint_dir,actual['step'],completed_shard)
    guard.check(force=True)
    return verified,manifest_path


def _required_shards_exist(ledger,state):
    p=_progress(state);completed=set(p['completed']);needed=[]
    for identity,path,expected,status in ledger.db.execute('SELECT id,path,sha,status FROM shards'):
        if identity not in completed and (status=='done' or identity==p['active']):needed.append((identity,path,expected))
    for identity,path,expected in needed:
        file=Path(path)
        if not file.is_absolute():raise ValueError('recovery requires absolute shard paths')
        if not file.is_file() or digest(file)!=expected:
            raise RuntimeError(f'checkpoint rollback requires retained shard {identity}: {file}')


def load_bundle(ledger,checkpoint_dir):
    _owner(ledger,checkpoint_dir);root=_root(checkpoint_dir)
    rows=ledger.db.execute('SELECT seq,manifest FROM checkpoints ORDER BY seq DESC').fetchall()
    if not rows:
        if ledger.get('current_manifest') is not None:raise RuntimeError('checkpoint head has no committed ledger record')
        return None,None,False
    current=ledger.get('current_manifest');head_seq=ledger.get('current_checkpoint_seq')
    if head_seq is None:head_seq=rows[0][0]
    rows=[r for r in rows if r[0]<=head_seq]
    errors=[]
    for seq,path in rows:
        try:
            manifest_path=_owned_path(path,root)
            if not manifest_path.name.endswith('.manifest.json'):raise ValueError('checkpoint ledger is not a manifest')
            state,mirror=verify_bundle(manifest_path)
            if state['durable_commit_seq']!=seq:raise ValueError('checkpoint ledger commit mismatch')
            _required_shards_exist(ledger,state)
        except Exception as exc:errors.append(f'{seq}: {type(exc).__name__}: {exc}');continue
        recovered=mirror or seq!=head_seq or str(manifest_path)!=current
        if recovered:
            completed=set(state['progress']['completed']);active=state['progress']['active']
            with ledger.transaction():
                ledger.put('current_manifest',str(manifest_path));ledger.put('current_checkpoint_seq',seq)
                ledger.put('current_global_step',state['step']);ledger.put('recovered_seq',seq)
                ledger.db.execute('DELETE FROM checkpoints WHERE seq>?',(seq,))
                for event_id,payload in ledger.db.execute('SELECT id,payload FROM events').fetchall():
                    event_seq=json.loads(payload).get('commit_seq',0)
                    if event_seq>seq:ledger.db.execute('DELETE FROM events WHERE id=?',(event_id,))
                for (identity,) in ledger.db.execute('SELECT id FROM shards').fetchall():
                    status='done' if identity in completed else 'prepared'
                    ledger.db.execute('UPDATE shards SET status=?,checkpoint=? WHERE id=?',
                                      (status,str(manifest_path) if identity in completed else None,identity))
            state=dict(state,recovered_seq=seq,recovery={'mirror_used':mirror,'discarded_newer_commits':seq!=head_seq,'errors':errors})
        return state,manifest_path,recovered
    raise RuntimeError('no recoverable committed checkpoint; shards retained: '+'; '.join(errors))


def cleanup_bundles(ledger,checkpoint_dir,guard=None):
    """Keep latest two commits and latest two boundaries; unlink owned files only."""
    _owner(ledger,checkpoint_dir);root=_root(checkpoint_dir)
    rows=ledger.db.execute('SELECT seq,manifest,active FROM checkpoints ORDER BY seq DESC').fetchall()
    keep={r[0] for r in rows[:2]}
    keep.update(r[0] for r in [x for x in rows if x[2] is None][:2])
    head=ledger.get('current_checkpoint_seq')
    if head is not None:keep.add(head)
    remove=[r for r in rows if r[0] not in keep]
    if remove:
        with ledger.transaction():
            ledger.db.executemany('DELETE FROM checkpoints WHERE seq=?',[(r[0],) for r in remove])
    referenced=set()
    for _,path,_ in rows:
        try:
            manifest_path=_owned_path(path,root)
            if any(r[1]==path and r[0] in keep for r in rows):
                manifest=read_manifest(manifest_path);referenced.add(manifest_path)
                for spec in manifest['files'].values():referenced.add(_owned_path(spec['path'],root))
        except Exception:
            # A damaged retained manifest must not cause its readable peers to
            # disappear. Its UUID still reserves every member of that bundle.
            match=OWNED.fullmatch(Path(path).name)
            if match and any(r[1]==path and r[0] in keep for r in rows):
                base='bundle-'+match.group(1)
                referenced.update(root/(base+suffix) for suffix in ('.pt','.mirror.pt','.zero','.manifest.json'))
    deleted=[]
    for file in root.iterdir():
        if not file.is_file() or not OWNED.fullmatch(file.name):continue
        resolved=_owned_path(file,root)
        if resolved not in referenced:
            # Resolved target is checked as an owned direct child immediately
            # before this nonrecursive delete. No shard directory is traversed.
            resolved.unlink();deleted.append(str(resolved))
    guard=guard or getattr(ledger,'_durable_guard',None)
    if guard is not None:guard.check(force=True)
    return {'kept_sequences':sorted(keep),'deleted_files':deleted}
