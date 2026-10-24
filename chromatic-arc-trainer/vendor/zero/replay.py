"""Bounded sparse replay; raw game shards are immutable durable sources."""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import base64
import zlib
import numpy as np
import torch
from .checkpoint import digest
from .data import atomic_json
from .encoding import INPUTS,ACTIONS,action,features,restore_board,outcome


@dataclass
class Prepared:
    x:np.ndarray
    policy:list
    legal:list
    wdl:np.ndarray
    group:str
    position:str
    identity:str
    recent:bool
    reference:dict


def prepare(record,reference=None,recent=True):
    if record.get('schema') not in (1,2) or record.get('source') not in ('bootstrap','selfplay'):
        raise ValueError('unsupported record schema/source')
    if 'wdl' not in record:raise ValueError('unfinished record has no outcome')
    board=restore_board(record)
    if not board.is_valid() or outcome(board):raise ValueError('invalid or terminal training position')
    x=features(board);legal=sorted(action(m) for m in board.legal_moves)
    seen=set();total=0
    for index,probability in record['policy']:
        if type(index) is not int or index not in legal or index in seen or not np.isfinite(probability) or probability<0:
            raise ValueError('invalid or duplicate policy action')
        seen.add(index);total+=probability
    if not record['policy'] or abs(total-1)>1e-5:raise ValueError('policy is not normalized')
    wdl=np.asarray(record['wdl'],dtype=np.float32)
    if wdl.shape!=(3,) or not np.isfinite(wdl).all() or (wdl<0).any() or abs(float(wdl.sum())-1)>1e-5:
        raise ValueError('invalid WDL target')
    if record['schema']==2:
        if record['legal_actions']!=legal:raise ValueError('recorded legal mask mismatch')
        payload=base64.b64decode(record['encoded_position'],validate=True)
        decoder=zlib.decompressobj();raw=decoder.decompress(payload,INPUTS*4+1)
        if len(raw)!=INPUTS*4 or not decoder.eof or decoder.unused_data:
            raise ValueError('invalid encoded position payload')
        if not np.array_equal(np.frombuffer(raw,dtype='<f4'),x):raise ValueError('encoded position mismatch')
        if record['selected_action'] not in legal:raise ValueError('selected illegal action')
        if record.get('termination') in (None,'TRUNCATED','IN_PROGRESS'):raise ValueError('missing actual termination')
    position=hashlib.sha256(x.tobytes()).hexdigest()
    identity=hashlib.sha256((position+json.dumps(record['policy'],sort_keys=True)+str(wdl.tolist())).encode()).hexdigest()
    if 'game_id' in record:group=record['game_id']
    elif record['source']=='bootstrap':group='teacher:'+str(record.get('index',position))
    else:group=str(record.get('model','fixture'))+':'+str(record.get('game',record['root_fen']))
    return Prepared(x,record['policy'],legal,wdl,str(group),position,identity,recent,reference or {})


class Replay:
    def __init__(self,paths,capacity=10000,recent_fraction=.75,seed=1):
        if capacity<1 or not 0<=recent_fraction<=1:raise ValueError('invalid replay capacity/mix')
        self.sources=[{'sha256':digest(p),'bytes':Path(p).stat().st_size} for p in paths]
        parents=list(dict.fromkeys(str(Path(p).parent) for p in paths))
        recent_parent=parents[-1] if parents else ''
        old_capacity=round(capacity*(1-recent_fraction)) if len(parents)>1 else 0
        quotas=[old_capacity,capacity-old_capacity];pools=[[],[]];counts=[0,0]
        rng=random.Random(seed);seen=OrderedDict();duplicate_count=0;total=0
        for path in paths:
            recent=str(Path(path).parent)==recent_parent
            pool_index=int(recent)
            with open(path,encoding='utf-8') as stream:
                for line_number,line in enumerate(stream,1):
                    total+=1
                    try:row=prepare(json.loads(line),{'path':str(Path(path).resolve()),'line':line_number},recent)
                    except Exception as exc:raise ValueError(f'{path}:{line_number}: {exc}') from exc
                    if row.identity in seen:duplicate_count+=1;continue
                    seen[row.identity]=None
                    if len(seen)>capacity*8:seen.popitem(last=False)
                    counts[pool_index]+=1
                    pool=pools[pool_index];quota=quotas[pool_index]
                    if len(pool)<quota:pool.append(row)
                    elif quota:
                        index=rng.randrange(counts[pool_index])
                        if index<quota:pool[index]=row
        # If one stratum is short, its unused space can be filled from the other source
        # on a subsequent build with a different fraction; never exceed capacity here.
        self.rows=pools[0]+pools[1]
        if not self.rows:raise ValueError('empty replay dataset')
        self.stats={'capacity':capacity,'selected':len(self.rows),'records_scanned':total,
                    'recent_selected':len(pools[1]),'older_selected':len(pools[0]),
                    'duplicates_in_window':duplicate_count,'dedup_window_capacity':capacity*8,
                    'source_recent_records':counts[1],'source_older_records':counts[0]}

    def split(self,fraction,seed):
        if not 0<=fraction<.5:raise ValueError('validation fraction must be [0,.5)')
        # Union games sharing the same encoded position to prevent position leakage.
        parent={row.group:row.group for row in self.rows}
        def find(x):
            while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
            return x
        positions={}
        for row in self.rows:
            if row.position in positions:parent[find(row.group)]=find(positions[row.position])
            else:positions[row.position]=row.group
        groups=sorted({find(row.group) for row in self.rows});random.Random(seed).shuffle(groups)
        holdout=set(groups[:max(1,int(len(groups)*fraction))]) if fraction and len(groups)>1 else set()
        train=[i for i,row in enumerate(self.rows) if find(row.group) not in holdout]
        valid=[i for i,row in enumerate(self.rows) if find(row.group) in holdout]
        self.stats.update(validation_rows=len(valid),training_rows=len(train),independent_groups=len(groups),
                          validation_status='held_out_games_and_positions' if valid else 'unavailable_single_group_or_disabled')
        return train,valid

    def batch(self,indices,device='cpu'):
        rows=[self.rows[int(i)] for i in indices]
        x=np.stack([row.x for row in rows]);p=np.zeros((len(rows),ACTIONS),dtype=np.float32)
        mask=np.zeros((len(rows),ACTIONS),dtype=np.bool_)
        for i,row in enumerate(rows):
            for a,prob in row.policy:p[i,a]=prob
            mask[i,row.legal]=True
        values=(x,p,np.stack([row.wdl for row in rows]),mask)
        return tuple(torch.from_numpy(value).to(device) for value in values)

    def persist(self,path):
        atomic_json(path,{'format':1,'sources':self.sources,'stats':self.stats,
                          'selected':[dict(row.reference,identity=row.identity,position=row.position,
                                           group=row.group,recent=row.recent) for row in self.rows]})
