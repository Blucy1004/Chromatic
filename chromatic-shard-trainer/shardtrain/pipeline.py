from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import itertools

from .storage import Ledger,ResourceGuard,atomic_json,canonical,digest,test_hook

class RunLock:
    def __init__(self,path):self.path=Path(path);self.file=None
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True);self.file=self.path.open('a+b')
        if self.file.tell()==0:self.file.write(b'0');self.file.flush()
        self.file.seek(0)
        try:
            if os.name=='nt':
                import msvcrt;msvcrt.locking(self.file.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl;fcntl.flock(self.file.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:self.file.close();raise RuntimeError('Another trainer holds this run lock') from exc
        return self
    def __exit__(self,*args):
        self.file.seek(0)
        if os.name=='nt':
            import msvcrt;msvcrt.locking(self.file.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl;fcntl.flock(self.file.fileno(),fcntl.LOCK_UN)
        self.file.close()

def replay_update(previous,records,capacity,seed):
    # Deterministic, bounded phase-stratified representatives, root retained.
    pools={p:{} for p in ['opening','middlegame','endgame']};per=max(1,capacity//3)
    root=None
    for r in itertools.chain(previous,records):
        if r['split']!='train':raise ValueError('holdout must never enter replay')
        if r.get('history_known') and not r.get('moves'):root=r
        phase=r['phase'];pool=pools[phase];pool[r['identity']]=r
        if len(pool)>per+1:
            worst=max(pool,key=lambda k:hashlib.sha256(f'{seed}:{k}'.encode()).digest());del pool[worst]
    selected=[]
    for phase,pool in pools.items():
        chosen=sorted(pool.values(),key=lambda r:hashlib.sha256(f"{seed}:{r['identity']}".encode()).digest())[:per]
        selected.extend(chosen)
    selected=sorted({r['identity']:r for r in selected}.values(),key=lambda r:r['identity'])[:capacity]
    if root is not None and root['identity'] not in {r['identity'] for r in selected}:selected=[root]+selected[:capacity-1]
    return selected

class Trainer:
    def __init__(self,config,source,calibration,anchors,excluded_families):
        self.config=config;self.data=Path(config['data_directory']);self.checkpoints=Path(config['checkpoint_directory'])
        self.guard=ResourceGuard([self.data,self.checkpoints],config['max_disk_gb'],config['max_memory_gb'])
        self.data.mkdir(parents=True,exist_ok=True);self.checkpoints.mkdir(parents=True,exist_ok=True)
        self.ledger=Ledger(self.data/'ledger.sqlite3');self.started=time.perf_counter();self.session_elapsed=0
        self.source=source;self.calibration=calibration;self.anchors=anchors;self.excluded=excluded_families
        self.state=None;self.manifest=None;self.shards=None
    def setup(self):
        from .model_adapter import torch,checkpoint,PolicyWDL,ModelConfig
        from .durable import load_bundle
        from .shards import ShardManager
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        random.seed(self.config['seed'])
        import numpy as np
        np.random.seed(self.config['seed']);torch.manual_seed(self.config['seed'])
        settings={k:v for k,v in self.config.items() if k not in ['resume','dry_run','max_disk_gb','max_memory_gb','keep_shards','max_download_mb','max_decompressed_mb']}
        signature=hashlib.sha256(canonical(settings).encode()).hexdigest()
        previous=self.ledger.get('settings_signature')
        if previous:
            if not self.config['resume']:raise ValueError('Run already exists; use -Resume or new data/checkpoint directories')
            if signature!=previous:raise ValueError('Resume settings/source differ; original fixed split and training configuration must be retained')
            self.state,self.manifest,recovered=load_bundle(self.ledger,self.checkpoints)
            self.recovered=recovered
        else:
            if self.config['resume']:raise ValueError('No existing run to resume')
            self.recovered=False
        if self.state is None:
            # A crash during initial streaming can precede the first model
            # commit. Reuse staged data, but initialize the never-trained model.
            initial,actual=checkpoint.load(self.config['initial_checkpoint'])
            self.state={'trainer_schema':1,'format':2,'config':initial['config'],'model':initial['model'],
                'optimizer':None,'scheduler':None,'step':0,'generator':None,'dataset':[{'source':self.source.location}],
                'settings':settings,'settings_signature':signature,'seed':self.config['seed'],'replay':[],
                'progress':{'completed':[],'active':None,'epoch':0,'offset':0,'permutation':[],'positions_processed':0},
                'initial_model_sha256':digest(actual),'elapsed_seconds':0.}
            with self.ledger.transaction():self.ledger.put('settings_signature',signature);self.ledger.put('settings',settings)
        self.model=PolicyWDL(ModelConfig(**self.state['config']));self.model.load_state_dict(self.state['model'])
        self.base_elapsed=self.state.get('elapsed_seconds',0.)
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=self.config['learning_rate'],weight_decay=1e-4)
        spec=self.config['scheduler_spec']
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer,lr_lambda=lambda step:max(spec['minimum_multiplier'],spec['decay']**step))
        self.generator=torch.Generator().manual_seed(self.config['seed'])
        if self.state['optimizer'] is not None:
            self.optimizer.load_state_dict(self.state['optimizer']);self.scheduler.load_state_dict(self.state['scheduler'])
            self.generator.set_state(self.state['generator']);checkpoint.restore_rng(self.state['rng'])
        self.shards=ShardManager(self.ledger,self.data,self.guard,self.config,self.calibration,self.source,self.anchors,excluded_families=self.excluded)
        if self.state['progress']['active'] is not None:
            row=self.ledger.db.execute('SELECT id,path,sha,rows FROM shards WHERE id=?',(self.state['progress']['active'],)).fetchone()
            if row is None:raise RuntimeError('Active recovery shard is missing from ledger')
            self.shards._prepared(row)
        self.validation,self.test=self.shards.prepare_fixed()
        self.holdout_digest=hashlib.sha256(canonical({'validation':[r['identity'] for r in self.validation],'test':[r['identity'] for r in self.test]}).encode()).hexdigest()
        if self.state.get('holdout_digest') and self.state['holdout_digest']!=self.holdout_digest:raise RuntimeError('Fixed holdout changed')
        self.state['holdout_digest']=self.holdout_digest
        with self.ledger.transaction():self.ledger.put('runtime_config',self.config)
        if self.manifest is None:self.commit({'kind':'initialized'})
        else:
            atomic_json(self.checkpoints/'status.json',{'status':'resumed',**self.event({'kind':'resumed'}),'latest_checkpoint':str(self.manifest),'recovered':self.recovered})
            self.ledger.export_logs(self.checkpoints/'logs')
    def event(self,extra):
        rss,disk=self.guard.check(force=True);p=self.state['progress'];elapsed=self.base_elapsed+time.perf_counter()-self.started
        active=0.
        if p['active'] is not None:
            row=self.ledger.db.execute('SELECT rows FROM shards WHERE id=?',(p['active'],)).fetchone()
            if row:active=(p['epoch']*row[0]+p['offset'])/self.config['epochs']
        position_equivalent=p['positions_processed']+active;speed=position_equivalent/max(elapsed,.001)
        return {'kind':'checkpoint','global_step':self.state['step'],'shard':p['active'],
                'completed_shards':len(p['completed']),'positions_processed':p['positions_processed'],
                'progress_percent':min(100.,100*position_equivalent/self.config['positions']),
                'positions_per_second':speed,'eta_seconds':(self.config['positions']-position_equivalent)/speed if speed else None,
                'memory_rss_bytes':rss,'disk_bytes':disk,'elapsed_seconds':elapsed,**extra}
    def commit(self,extra,completed_shard=None):
        from .model_adapter import checkpoint
        from .durable import save_bundle,cleanup_bundles
        self.state.update(model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),scheduler=self.scheduler.state_dict(),
                          generator=self.generator.get_state(),rng=checkpoint.rng_state(),source_cursor=self.ledger.get('source_cursor'))
        event=self.event(extra)
        self.state['elapsed_seconds']=event['elapsed_seconds']
        self.state,self.manifest=save_bundle(self.ledger,self.checkpoints,self.model,self.state,self.guard,event,completed_shard=completed_shard)
        event['latest_checkpoint']=str(self.manifest)
        atomic_json(self.checkpoints/'status.json',{'status':'running',**event,'recovered':self.recovered})
        self.ledger.export_logs(self.checkpoints/'logs');cleanup_bundles(self.ledger,self.checkpoints)
        eta='?' if event['eta_seconds'] is None else f"{event['eta_seconds']:.1f}s"
        print(f"{event['progress_percent']:6.2f}% shard={event['shard']} positions={event['positions_processed']}/{self.config['positions']} step={event['global_step']} policy={event.get('policy_loss')} wdl={event.get('wdl_loss')} validation={event.get('validation_loss')} {event['positions_per_second']:.1f}pos/s ETA={eta} RSS={event['memory_rss_bytes']/1024**2:.1f}MiB disk={event['disk_bytes']/1024**2:.1f}MiB checkpoint={self.manifest}",flush=True)
    def train(self):
        from .model_adapter import torch,weighted_losses,measurements
        self.setup();p=self.state['progress']
        self.shards.cleanup(p['completed'],self.config['keep_shards'])
        while p['positions_processed']<self.config['positions']:
            self.guard.check(force=True)
            if p['active'] is None:
                shard=self.shards.next_shard(self.config['positions']-p['positions_processed'])
                if shard is None:raise RuntimeError('Source ended before requested balanced training positions')
                p.update(active=shard['id'],epoch=0,offset=0,permutation=[])
                self.commit({'kind':'shard_started'});p=self.state['progress']
            shard_id=p['active'];row=self.ledger.db.execute('SELECT path,sha,rows FROM shards WHERE id=?',(shard_id,)).fetchone()
            if row is None or digest(row[0])!=row[1]:raise RuntimeError('Active shard missing or corrupt; retained for inspection')
            identities=[r['identity'] for _,r in self.ledger.rows(shard=shard_id)]
            if len(identities)!=row[2]:raise RuntimeError('Active shard row count differs from durable ledger')
            while p['epoch']<self.config['epochs']:
                if not p['permutation']:
                    p['permutation']=[identities[i] for i in torch.randperm(len(identities),generator=self.generator).tolist()];p['offset']=0
                while p['offset']<len(p['permutation']):
                    self.guard.check()
                    replay=self.state['replay'];extra=max(0,self.config['batch_size']//5) if replay else 0
                    current_batch=max(1,self.config['batch_size']-extra)
                    chosen=p['permutation'][p['offset']:p['offset']+current_batch]
                    records=self.ledger.load_ids(chosen)
                    replay_count=min(extra,self.config['batch_size']-len(records))
                    if replay_count:
                        for index in torch.randint(len(replay),(replay_count,),generator=self.generator).tolist():records.append(replay[index])
                    self.model.train();self.optimizer.zero_grad(set_to_none=True)
                    loss,pl,vl,pw,vw,_=weighted_losses(self.model,records)
                    if not torch.isfinite(loss):raise RuntimeError('NaN/Inf loss; uncommitted shard retained')
                    loss.backward();grad=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.,error_if_nonfinite=True)
                    self.optimizer.step();self.scheduler.step()
                    if any(not torch.isfinite(v).all() for v in self.model.parameters()):raise RuntimeError('NaN/Inf weights; last durable checkpoint retained')
                    for values in self.optimizer.state.values():
                        for value in values.values():
                            if isinstance(value,torch.Tensor) and not torch.isfinite(value).all():raise RuntimeError('NaN/Inf optimizer state; unfinished shard retained')
                    p['offset']+=len(chosen);self.state['step']+=1
                    metrics={'policy_loss':float(((pl*pw).sum()/pw.sum()).detach()) if pw.sum()>0 else None,
                             'wdl_loss':float(((vl*vw).sum()/vw.sum()).detach()) if vw.sum()>0 else None,
                             'gradient_norm':float(grad),'learning_rate':self.optimizer.param_groups[0]['lr']}
                    if self.state['step']%self.config['checkpoint_every']==0:
                        self.commit(metrics);p=self.state['progress']
                p['epoch']+=1;p['offset']=0;p['permutation']=[]
            # Evaluate before recording a completed shard or making it eligible
            # for deletion. Failed validation/export never consumes that shard.
            validation=measurements(self.model,self.validation)
            if any(v is not None and not math.isfinite(v) for k,v in validation.items() if k.endswith('loss')):raise RuntimeError('Nonfinite independent validation')
            active_records=(r for _,r in self.ledger.rows(shard=shard_id))
            self.state['replay']=replay_update(self.state['replay'],active_records,self.config['replay_size'],self.config['seed'])
            p['completed'].append(shard_id);p['positions_processed']+=len(identities)
            p.update(active=None,epoch=0,offset=0,permutation=[])
            self.commit({'kind':'shard_completed','validation_loss':validation['policy_loss'],'validation_wdl_loss':validation['wdl_loss']},completed_shard=shard_id)
            p=self.state['progress']
            self.shards.cleanup(p['completed'],self.config['keep_shards'])
        final={'validation':measurements(self.model,self.validation),'test':measurements(self.model,self.test),
               'holdout_digest':self.holdout_digest,'positions':p['positions_processed'],'global_step':self.state['step'],
               'completed_shards':p['completed'],'candidate_manifest':str(self.manifest),'champion_overwritten':False,
               'calibration_status':self.calibration.artifact['status'] if self.calibration else 'unavailable',
               'quality':self.ledger.get('quality_stats',{}),'source_cursor':self.ledger.get('source_cursor'),
               'promotion':{'eligible':False,'reason':'candidate only; independent outcome/tactical/match gates required; champion never automatically overwritten'}}
        atomic_json(self.checkpoints/'training-report.json',final)
        path=self.checkpoints/'status.json'
        status=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        status.update(status='completed',progress_percent=100.,**{k:final[k] for k in ['global_step','positions']})
        atomic_json(self.checkpoints/'status.json',status)
        return final
    def close(self):
        if self.shards is not None:self.shards.close()
        self.ledger.close()
