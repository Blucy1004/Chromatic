from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import itertools
import copy
from contextlib import nullcontext

from .storage import Ledger,ResourceGuard,atomic_json,canonical,digest,test_hook,training_settings

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
        self.runtime=None;self.cache=None;self.profile=None;self.sampler=None
    def span(self,name):
        return self.profile.span(name,sync=name in ['validation','checkpoint_save','host_to_device','device_batch_gather']) if self.profile is not None else nullcontext()
    def setup(self):
        from .model_adapter import torch,checkpoint,PolicyWDL,ModelConfig
        from .durable import load_bundle
        from .shards import ShardManager
        from .devices import DeviceRuntime,cpu_state,make_optimizer,optimizer_to_device,restore_xpu_rng
        from .profiling import Profile,install_profile,ResourceSampler
        from .fastdata import EncodedCache
        from .execution import Execution
        torch.use_deterministic_algorithms(True)
        self.runtime=DeviceRuntime(self.config.get('resolved_device',self.config.get('device','cpu')),
            precision=self.config.get('precision','fp32'),cpu_threads=self.config.get('cpu_threads',1),fallback=self.config.get('fallback','error'))
        if self.config.get('device')=='auto':
            self.runtime.decision.update(requested='auto',reason='automatic choice; see device-recommendation.json for measured scope')
        self.profile=Profile(enabled=self.config.get('profile',False),output_directory=self.checkpoints/'profile',synchronize=self.runtime.synchronize)
        install_profile(self.profile if self.config.get('profile',False) else None)
        self.sampler=ResourceSampler(interval=.2,device=self.runtime.device.type,output_directory=self.checkpoints/'resources').start()
        self.runtime.reset_peak_memory()
        random.seed(self.config['seed'])
        import numpy as np
        np.random.seed(self.config['seed']);torch.manual_seed(self.config['seed'])
        self.runtime.seed(self.config['seed'])
        settings=training_settings(self.config)
        signature=hashlib.sha256(canonical(settings).encode()).hexdigest()
        previous=self.ledger.get('settings_signature')
        legacy=self.ledger.get('legacy_import');legacy_migration=False
        if self.ledger.get('legacy_import_failed'):raise ValueError('Incomplete legacy copy; use a new empty import destination')
        if previous:
            if not self.config['resume']:raise ValueError('Run already exists; use -Resume or new data/checkpoint directories')
            if signature!=previous:
                if legacy is None:raise ValueError('Resume settings/source differ; original fixed split and training configuration must be retained')
                from .legacy_resume import authorize
                authorize(legacy,self.config,previous)
            with self.span('checkpoint_recovery'):self.state,self.manifest,recovered=load_bundle(self.ledger,self.checkpoints)
            self.recovered=recovered
            if legacy is not None and self.state is not None and self.state.get('settings_signature')!=signature:
                from .legacy_resume import authorize
                authorize(legacy,self.config,previous,self.state)
                self.state.update(settings=settings,settings_signature=signature,trainer_schema=2,
                    legacy_origin={'imported_step':legacy['imported_step'],'source_data_directory':legacy['source_data_directory'],
                                   'source_checkpoint_directory':legacy['source_checkpoint_directory']})
                with self.ledger.transaction():self.ledger.put('settings_signature',signature);self.ledger.put('settings',settings)
                legacy_migration=True
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
        # Deep-copy avoids consuming any extra model-initialization RNG. Export
        # and redundant checkpoint validation remain CPU-only.
        self.export_model=copy.deepcopy(self.model).cpu()
        self.model.to(self.runtime.device)
        self.base_elapsed=self.state.get('elapsed_seconds',0.)
        self.optimizer=make_optimizer(self.model,lr=self.config['learning_rate'],mode=self.config.get('optimizer','default'))
        spec=self.config['scheduler_spec']
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer,lr_lambda=lambda step:max(spec['minimum_multiplier'],spec['decay']**step))
        self.generator=torch.Generator().manual_seed(self.config['seed'])
        if self.state['optimizer'] is not None:
            self.optimizer.load_state_dict(self.state['optimizer']);self.scheduler.load_state_dict(self.state['scheduler'])
            self.generator.set_state(self.state['generator']);checkpoint.restore_rng(self.state['rng'])
            optimizer_to_device(self.optimizer,self.runtime.device)
            restore_xpu_rng(self.state.get('rng_xpu'),strict=self.runtime.device.type=='xpu')
        if self.state.get('effective_precision')=='fp32' and self.config.get('precision')=='bf16':
            self.runtime.fallback_fp32('restored durable mixed-precision fallback')
        self.execution=Execution(self.model,self.optimizer,self.scheduler,self.runtime,self.profile,self.config,start_step=self.state['step'])
        cache_mode=self.config.get('cache_mode','auto')
        self.cache=EncodedCache(self.data/'encoded-cache',mode='memory' if cache_mode=='device' else cache_mode,
            memory_budget_bytes=min(int(self.config['max_memory_gb']*1024**3*.15),256*1024**2),guard=self.guard,profile=self.profile)
        self.shards=ShardManager(self.ledger,self.data,self.guard,self.config,self.calibration,self.source,self.anchors,excluded_families=self.excluded)
        if self.state['progress']['active'] is not None:
            row=self.ledger.db.execute('SELECT id,path,sha,rows FROM shards WHERE id=?',(self.state['progress']['active'],)).fetchone()
            if row is None:raise RuntimeError('Active recovery shard is missing from ledger')
            self.shards._prepared(row)
        with self.span('data_preparation'):self.validation,self.test=self.shards.prepare_fixed()
        self.validation_encoded=self.cache.prepare('validation',self.validation)
        self.test_encoded=self.cache.prepare('test',self.test)
        self.holdout_digest=hashlib.sha256(canonical({'validation':[r['identity'] for r in self.validation],'test':[r['identity'] for r in self.test]}).encode()).hexdigest()
        if self.state.get('holdout_digest') and self.state['holdout_digest']!=self.holdout_digest:raise RuntimeError('Fixed holdout changed')
        self.state['holdout_digest']=self.holdout_digest
        with self.ledger.transaction():self.ledger.put('runtime_config',self.config)
        if legacy_migration:
            self.commit({'kind':'legacy_import_migrated','legacy_global_step':legacy['imported_step']})
            with self.ledger.transaction():self.ledger.put('legacy_import',dict(legacy,status='migrated',arc_manifest=str(self.manifest)))
        elif self.manifest is None:self.commit({'kind':'initialized'})
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
        from .devices import cpu_state,capture_xpu_rng
        from .durable import save_bundle,cleanup_bundles
        self.runtime.synchronize()
        self.state.update(model=cpu_state(self.model.state_dict()),optimizer=cpu_state(self.optimizer.state_dict()),scheduler=self.scheduler.state_dict(),
                          generator=self.generator.get_state(),rng=checkpoint.rng_state(),rng_xpu=capture_xpu_rng() if self.runtime.device.type=='xpu' else None,
                          effective_precision=self.runtime.precision,source_cursor=self.ledger.get('source_cursor'))
        self.export_model.load_state_dict(self.state['model'])
        event=self.event(extra)
        self.state['elapsed_seconds']=event['elapsed_seconds']
        with self.span('checkpoint_save'):
            self.state,self.manifest=save_bundle(self.ledger,self.checkpoints,self.export_model,self.state,self.guard,event,completed_shard=completed_shard)
        event['latest_checkpoint']=str(self.manifest)
        atomic_json(self.checkpoints/'status.json',{'status':'running',**event,'recovered':self.recovered})
        with self.span('checkpoint_housekeeping'):
            self.ledger.export_logs(self.checkpoints/'logs');cleanup_bundles(self.ledger,self.checkpoints)
        eta='?' if event['eta_seconds'] is None else f"{event['eta_seconds']:.1f}s"
        print(f"{event['progress_percent']:6.2f}% shard={event['shard']} positions={event['positions_processed']}/{self.config['positions']} step={event['global_step']} policy={event.get('policy_loss')} wdl={event.get('wdl_loss')} validation={event.get('validation_loss')} {event['positions_per_second']:.1f}pos/s ETA={eta} RSS={event['memory_rss_bytes']/1024**2:.1f}MiB disk={event['disk_bytes']/1024**2:.1f}MiB checkpoint={self.manifest}",flush=True)
    def train(self):
        from .model_adapter import torch,weighted_losses,measurements
        from .fastdata import Prefetch
        self.setup();p=self.state['progress']
        self.shards.cleanup(p['completed'],self.config['keep_shards'])
        while p['positions_processed']<self.config['positions']:
            self.guard.check(force=True)
            if p['active'] is None:
                with self.span('data_preparation'):shard=self.shards.next_shard(self.config['positions']-p['positions_processed'])
                if shard is None:raise RuntimeError('Source ended before requested balanced training positions')
                p.update(active=shard['id'],epoch=0,offset=0,permutation=[])
                self.commit({'kind':'shard_started'});p=self.state['progress']
            shard_id=p['active'];row=self.ledger.db.execute('SELECT path,sha,rows FROM shards WHERE id=?',(shard_id,)).fetchone()
            if row is None or digest(row[0])!=row[1]:raise RuntimeError('Active shard missing or corrupt; retained for inspection')
            identities=[r['identity'] for _,r in self.ledger.rows(shard=shard_id)]
            if len(identities)!=row[2]:raise RuntimeError('Active shard row count differs from durable ledger')
            # A single bounded current-shard + replay cache replaces repeated
            # SQLite JSON reads, FEN parsing and legal move generation each epoch.
            with self.span('data_loading'):
                current_records=[r for _,r in self.ledger.rows(shard=shard_id)]
                unique={r['identity']:r for r in current_records}
                for record in self.state['replay']:unique.setdefault(record['identity'],record)
                cache_records=list(unique.values());lookup={r['identity']:i for i,r in enumerate(cache_records)}
                encoded=self.cache.prepare('shard-'+str(shard_id),cache_records)
            resident=None
            if self.config.get('cache_mode')=='device' and self.runtime.device.type=='xpu':
                # Explicit opt-in only, and independently bounded to 15% of the
                # shared system-memory budget. mmap remains available for large shards.
                dense_bytes=len(cache_records)*(1280*4+4672*4+4672+3*4+2*4)
                if dense_bytes>min(int(self.config['max_memory_gb']*1024**3*.15),256*1024**2):
                    raise RuntimeError('Device cache exceeds bounded cache budget; use auto or mmap')
                self.runtime.synchronize();transfer_started=time.perf_counter()
                with self.span('host_to_device'):
                    resident=self.runtime.transfer(encoded.batch(list(range(len(cache_records)))))
                self.runtime.synchronize()
                self.execution.transfer_seconds+=time.perf_counter()-transfer_started
            while p['epoch']<self.config['epochs']:
                if not p['permutation']:
                    p['permutation']=[identities[i] for i in torch.randperm(len(identities),generator=self.generator).tolist()];p['offset']=0
                replay=self.state['replay'];extra=max(0,self.config['batch_size']//5) if replay else 0
                current_batch=max(1,self.config['batch_size']-extra)
                permutation=list(p['permutation']);begin=p['offset']
                local_generator=torch.Generator();local_generator.set_state(self.generator.get_state())
                def plans():
                    for offset in range(begin,len(permutation),current_batch):
                        chosen=permutation[offset:offset+current_batch];replay_ids=[]
                        replay_count=min(extra,self.config['batch_size']-len(chosen))
                        if replay_count:
                            for index in torch.randint(len(replay),(replay_count,),generator=local_generator).tolist():replay_ids.append(replay[index]['identity'])
                        yield ([lookup[i] for i in chosen+replay_ids],len(chosen),local_generator.get_state().clone())
                def prepare(plan):
                    indices,count,rng_after=plan
                    with self.profile.span('data_loading',sync=False):
                        cpu_tensors=encoded.batch(indices) if resident is None else None
                    return indices,count,rng_after,cpu_tensors
                queue=Prefetch(plans(),prepare_fn=prepare,depth=self.config.get('prefetch',0))
                try:
                    for indices,count,rng_after,cpu_tensors in queue:
                        self.guard.check()
                        if resident is None:tensors=self.execution.to_device(cpu_tensors)
                        else:
                            with self.span('device_batch_gather'):
                                index_tensor=torch.tensor(indices,device=self.runtime.device,dtype=torch.long)
                                tensors=tuple(t.index_select(0,index_tensor) for t in resident)
                        self.execution.step(tensors)
                        # Only consumed batches advance the authoritative RNG.
                        # A producer may prepare future batches but cannot commit them.
                        self.generator.set_state(rng_after)
                        p['offset']+=count;self.state['step']+=1
                        if self.state['step']%self.config['checkpoint_every']==0:
                            self.commit(self.execution.metrics());p=self.state['progress']
                finally:queue.close()
                p['epoch']+=1;p['offset']=0;p['permutation']=[]
            # Evaluate before recording a completed shard or making it eligible
            # for deletion. Failed validation/export never consumes that shard.
            with self.span('validation'):validation=measurements(self.model,self.validation,self.runtime,self.validation_encoded)
            if any(v is not None and not math.isfinite(v) for k,v in validation.items() if k.endswith('loss')):raise RuntimeError('Nonfinite independent validation')
            active_records=(r for _,r in self.ledger.rows(shard=shard_id))
            self.state['replay']=replay_update(self.state['replay'],active_records,self.config['replay_size'],self.config['seed'])
            p['completed'].append(shard_id);p['positions_processed']+=len(identities)
            p.update(active=None,epoch=0,offset=0,permutation=[])
            self.commit({**self.execution.metrics(),'kind':'shard_completed','validation_loss':validation['policy_loss'],'validation_wdl_loss':validation['wdl_loss']},completed_shard=shard_id)
            p=self.state['progress']
            self.shards.cleanup(p['completed'],self.config['keep_shards'])
            encoded.close();resident=None
            if not self.config['keep_shards']:self.cache.remove('shard-'+str(shard_id))
        with self.span('validation'):
            final_validation=measurements(self.model,self.validation,self.runtime,self.validation_encoded)
            final_test=measurements(self.model,self.test,self.runtime,self.test_encoded)
        self.runtime.synchronize()
        final={'validation':final_validation,'test':final_test,
               'holdout_digest':self.holdout_digest,'positions':p['positions_processed'],'global_step':self.state['step'],
               'completed_shards':p['completed'],'candidate_manifest':str(self.manifest),'champion_overwritten':False,
               'calibration_status':self.calibration.artifact['status'] if self.calibration else 'unavailable',
               'quality':self.ledger.get('quality_stats',{}),'source_cursor':self.ledger.get('source_cursor'),
               'execution':self.execution.snapshot(),'device':self.runtime.decision,'gpu_memory':self.runtime.memory(),
               'resources':self.sampler.snapshot() if self.sampler is not None else None,
               'process_seconds':time.perf_counter()-self.started,
               'promotion':{'eligible':False,'reason':'candidate only; independent outcome/tactical/match gates required; champion never automatically overwritten'}}
        atomic_json(self.checkpoints/'training-report.json',final)
        if self.profile is not None:self.profile.write()
        path=self.checkpoints/'status.json'
        status=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        status.update(status='completed',progress_percent=100.,**{k:final[k] for k in ['global_step','positions']})
        atomic_json(self.checkpoints/'status.json',status)
        return final
    def close(self):
        if self.shards is not None:self.shards.close()
        self.ledger.close()
        for name in ['validation_encoded','test_encoded']:
            if hasattr(self,name):getattr(self,name).close()
        if self.profile is not None:self.profile.write()
        if self.sampler is not None:self.sampler.stop()
        if self.cache is not None:self.cache.close()
        if self.profile is not None:self.profile.close()
        from .profiling import install_profile
        install_profile(None)
