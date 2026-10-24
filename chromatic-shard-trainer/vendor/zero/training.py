from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
import torch
from . import checkpoint
from .data import atomic_json
from .model import ModelConfig,PolicyWDL
from .replay import Replay


def measurements(model,replay,ids,device):
    if not ids:return None
    model.eval();sums={'policy_loss':0.,'wdl_loss':0.,'policy_entropy':0.,'wdl_brier':0.}
    bins=np.zeros((10,3),dtype=np.float64);draw=0.;n=0
    with torch.no_grad():
        for start in range(0,len(ids),64):
            x,p,v,mask=replay.batch(ids[start:start+64],device)
            policy,wdl=model(x);logs=torch.log_softmax(policy.masked_fill(~mask,-1e9),-1)
            value_logs=torch.log_softmax(wdl,-1);probs=value_logs.exp();count=len(x)
            sums['policy_loss']+=float(-(p*logs).sum())
            sums['wdl_loss']+=float(-(v*value_logs).sum())
            sums['policy_entropy']+=float(-(logs.exp()*logs).sum())
            sums['wdl_brier']+=float(((probs-v)**2).sum())
            confidence,choice=probs.max(-1);target=v.gather(1,choice[:,None]).squeeze(1)
            for c,t in zip(confidence.cpu().numpy(),target.cpu().numpy()):
                index=min(9,int(c*10));bins[index]+=[1,c,t]
            draw+=float(probs[:,1].sum());n+=count
    result={key:value/n for key,value in sums.items()}
    result['wdl_ece']=float(sum(abs(c-t) for count,c,t in bins if count)/n)
    result['mean_predicted_draw']=draw/n;result['rows']=n
    return result


def train(paths,output,steps=1000,batch_size=64,lr=.001,seed=1,resume=None,init=None,
          width=64,blocks=2,save_every=100,device='cpu',capacity=10000,recent_fraction=.75,
          validation_fraction=.15,optimizer_name='adamw',weight_decay=1e-4,decay=.999):
    from .train import seed_all
    started=time.perf_counter()
    if steps<0 or batch_size<1 or save_every<1 or not np.isfinite(lr) or lr<=0 or optimizer_name not in ('adamw','sgd'):
        raise ValueError('invalid training settings')
    if not 0<decay<=1 or not 0<=weight_decay<=1:raise ValueError('invalid schedule or weight decay')
    if resume and init:raise ValueError('choose resume or init')
    torch.set_num_threads(1);seed_all(seed)
    replay=Replay(paths,capacity,recent_fraction,seed)
    training,validation=replay.split(validation_fraction,seed)
    settings={'version':2,'batch_size':batch_size,'lr':lr,'device':device,'decay':decay,
              'torch':str(torch.__version__),'capacity':capacity,'recent_fraction':recent_fraction,
              'validation_fraction':validation_fraction,'optimizer':optimizer_name,'weight_decay':weight_decay,'seed':seed}
    state,recovered=checkpoint.load(resume or init) if resume or init else (None,None)
    config=ModelConfig(**state['config']) if state else ModelConfig(width,blocks)
    model=PolicyWDL(config).to(device)
    if optimizer_name=='adamw':optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    else:optimizer=torch.optim.SGD(model.parameters(),lr=lr,momentum=.9,weight_decay=weight_decay)
    generator=torch.Generator().manual_seed(seed);start=0;sampling={'recent':0,'older':0}
    if state:
        model.load_state_dict(state['model'])
        if resume:
            if state.get('settings',{}).get('version')!=2:raise ValueError('v0.1 continuation requires --legacy; use --init to start CHROMATIC training')
            if replay.sources!=state['dataset'] or settings!=state['settings']:
                raise ValueError('resume requires identical dataset and training settings; use --init for a new generation')
            if state['optimizer'] is None:raise ValueError('no optimizer state; use --init')
            optimizer.load_state_dict(state['optimizer']);generator.set_state(state['generator'])
            start=state['step'];sampling=state.get('sampling',sampling);checkpoint.restore_rng(state['rng'])
    if steps<start:raise ValueError('steps is a total target smaller than checkpoint step')
    replay.persist(Path(output).with_suffix('.replay.json'))
    initial={'training':measurements(model,replay,training,device),'validation':measurements(model,replay,validation,device)}
    compute_seconds=0.;history=[]
    def save(step):
        checkpoint.save(output,{'config':asdict(config),'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                               'step':step,'generator':generator.get_state(),'dataset':replay.sources,
                               'settings':settings,'seed':seed,'sampling':sampling})
        model.export(Path(output).with_suffix('.zero'))
    for step in range(start,steps):
        before=time.perf_counter();model.train()
        chosen=torch.randint(len(training),(batch_size,),generator=generator)
        ids=[training[int(i)] for i in chosen]
        for index in ids:sampling['recent' if replay.rows[index].recent else 'older']+=1
        x,p,v,mask=replay.batch(ids,device)
        for group in optimizer.param_groups:group['lr']=lr*decay**step
        optimizer.zero_grad(set_to_none=True);policy,wdl=model(x)
        pl=-(p*torch.log_softmax(policy.masked_fill(~mask,-1e9),-1)).sum(-1).mean()
        vl=-(v*torch.log_softmax(wdl,-1)).sum(-1).mean();loss=pl+vl
        if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();compute_seconds+=time.perf_counter()-before
        history.append({'step':step+1,'policy_loss':float(pl.detach()),'wdl_loss':float(vl.detach()),'gradient_norm_before_clip':float(grad)})
        if (step+1)%save_every==0:save(step+1)
    if steps==start or steps%save_every:save(steps)
    final={'training':measurements(model,replay,training,device),'validation':measurements(model,replay,validation,device)}
    elapsed=time.perf_counter()-started;samples=(steps-start)*batch_size
    report={'format':2,'records':len(replay.rows),'start_step':start,'step':steps,'recovered_from':str(recovered) if recovered else None,
            'device':device,'replay':replay.stats,'sampling_counts':sampling,'initial':initial,'final':final,
            'last':history[-1] if history else None,'compute_seconds':compute_seconds,'total_seconds':elapsed,
            'compute_samples_per_second':samples/compute_seconds if compute_seconds else None,
            'total_samples_per_second_including_load_save_validation':samples/elapsed,
            'history':history[-256:]}
    atomic_json(Path(output).with_suffix('.train.json'),report)
    return report
