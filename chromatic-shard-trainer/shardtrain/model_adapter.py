"""Reuse preserved CHROMATIC encoding, model, optimizer checkpoint format."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor'))
sys.dont_write_bytecode=True
import numpy as np
import torch
from zero.model import PolicyWDL,ModelConfig
from zero.encoding import INPUTS,ACTIONS,features,restore_board,action
from zero import checkpoint

def batch(records):
    n=len(records);x=np.zeros((n,INPUTS),np.float32);p=np.zeros((n,ACTIONS),np.float32)
    mask=np.zeros((n,ACTIONS),np.bool_);v=np.zeros((n,3),np.float32);pw=[];vw=[]
    for i,r in enumerate(records):
        board=restore_board(r)
        if not board.is_valid():raise ValueError('invalid normalized board')
        legal={action(m) for m in board.legal_moves};x[i]=features(board)
        for a in legal:mask[i,a]=True
        for a,q in r['policy']:
            if a not in legal or not np.isfinite(q) or q<0:raise ValueError('invalid normalized policy')
            p[i,a]+=q
        if abs(float(p[i].sum())-1)>1e-5:raise ValueError('policy normalization')
        v[i]=r['wdl']
        if not np.isfinite(v[i]).all() or abs(float(v[i].sum())-1)>1e-5 or (v[i]<0).any():raise ValueError('invalid normalized WDL')
        weights=[float(r['policy_weight']),float(r['value_weight'])]
        if not all(np.isfinite(w) and 0<=w<=1 for w in weights):raise ValueError('invalid confidence weights')
        pw.append(weights[0]);vw.append(weights[1])
    return tuple(torch.from_numpy(a) for a in [x,p,v,mask,np.asarray(pw,np.float32),np.asarray(vw,np.float32)])

def weighted_losses(model,records):
    x,p,v,mask,pw,vw=batch(records);policy,wdl=model(x)
    pl=-(p*torch.log_softmax(policy.masked_fill(~mask,-1e9),-1)).sum(-1)
    vl=-(v*torch.log_softmax(wdl,-1)).sum(-1)
    # Divide by all rows: a weak label's confidence reduces its contribution.
    loss=(pl*pw+vl*vw).mean()
    return loss,pl,vl,pw,vw,torch.softmax(wdl,-1)

def measurements(model,records):
    totals={'policy_numerator':0.,'policy_weight':0.,'value_numerator':0.,'value_weight':0.,'rows':0,'value_rows':0}
    model.eval()
    with torch.no_grad():
        for start in range(0,len(records),32):
            part=records[start:start+32];_,pl,vl,pw,vw,_=weighted_losses(model,part)
            totals['policy_numerator']+=float((pl*pw).sum());totals['policy_weight']+=float(pw.sum())
            totals['value_numerator']+=float((vl*vw).sum());totals['value_weight']+=float(vw.sum())
            totals['rows']+=len(part);totals['value_rows']+=int((vw>0).sum())
    totals['policy_loss']=totals['policy_numerator']/totals['policy_weight'] if totals['policy_weight'] else None
    totals['wdl_loss']=totals['value_numerator']/totals['value_weight'] if totals['value_weight'] else None
    return totals
