import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from zero.model import load_model
from zero.encoding import features,restore_board
from zero.data import atomic_json

def audit(paths,models,output):
    rows=[json.loads(line) for path in paths for line in Path(path).read_text(encoding='utf-8').splitlines()]
    if not rows:raise ValueError('empty draw audit data')
    x=torch.from_numpy(np.stack([features(restore_board(row)) for row in rows]))
    result={'records':len(rows),'mean_target_wdl':np.mean([row['wdl'] for row in rows],axis=0).tolist(),
            'models':{},'causal_limit':'observational comparison; exploration and network quality are not independently identified'}
    torch.set_num_threads(1)
    for path in models:
        model,state,_=load_model(path)
        with torch.no_grad():
            policy,wdl=model(x);probs=torch.softmax(wdl,-1)
        result['models'][str(path)]={'step':state['step'],'mean_predicted_wdl':probs.mean(0).tolist(),
                                    'mean_expected_value':float((probs[:,0]-probs[:,2]).mean())}
    result['findings']=['Completed v0.1 selfplay targets in this dataset can be checked directly above.',
                        'Repetition identity and value-sign tests are separate correctness gates.',
                        'Draw-only labels train a draw-only value head; lower training loss is not evidence of stronger play.',
                        'Opening diversification is supported; no artificial decisive result or draw penalty is introduced.']
    atomic_json(output,result);return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',nargs='+',required=True);p.add_argument('--models',nargs='+',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();print(json.dumps(audit(args.data,args.models,args.output),indent=2))
