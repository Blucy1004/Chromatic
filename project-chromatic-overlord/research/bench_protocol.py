#!/usr/bin/env python3
"""Native protocol checks for optional telemetry and all three research budgets."""
import argparse,json,platform,sys
from pathlib import Path
import chess
from run import Engine,MODELS,options,identity,sha,atomic

def main():
    p=argparse.ArgumentParser();p.add_argument('--engine',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--require-windows',action='store_true');a=p.parse_args()
    if a.require_windows:assert platform.system()=='Windows'
    checks=[]
    for model,path in MODELS.items():
        for mask in [0,15,16,31]:
            with Engine(a.engine,options(mask),model=path) as e:
                loaded=identity(e,path);e.position(chess.STARTING_FEN)
                for mode,budget in [('time',100),('nodes',32),('nn',16)]:
                    e.send(f'zero bench {mode} {budget} 1');r=json.loads(e.until(lambda s:s.startswith('{'),5)[-1]);assert 'error' not in r
                    assert chess.Move.from_uci(r['bestmove']) in chess.Board().legal_moves
                    q=r['research'];assert q and sum(q['leaf_depth'])==q['depth_samples'];assert r['search_ns']>0
                    if mode=='nn':assert r['neural_evals']+r['verification_evals']<=budget
                    if mode=='nodes':assert r['simulations']==budget
                    checks.append({'model':model,'mask':mask,'mode':mode,'budget':budget,'actual_nn':r['neural_evals']+r['verification_evals'],'search_ns':r['search_ns'],'loaded_sha256':loaded['sha256']})
                if mask==0:
                    for command in ['zero bench invalid 1 1','zero bench time 0 1','zero bench nn -1 1','zero bench nodes 100001 1','zero bench time']:
                        e.send(command);r=json.loads(e.until(lambda s:s.startswith('{'))[-1]);assert 'error' in r;checks.append({'model':model,'rejected':command})
                r=e.search(chess.STARTING_FEN,ms=100);assert r['detail']['research'] is not None
                e.set({'ResearchMetrics':False});r=e.search(chess.STARTING_FEN,ms=100);assert r['detail']['research'] is None
    atomic(a.out,{'passed':True,'os':platform.system(),'engine_sha256':sha(a.engine),'checks':checks})
    print(json.dumps({'passed':True,'checks':len(checks),'os':platform.system()}))
if __name__=='__main__':main()
