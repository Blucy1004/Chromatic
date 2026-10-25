#!/usr/bin/env python3
"""Validate every PGN against raw trace and compute opening-pair uncertainty."""
import argparse,io,json,math
from collections import defaultdict
from pathlib import Path
import chess,chess.pgn,numpy as np
from analyze import csvfile,read,save

def elo(score):
    return '-infinity' if score<=0 else 'infinity' if score>=1 else 400*math.log10(score/(1-score))
def analyze(root):
    manifest=read(root/'manifest.json');summaries=[];flat=[]
    for path in sorted((root/'matches').glob('*.json')):
        rows=read(path);pairs=defaultdict(list)
        for r in rows:
            assert not r['error'],(path,r['error']);assert r['model_sha256']==manifest['models'][r['model']]['sha256'];assert r['engine_sha256']==manifest['engine_sha256']
            board=chess.Board()
            for m in r['moves']:
                move=chess.Move.from_uci(m);assert move in board.legal_moves;board.push(move)
            pgn=chess.pgn.read_game(io.StringIO(r['pgn']));assert not pgn.errors;assert [m.uci() for m in pgn.mainline_moves()]==r['moves']
            assert pgn.headers['Result']==r['result'];outcome=board.outcome(claim_draw=True)
            if outcome:
                assert outcome.result()==r['result'];score=.5 if outcome.winner is None else float(outcome.winner==r['candidate_white']);assert score==r['score']
            else:assert r['result']=='*' and r['score'] is None
            trace=r['trace'];assert len(trace)==r['plies'];assert [x['move'] for x in trace]==r['moves'][-len(trace):] if trace else True
            pairs[r['pair']].append(r)
            flat.append({**{k:r[k] for k in ['model','mask','pair','candidate_white','score','result','termination','plies','threads','movetime_ms','model_sha256','engine_sha256']},
                'os':manifest['os'],'scope':root.parent.name+'/'+root.name,'validated':True})
        finished=[r for r in rows if r['score'] is not None];complete=[]
        for pair,rs in pairs.items():
            assert len(rs)<=2
            if len(rs)==2:assert {r['candidate_white'] for r in rs}=={False,True}
            if len(rs)==2 and all(r['score'] is not None for r in rs):complete.append(sum(r['score'] for r in rs)/2)
        point=float(np.mean(complete)) if complete else None
        if complete:
            rng=np.random.default_rng(240125);draw=np.mean(rng.choice(complete,size=(20000,len(complete)),replace=True),axis=1);boot=np.quantile(draw,[.025,.975]).tolist()
            radius=math.sqrt(math.log(40)/(2*len(complete)));bound=[max(0,point-radius),min(1,point+radius)]
        else:boot=[0,1];bound=[0,1]
        summaries.append({'scope':root.parent.name+'/'+root.name,'os':manifest['os'],'series':path.stem,'model':rows[0]['model'],'mask':rows[0]['mask'],
            'games':len(rows),'finished':len(finished),'unfinished':len(rows)-len(finished),'wins':sum(r['score']==1 for r in finished),'draws':sum(r['score']==.5 for r in finished),'losses':sum(r['score']==0 for r in finished),
            'complete_opening_pairs':len(complete),'paired_score':point,'pair_bootstrap_ci95':boot,'pair_hoeffding_ci95':bound,
            'descriptive_elo':elo(point) if point is not None else None,'elo_bootstrap_ci95':[elo(x) for x in boot],'elo_hoeffding_ci95':[elo(x) for x in bound],
            'pgn_trace_validation':True,'confirmed_elo_gain':False,'promote':False,
            'reason':'bounded fixed 8-opening suite is below preregistered 64 independent pairs; incomplete games are not draws; intervals are conditional on opening-sampling assumptions'})
    return summaries,flat
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,action='append',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();s=[];r=[]
    for root in a.run:
        x,y=analyze(root);s.extend(x);r.extend(y)
    save(a.out/'matches.json',s);csvfile(a.out/'games.csv',r)
if __name__=='__main__':main()
