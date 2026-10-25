#!/usr/bin/env python3
"""Repeated wall-time variability without inflating independent puzzle counts."""
import argparse,csv,itertools,json,statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
from analyze import csvfile,save,st
def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=5000);a=p.parse_args()
    groups=defaultdict(lambda:defaultdict(list))
    with (a.analysis/'per-puzzle.csv').open() as f:
        for r in csv.DictReader(f):
            if r['stage'] in ['screen','anchors']:groups[r['scope'],r['os'],r['model'],int(r['mask'])][int(r['repeat'])].append(r)
    summaries=[];flat=[]
    for (scope,os,model,mask),repeats in groups.items():
        if set(repeats)!={0,1,2}:continue
        rs=[repeats[i] for i in range(3)];ids=[r['id'] for r in rs[0]]
        for rows in rs:assert [r['id'] for r in rows]==ids
        hits=np.array([[int(r['hit']) for r in rows] for rows in rs]);score=hits.mean(axis=1);mean=hits.mean(axis=0)
        ci=st.cluster_bootstrap(mean,[r['game'] for r in rs[0]],a.bootstrap);variability=[]
        for i,j in itertools.combinations(range(3),2):
            variability.append({'repeats':[i,j],'correctness_flips':int((hits[i]!=hits[j]).sum()),
                'move_flips':sum(x['bestmove']!=y['bestmove'] for x,y in zip(rs[i],rs[j]))})
        summaries.append({'scope':scope,'os':os,'model':model,'mask':mask,'puzzles':len(ids),'repeats':3,
            'run_hits':hits.sum(axis=1).tolist(),'accuracy_mean':float(mean.mean()),'accuracy_mean_cluster_ci95':ci['ci95'][0],
            'run_accuracy_sd':float(score.std(ddof=1)),'puzzles_with_correctness_variation':int(((hits.max(axis=0)-hits.min(axis=0))>0).sum()),
            'comparisons':variability,'independent_source_game_units':ci['clusters']})
        for k,id in enumerate(ids):
            flat.append({'scope':scope,'os':os,'model':model,'mask':mask,'id':id,'game':rs[0][k]['game'],
                'correct_012':''.join(str(int(hits[i,k])) for i in range(3)),
                **{f'moves_r{i}':rs[i][k]['bestmove'] for i in range(3)},
                **{f'visits_r{i}':rs[i][k]['completed_visits'] for i in range(3)},
                **{f'nn_r{i}':rs[i][k]['nn_total'] for i in range(3)}})
    save(a.analysis/'repeats.json',summaries);csvfile(a.analysis/'repeat-puzzles.csv',flat)
if __name__=='__main__':main()
