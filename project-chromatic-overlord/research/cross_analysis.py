#!/usr/bin/env python3
"""Native cross-host numeric, fixed-work, timing and puzzle-level comparisons."""
import argparse,csv,gzip,json,math,struct
from pathlib import Path
import numpy as np
from analyze import csvfile,read,save,MODELS,label,st

def records(path):return [json.loads(x) for x in gzip.decompress(path.read_bytes()).decode().splitlines()]
def cells(root,stage):
    out={}
    for path in sorted((root/stage).glob('*.jsonl.gz')):
        meta=read(path.with_name(path.name.replace('.jsonl.gz','.meta.json')))
        if meta.get('complete'):out[meta['model'],meta['mask'],meta['repeat']]=(meta,records(path))
    return out
def ulp32(a,b):
    # Map IEEE sign/magnitude to monotonic unsigned integers, identifying +/-0.
    def ordered(x):
        v=np.asarray(x,dtype=np.float32).view(np.uint32).astype(np.int64)
        return np.where(v&0x80000000,0x80000000-(v&0x7fffffff),0x80000000+v)
    return np.abs(ordered(a)-ordered(b))
def numeric(linux,windows):
    summaries=[];case_rows=[]
    for model in MODELS:
        lp=linux/f'numeric-{model}.jsonl.gz';wp=windows/f'numeric-{model}.jsonl.gz'
        if not lp.exists() or not wp.exists():continue
        l={(x['backend'],x['id']):x for x in records(lp)};w={(x['backend'],x['id']):x for x in records(wp)};assert l.keys()==w.keys()
        for backend in ['SIMD','Scalar']:
            delta=[];ulps=[];order_equal=0;wdl_max=0;logit_max=0;exact=0
            for key in [k for k in l if k[0]==backend]:
                a,b=l[key],w[key];assert a['loaded_sha256']==b['loaded_sha256'];same=a['legal']==b['legal'];order_equal+=same
                if not same:raise ValueError('legal order mismatch')
                x=np.array(a['wdl']+a['logits']);y=np.array(b['wdl']+b['logits']);d=np.abs(x-y);u=ulp32(x,y)
                delta.extend(d);ulps.extend(u);exact+=bool(np.array_equal(x,y));wdl_max=max(wdl_max,float(d[:3].max()));logit_max=max(logit_max,float(d[3:].max(initial=0)))
                case_rows.append({'model':model,'backend':backend,'id':key[1],'legal_order_equal':same,'max_abs':float(d.max()),'max_float32_ulp':int(u.max()),'wdl_abs_max':float(d[:3].max()),'legal_logit_abs_max':float(d[3:].max(initial=0))})
            summaries.append({'model':model,'backend':backend,'positions':len(l)//2,'identical_json_positions':exact,'legal_order_equal':order_equal,'max_abs':float(max(delta)),'p99_abs':float(np.quantile(delta,.99)),
                              'max_float32_ulp':int(max(ulps)),'p99_float32_ulp':float(np.quantile(ulps,.99)),'wdl_max_abs':wdl_max,'legal_logit_max_abs':logit_max})
    return summaries,case_rows

def work(linux,windows):
    summaries=[];differences=[]
    for stage in ['fixed_nodes','fixed_nn','scalar_fixed','screen','one_second','five_seconds','ten_seconds','holdout']:
        l=cells(linux,stage);w=cells(windows,stage)
        for key in sorted(l.keys()&w.keys()):
            lm,lr=l[key];wm,wr=w[key];assert lm['options']==wm['options'];assert lm['ids']==wm['ids'];assert lm['model_sha256']==wm['model_sha256']
            same_move=same_n=same_q=same_pv=0;maxq=0;gains=losses=0
            for a,b in zip(lr,wr):
                x,y=a['result'],b['result'];assert a['id']==b['id'];assert a['fen']==b['fen'] and a['expected']==b['expected']
                same=x['bestmove']==y['bestmove'];same_move+=same
                nsame=[(v['uci'],v['n']) for v in x['visits']]==[(v['uci'],v['n']) for v in y['visits']];same_n+=nsame
                assert [v['uci'] for v in x['visits']]==[v['uci'] for v in y['visits']]
                dq=max([abs(v['q']-z['q']) for v,z in zip(x['visits'],y['visits'])] or [0]);maxq=max(maxq,dq);same_q+=dq==0
                pv=x['research']['pv']==y['research']['pv'];same_pv+=pv;gains+=a['hit'] and not b['hit'];losses+=b['hit'] and not a['hit']
                differences.append({'stage':stage,'model':key[0],'mask':key[1],'id':a['id'],'linux_move':x['bestmove'],'windows_move':y['bestmove'],
                    'linux_pv':' '.join(x['research']['pv']),'windows_pv':' '.join(y['research']['pv']),'linux_hit':int(a['hit']),'windows_hit':int(b['hit']),
                    'root_visit_counts_equal':nsame,'root_q_max_abs':dq,'linux_visits':x['research']['depth_samples'],'windows_visits':y['research']['depth_samples'],
                    'linux_nn':x['neural_evals']+x['verification_evals'],'windows_nn':y['neural_evals']+y['verification_evals'],
                    'linux_search_ms':x['search_ns']/1e6,'windows_search_ms':y['search_ns']/1e6,'linux_wall_ms':a['wall_ns']/1e6,'windows_wall_ms':b['wall_ns']/1e6,
                    'linux_cpu_ms':a['cpu_seconds']*1000,'windows_cpu_ms':b['cpu_seconds']*1000,
                    'linux_mcts_ms':x['research']['mcts_ns']/1e6,'windows_mcts_ms':y['research']['mcts_ns']/1e6,
                    'linux_verify_ms':x['research']['verification_ns']/1e6,'windows_verify_ms':y['research']['verification_ns']/1e6,
                    'linux_verify_candidates':x['verified_candidates'],'windows_verify_candidates':y['verified_candidates'],
                    'linux_ties':x['research']['selection_ties'],'windows_ties':y['research']['selection_ties'],
                    'linux_min_gap':x['research']['minimum_positive_gap'],'windows_min_gap':y['research']['minimum_positive_gap']})
            summaries.append({'stage':stage,'model':key[0],'mask':key[1],'factors':label(key[1]),'n':len(lr),'same_move':same_move,'same_root_counts':same_n,
                'same_root_q_exact':same_q,'max_q_abs':maxq,'same_pv':same_pv,'linux_only_hits':gains,'windows_only_hits':losses,
                'linux_hits':sum(a['hit'] for a in lr),'windows_hits':sum(a['hit'] for a in wr),
                'linux_total_wall_ms':sum(a['wall_ns'] for a in lr)/1e6,'windows_total_wall_ms':sum(a['wall_ns'] for a in wr)/1e6,
                'linux_total_nn':sum(a['result']['neural_evals']+a['result']['verification_evals'] for a in lr),
                'windows_total_nn':sum(a['result']['neural_evals']+a['result']['verification_evals'] for a in wr)})
    return summaries,differences

def interaction(linux,windows,repetitions):
    out=[]
    for stage in ['screen','holdout','fixed_nodes','fixed_nn']:
        l=cells(linux,stage);w=cells(windows,stage)
        for model in MODELS:
            masks=sorted(k[1] for k in l if k[0]==model and k[2]==0 and k in w)
            if 0 not in masks:continue
            def v(data,mask):return np.array([r['hit'] for r in data[model,mask,0][1]],float)
            groups=[r['game'] for r in l[model,0,0][1]]
            masks=[m for m in masks if m];x=np.stack([(v(l,m)-v(l,0))-(v(w,m)-v(w,0)) for m in masks],axis=1);ci=st.cluster_bootstrap(x,groups,repetitions)
            for i,m in enumerate(masks):out.append({'stage':stage,'model':model,'mask':m,'factors':label(m),'linux_minus_windows_search_effect_pp':ci['mean'][i]*100,
                'lo95_pp':ci['ci95'][i][0]*100,'hi95_pp':ci['ci95'][i][1]*100,'simultaneous_lo95_pp':ci['simultaneous_ci95'][i][0]*100,'simultaneous_hi95_pp':ci['simultaneous_ci95'][i][1]*100,
                'causal_limit':'joint OS/hardware/compiler contrast, not isolated OS effect'})
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--linux',type=Path,required=True);p.add_argument('--windows',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=5000);a=p.parse_args()
    lm,wm=[read(x/'manifest.json') for x in [a.linux,a.windows]]
    for key in ['cpp_sha256','runner_sha256','seed','old_fixture_sha256','holdout_sha256']:assert lm[key]==wm[key],key
    n,details=numeric(a.linux,a.windows);save(a.out/'numeric-comparison.json',n);csvfile(a.out/'numeric-per-puzzle.csv',details)
    s,d=work(a.linux,a.windows);csvfile(a.out/'cross-platform-cells.csv',s);csvfile(a.out/'cross-platform-puzzles.csv',d)
    csvfile(a.out/'os-search-interactions.csv',interaction(a.linux,a.windows,a.bootstrap))
    save(a.out/'cross-platform-controls.json',{'same_cpp_runner_seed_corpora':True,'same_model_files':all(lm['models'][m]['sha256']==wm['models'][m]['sha256'] for m in MODELS),
        'linux':{k:lm[k] for k in ['os','platform','processor','host','logical_cpus','engine_cpu_affinity','engine_sha256']},
        'windows':{k:wm[k] for k in ['os','platform','processor','host','logical_cpus','engine_cpu_affinity','engine_sha256']},
        'limit':'different host hardware and compiler; throughput differences cannot be assigned to OS alone'})
if __name__=='__main__':main()
