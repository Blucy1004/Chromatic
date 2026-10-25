#!/usr/bin/env python3
"""Rebuild auditable CSV/JSON summaries from completed immutable research cells."""
import argparse,csv,gzip,hashlib,importlib.util,itertools,json,math,statistics,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import chess
spec=importlib.util.spec_from_file_location('factorial_statistics',Path(__file__).with_name('factorial_stats.py'))
st=importlib.util.module_from_spec(spec);spec.loader.exec_module(st)
BITS='CAFTP';MODELS=['zero128','lichess100','lichess1000','lichess10000']
MATE_CACHE={}
def mate_diagnostic(r):
    key=r['fen']
    if key not in MATE_CACHE:
        board=chess.Board(key);mates=[]
        for move in list(board.legal_moves):
            board.push(move)
            if board.is_checkmate():mates.append(move.uci())
            board.pop()
        MATE_CACHE[key]=mates
    mates=MATE_CACHE[key];selected=r['result']['bestmove'] in mates;expected=r['expected'] in mates
    return {'root_has_mate1':int(bool(mates)),'expected_mate1':int(expected),'selected_mate1':int(selected),
            'mate_equivalent_hit':int(r['hit'] or (expected and selected)),
            'admission_mate1':int(r['result']['mate_proven'] and r['result']['proof_nodes']==0)}

def read(p):return json.loads(p.read_text())
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def csvfile(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def label(mask):return ''.join(c for i,c in enumerate(BITS) if mask>>i&1) or 'base'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def finite(v):return None if isinstance(v,float) and not math.isfinite(v) else v

def ingest(root,case_writer,scope):
    manifest=read(root/'manifest.json');manifest_sha=sha(root/'manifest.json');osname=manifest['os'];cells=[];skipped=[];seen=set()
    for meta_path in sorted(root.glob('*/*.meta.json')):
        meta=read(meta_path);path=meta_path.with_name(meta_path.name.replace('.meta.json','.jsonl.gz'))
        if not meta.get('complete') or not path.exists():skipped.append(str(meta_path));continue
        assert sha(path)==meta['raw_gzip_sha256'],str(path)
        assert meta['engine_sha256']==manifest['engine_sha256']
        loaded=meta['loaded_model'];assert loaded==meta['loaded_model_after'] and loaded['sha256']==meta['model_sha256']==manifest['models'][meta['model']]['sha256']
        raw=[json.loads(x) for x in gzip.decompress(path.read_bytes()).decode().splitlines()]
        assert [x['id'] for x in raw]==meta['ids'];assert len(set(meta['ids']))==len(raw)
        cell_identity=(meta_path.parent.name,raw[0]['cell']);assert cell_identity not in seen;seen.add(cell_identity)
        simple=[];hist=np.zeros(513,dtype=np.int64)
        for r in raw:
            assert r['stage']==meta_path.parent.name
            for field in ['model','mask','mode','budget']:assert r[field]==meta[field]
            z=r['result'];q=z['research'];assert len(q['leaf_depth'])==513
            hist+=q['leaf_depth'];assert sum(q['leaf_depth'])==q['depth_samples']
            assert q['node_creations']+1-z['nodes_evicted']==z['arena_nodes']
            if meta['mode']=='nodes':assert z['simulations']==meta['budget']
            if meta['mode']=='nn':assert z['neural_evals']+z['verification_evals']<=meta['budget']
            for bit,counter in [(0,'counterexample_calls'),(1,'adaptive_calls'),(2,'forcing_checks'),(3,'material_calls')]:
                if not (r['mask'] & (1<<bit)):assert q.get(counter,0)==0
            if not (r['mask'] & 13):assert q.get('verification_calls',0)==0
            if not (r['mask'] & 12):assert q.get('admission_calls',0)==0
            b={'scope':scope,'os':osname,'stage':r['stage'],'model':r['model'],'mask':r['mask'],'factors':label(r['mask']),
               'mode':r['mode'],'budget':r['budget'],'repeat':r['repeat'],'cell':r['cell'],'id':r['id'],'game':r['game'],
               'seed':manifest.get('seed',1),'threads':meta['options'].get('Threads',1),'run_manifest_sha256':manifest_sha,
               'hit':int(r['hit']),'expected':r['expected'],'bestmove':z['bestmove'],'pv':' '.join(q['pv']),
               'wall_ms':r['wall_ns']/1e6,'search_ms':z['search_ns']/1e6,'cpu_ms':r['cpu_seconds']*1000,
               'rss_bytes':r['rss_after_bytes'],'nn_total':z['neural_evals']+z['verification_evals'],
               'nn_mcts':z['neural_evals'],'nn_verification':z['verification_evals'],'completed_visits':q['depth_samples'],
               'simulations':z['simulations'],'attempted_simulations':z['attempted_simulations'],'aborted_simulations':z['aborted_simulations'],
               'max_depth':z['max_depth'],'prune_count':z['prune_count'],'nodes_evicted':z['nodes_evicted'],
               'edges_evicted':z['edges_evicted'],'retained_nodes':z['arena_nodes'],'memory_bytes':z['memory_bytes'],
               'peak_memory_bytes':z['peak_memory_bytes'],'verification_nodes':z['verification_nodes'],'verified_candidates':z['verified_candidates'],
               'verification_changed':int(z['verification_changed']),'mate_proven':int(z['mate_proven']),
               'verification_completed':int(bool(z['mate_proven'] or z['verified_candidates']>=min(4,len(z['visits'])))) if r['mask']&13 else '',
               'importance':z['importance'],'discarded_nn':z['discarded_neural_evaluations'],
               'engine_sha256':meta['engine_sha256'],'model_sha256':meta['model_sha256'],'options_sha256':hashlib.sha256(json.dumps(meta['options'],sort_keys=True).encode()).hexdigest()}
            b.update(mate_diagnostic(r))
            b['nn_completed_leaf_visits']=max(0,z['neural_evals']-(0 if z.get('terminal') else 1)-z['discarded_neural_evaluations'])
            b['non_nn_completed_endpoints']=q['depth_samples']-b['nn_completed_leaf_visits']
            assert b['non_nn_completed_endpoints']>=0
            for k,v in q.items():
                if k not in ['leaf_depth','pv','call_ns','algorithm_ns']:
                    b[k[:-3]+'_ms' if k.endswith('_ns') else k]=v/1e6 if k.endswith('_ns') else v
            b['algorithm_ms']=q['algorithm_ns']/1e6;b['cleanup_ms']=(z['search_ns']-q['algorithm_ns'])/1e6
            case_writer(b);simple.append(b)
        base={k:simple[0][k] for k in ['scope','os','stage','model','mask','factors','mode','budget','repeat','cell','engine_sha256','model_sha256','options_sha256']}
        base.update(n=len(simple),hits=sum(x['hit'] for x in simple),accuracy=sum(x['hit'] for x in simple)/len(simple),
                    source_games=len(set(x['game'] for x in simple)),complete=True)
        base['nn_per_search_second']=sum(x['nn_total'] for x in simple)/(sum(x['search_ms'] for x in simple)/1000)
        base['leaf_visits_per_search_second']=sum(x['completed_visits'] for x in simple)/(sum(x['search_ms'] for x in simple)/1000)
        cpu=sum(x['cpu_ms'] for x in simple)/1000;base['nn_per_cpu_second']=sum(x['nn_total'] for x in simple)/cpu if cpu else None
        excluded=set(base)|{'id','game','hit','expected','bestmove','pv'}
        for k,v in simple[0].items():
            if k not in excluded and isinstance(v,(float,int)):
                values=[x[k] for x in simple];base['mean_'+k]=statistics.mean(values);base['median_'+k]=statistics.median(values);base['max_'+k]=max(values)
        total=int(hist.sum());base['pooled_depth_mean']=float(hist@np.arange(513)/total) if total else 0
        base['pooled_depth_median']=int(np.searchsorted(np.cumsum(hist),(total+1)//2)) if total else 0
        base['pooled_depth_max']=int(np.flatnonzero(hist)[-1]) if total else 0
        cells.append({'summary':base,'cases':simple,'meta':meta})
    return {'root':str(root),'scope':scope,'manifest':manifest,'cells':cells,'incomplete_cells':skipped,
            'completed':read(root/'completed.json') if (root/'completed.json').exists() else None,
            'usage':read(root/'usage.json') if (root/'usage.json').exists() else None}

def vectors(cells):
    bymask={c['summary']['mask']:c for c in cells};first=next(iter(bymask.values()))['cases'];ids=[r['id'] for r in first]
    groups=[r['game'] for r in first];out={}
    for mask,c in bymask.items():
        assert [r['id'] for r in c['cases']]==ids
        out[mask]=np.array([r['hit'] for r in c['cases']],float)
    return out,groups

def estimates(run,repetitions):
    grouped=defaultdict(list)
    for c in run['cells']:
        s=c['summary']
        if s['repeat']==0 and s['stage'] not in ['legacy_proof','scalar_fixed']:
            grouped[s['stage'],s['model'],s['mode'],s['budget']].append(c)
    contrasts=[];factorial=[];interactions=[];frontier=[]
    for (stage,model,mode,budget),cells in grouped.items():
        values,groups=vectors(cells)
        common={'scope':run['scope'],'os':run['manifest']['os'],'stage':stage,'model':model,'mode':mode,'budget':budget,'n':len(groups)}
        if 0 in values:
            masks=sorted(set(values)-{0});diffs=np.stack([values[m]-values[0] for m in masks],axis=1)
            ci=st.cluster_bootstrap(diffs,groups,repetitions) if masks else None;ps=[];items=[]
            for i,m in enumerate(masks):
                g=int(((values[m]==1)&(values[0]==0)).sum());l=int(((values[m]==0)&(values[0]==1)).sum());p=st.mcnemar(g,l);ps.append(p)
                lookup={c['summary']['mask']:c for c in cells};costs={}
                for metric in ['cpu_ms','wall_ms','completed_visits','nn_total','verification_nodes','node_creations']:
                    old=lookup[0]['summary']['mean_'+metric];new=lookup[m]['summary']['mean_'+metric]
                    costs['delta_'+metric]=new-old;costs['ratio_'+metric]=new/old if old else None
                base_cases=lookup[0]['cases'];new_cases=lookup[m]['cases']
                costs['gains_expected_mate1']=sum(a['hit'] and not b['hit'] and a['expected_mate1'] for a,b in zip(new_cases,base_cases))
                costs['losses_expected_mate1']=sum(b['hit'] and not a['hit'] and a['expected_mate1'] for a,b in zip(new_cases,base_cases))
                costs['gains_other']=g-costs['gains_expected_mate1'];costs['losses_other']=l-costs['losses_expected_mate1']
                items.append({**common,'comparison':f'{label(m)}-base','mask':m,'gains':g,'losses':l,'delta_pp':ci['mean'][i]*100,
                              'lo95_pp':ci['ci95'][i][0]*100,'hi95_pp':ci['ci95'][i][1]*100,'p_exact':p,'clusters':ci['clusters'],**costs})
            for item,p in zip(items,st.holm(ps)):item['p_holm_within_model_stage']=p
            contrasts.extend(items)
        if len(values)==32:
            designs=[('all5',range(5),0),('serial4',range(4),0),('parallel4',range(4),16)]
            for name,bits,bg in designs:
                bits=list(bits);s=st.shapley(values,bits,bg);assert np.allclose(s.sum(axis=1),values[bg|sum(1<<b for b in bits)]-values[bg])
                ci=st.cluster_bootstrap(s,groups,repetitions)
                cost_shapley={metric:st.shapley({c['summary']['mask']:np.array([r[metric] for r in c['cases']]) for c in cells},bits,bg).mean(axis=0)
                              for metric in ['cpu_ms','wall_ms','completed_visits','nn_total']}
                for i,bit in enumerate(bits):
                    full=bg|sum(1<<b for b in bits);alone=values[bg|1<<bit]-values[bg];removed=values[full]-values[full&~(1<<bit)]
                    aci=st.cluster_bootstrap(np.stack([alone,removed],axis=1),groups,repetitions)
                    factorial.append({**common,'design':name,'factor':BITS[bit],'shapley_pp':ci['mean'][i]*100,
                        'lo95_pp':ci['ci95'][i][0]*100,'hi95_pp':ci['ci95'][i][1]*100,
                        'simultaneous_lo95_pp':ci['simultaneous_ci95'][i][0]*100,'simultaneous_hi95_pp':ci['simultaneous_ci95'][i][1]*100,
                        'standalone_pp':float(alone.mean()*100),'standalone_lo95_pp':aci['ci95'][0][0]*100,'standalone_hi95_pp':aci['ci95'][0][1]*100,
                        'full_remove_pp':float(removed.mean()*100),'remove_lo95_pp':aci['ci95'][1][0]*100,'remove_hi95_pp':aci['ci95'][1][1]*100,
                        **{'shapley_'+metric:float(v[i]) for metric,v in cost_shapley.items()}})
            masks=[m for m in range(32) if m.bit_count()>=2];x=np.stack([st.mobius(values,m) for m in masks],axis=1);ci=st.cluster_bootstrap(x,groups,repetitions)
            for i,m in enumerate(masks):
                interactions.append({**common,'mask':m,'factors':label(m),'order':m.bit_count(),'interaction_pp':ci['mean'][i]*100,
                    'lo95_pp':ci['ci95'][i][0]*100,'hi95_pp':ci['ci95'][i][1]*100,
                    'simultaneous_lo95_pp':ci['simultaneous_ci95'][i][0]*100,'simultaneous_hi95_pp':ci['simultaneous_ci95'][i][1]*100,'exploratory':True})
        # Pareto efficiency is descriptive. It is NOT an uncertainty-aware winner test.
        for cost in ['cpu_ms','wall_ms','nn_total','completed_visits']:
            for c in cells:
                a=c['summary'];dom=[]
                for other in cells:
                    b=other['summary']
                    if b['accuracy']>=a['accuracy'] and b['mean_'+cost]<=a['mean_'+cost] and (b['accuracy']>a['accuracy'] or b['mean_'+cost]<a['mean_'+cost]):dom.append(b['mask'])
                frontier.append({**common,'mask':a['mask'],'factors':label(a['mask']),'cost':cost,'mean_cost':a['mean_'+cost],
                                 'accuracy':a['accuracy'],'pareto':not dom,'dominated_by':' '.join(map(str,dom))})
    return contrasts,factorial,interactions,frontier

def model_effects(run,repetitions):
    groups=defaultdict(dict)
    for c in run['cells']:
        s=c['summary']
        if s['repeat']==0 and s['stage'] not in ['legacy_proof','scalar_fixed','ten_seconds']:groups[s['stage'],s['mask'],s['mode'],s['budget']][s['model']]=c
    out=[]
    for (stage,mask,mode,budget),models in groups.items():
        if 'zero128' not in models:continue
        base=models['zero128']['cases'];ids=[r['id'] for r in base];g=[r['game'] for r in base];old=[r['hit'] for r in base]
        candidates=sorted(set(models)-{'zero128'})
        if not candidates:continue
        diffs=[]
        for m in candidates:
            rows=models[m]['cases'];assert [r['id'] for r in rows]==ids;diffs.append(np.array([r['hit'] for r in rows])-old)
        ci=st.cluster_bootstrap(np.stack(diffs,axis=1),g,repetitions)
        for i,m in enumerate(candidates):
            new=np.array([r['hit'] for r in models[m]['cases']]);a=np.array(old);gain=int(((new==1)&(a==0)).sum());loss=int(((new==0)&(a==1)).sum())
            out.append({'scope':run['scope'],'os':run['manifest']['os'],'stage':stage,'mask':mask,'factors':label(mask),'mode':mode,'budget':budget,'model':m,'reference':'zero128','n':len(base),
                        'gains':gain,'losses':loss,'delta_pp':ci['mean'][i]*100,'lo95_pp':ci['ci95'][i][0]*100,'hi95_pp':ci['ci95'][i][1]*100,'p_exact':st.mcnemar(gain,loss)})
    # Holm across ALL checkpoint contrasts per run; not just one convenient configuration.
    for item,p in zip(out,st.holm([x['p_exact'] for x in out])):item['p_holm_run']=p
    return out

def marginal_effects(run,repetitions):
    out=[]
    for stage in ['screen','fixed_nodes','fixed_nn','holdout']:
        for model in MODELS:
            cells=[c for c in run['cells'] if c['summary']['stage']==stage and c['summary']['model']==model and c['summary']['repeat']==0]
            if (len(cells)!=32 and stage!='holdout') or len(cells)<2:continue
            values,groups=vectors(cells);lookup={c['summary']['mask']:c for c in cells}
            keys=[(bit,bg) for bit in range(5) for bg in sorted(values) if not (bg & (1<<bit)) and (bg | (1<<bit)) in values]
            if not keys:continue
            x=np.stack([values[bg|1<<bit]-values[bg] for bit,bg in keys],axis=1);ci=st.cluster_bootstrap(x,groups,repetitions)
            for i,(bit,bg) in enumerate(keys):
                a,b=values[bg|1<<bit],values[bg];gain=int(((a==1)&(b==0)).sum());loss=int(((a==0)&(b==1)).sum())
                out.append({'scope':run['scope'],'os':run['manifest']['os'],'stage':stage,'model':model,'factor':BITS[bit],'background':bg,'background_factors':label(bg),
                    'gains':gain,'losses':loss,'p_exact':st.mcnemar(gain,loss),'delta_pp':ci['mean'][i]*100,'lo95_pp':ci['ci95'][i][0]*100,'hi95_pp':ci['ci95'][i][1]*100,
                    **{'delta_'+metric:lookup[bg|1<<bit]['summary']['mean_'+metric]-lookup[bg]['summary']['mean_'+metric] for metric in ['cpu_ms','wall_ms','completed_visits','nn_total']}})
    return out

def model_search_interactions(run,repetitions):
    out=[]
    for stage in ['screen','holdout','fixed_nodes','fixed_nn']:
        lookup={(c['summary']['model'],c['summary']['mask']):c for c in run['cells'] if c['summary']['stage']==stage and c['summary']['repeat']==0}
        for model in MODELS[1:]:
            if ('zero128',0) not in lookup or (model,0) not in lookup:continue
            masks=sorted(m for mod,m in lookup if mod==model and m and ('zero128',m) in lookup)
            if not masks:continue
            base=lookup['zero128',0]['cases'];ids=[r['id'] for r in base];groups=[r['game'] for r in base]
            def v(mod,mask):
                rows=lookup[mod,mask]['cases'];assert [r['id'] for r in rows]==ids;return np.array([r['hit'] for r in rows],float)
            x=np.stack([(v(model,m)-v(model,0))-(v('zero128',m)-v('zero128',0)) for m in masks],axis=1);ci=st.cluster_bootstrap(x,groups,repetitions)
            for i,m in enumerate(masks):out.append({'scope':run['scope'],'os':run['manifest']['os'],'stage':stage,'model':model,'reference':'zero128','mask':m,'factors':label(m),
                'interaction_pp':ci['mean'][i]*100,'lo95_pp':ci['ci95'][i][0]*100,'hi95_pp':ci['ci95'][i][1]*100,
                'simultaneous_lo95_pp':ci['simultaneous_ci95'][i][0]*100,'simultaneous_hi95_pp':ci['simultaneous_ci95'][i][1]*100})
    return out

def holdout_confirmation(runs,repetitions):
    out=[];all_vectors=[];all_groups=None;all_ids=None;ps=[]
    for run in runs:
        path=Path(run['root'])/'selection-plan.json'
        if not path.exists():continue
        plan=read(path);a,b=[1<<x for x in plan['primary_pair']];needed={0,a,b,a|b};models={}
        for model in MODELS:
            cells=[c for c in run['cells'] if c['summary']['stage']=='holdout' and c['summary']['model']==model]
            if not needed.issubset({c['summary']['mask'] for c in cells}):continue
            v,g=vectors(cells);models[model]=(v,g)
        if not models:continue
        first=next(iter(models.values()))[1];names=list(models)
        x=np.stack([models[m][0][a|b]-models[m][0][a]-models[m][0][b]+models[m][0][0] for m in names],axis=1)
        ids=[r['id'] for c in run['cells'] if c['summary']['stage']=='holdout' and c['summary']['model']==names[0] for r in c['cases']][:len(first)]
        if all_groups is None:all_groups=first;all_ids=ids
        else:assert first==all_groups and ids==all_ids
        all_vectors.extend(x.T)
        items=[]
        for i,m in enumerate(names):
            v,g=models[m];armtests=[]
            for ref in [a,b]:
                gain=int(((v[a|b]==1)&(v[ref]==0)).sum());loss=int(((v[a|b]==0)&(v[ref]==1)).sum());p=st.mcnemar(gain,loss);ps.append(p)
                armtests.append({'reference_mask':ref,'gains':gain,'losses':loss,'delta_pp':float((v[a|b]-v[ref]).mean()*100),'p_exact':p})
            items.append({'scope':run['scope'],'os':run['manifest']['os'],'model':m,'pair':label(a|b),'mask':a|b,'n':len(g),
                'component_tests':armtests})
        out.extend(items)
    if out:
        ci=st.cluster_bootstrap(np.stack(all_vectors,axis=1),all_groups,repetitions);adjusted=iter(st.holm(ps))
        for i,item in enumerate(out):
            item.update(synergy_pp=ci['mean'][i]*100,lo95_pp=ci['ci95'][i][0]*100,hi95_pp=ci['ci95'][i][1]*100,
                simultaneous_lo95_pp=ci['simultaneous_ci95'][i][0]*100,simultaneous_hi95_pp=ci['simultaneous_ci95'][i][1]*100,
                simultaneous_family_size=len(out),component_test_family_size=len(ps))
            for test in item['component_tests']:test['p_holm_all_os_models']=next(adjusted)
            item['positive_synergy_confirmed']=item['simultaneous_lo95_pp']>0 and all(t['delta_pp']>0 and t['p_holm_all_os_models']<.05 for t in item['component_tests'])
    return out

def selected_holdout(runs,repetitions):
    items=[];arrays=[];groups=None;ids=None
    for run in runs:
        path=Path(run['root'])/'selection-plan.json'
        if not path.exists():continue
        plan=read(path)
        for model,mask in plan['best_serial'].items():
            cells=[c for c in run['cells'] if c['summary']['stage']=='holdout' and c['summary']['model']==model and c['summary']['mask'] in {0,mask}]
            if len(cells)!=2:continue
            v,g=vectors(cells);first=cells[0]['cases'];these_ids=[r['id'] for r in first]
            if groups is None:groups=g;ids=these_ids
            else:assert g==groups and these_ids==ids
            gain=int(((v[mask]==1)&(v[0]==0)).sum());loss=int(((v[mask]==0)&(v[0]==1)).sum());arrays.append(v[mask]-v[0])
            items.append({'scope':run['scope'],'os':run['manifest']['os'],'model':model,'mask':mask,'n':len(g),
                          'gains':gain,'losses':loss,'p_exact':st.mcnemar(gain,loss)})
    if items:
        ci=st.cluster_bootstrap(np.stack(arrays,axis=1),groups,repetitions);p=st.holm([r['p_exact'] for r in items])
        for i,r in enumerate(items):r.update(delta_pp=ci['mean'][i]*100,lo95_pp=ci['ci95'][i][0]*100,hi95_pp=ci['ci95'][i][1]*100,
            simultaneous_lo95_pp=ci['simultaneous_ci95'][i][0]*100,simultaneous_hi95_pp=ci['simultaneous_ci95'][i][1]*100,p_holm_family=p[i],family_size=len(items),
            scope_note='secondary family: the screen-locked serial candidate per model and OS; no holdout retuning; not a universal optimality test')
    return items

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,action='append',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=5000);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    # Stream case CSV so root visit lists/depth histograms do not remain in RAM.
    f=(a.out/'per-puzzle.csv').open('w',newline='');writer=None
    def emit(row):
        nonlocal writer
        if writer is None:writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
        assert set(row)==set(writer.fieldnames);writer.writerow(row)
    runs=[]
    for path in a.run:
        scope=path.parent.name+'/'+path.name;runs.append(ingest(path,emit,scope));print('ingested',scope,len(runs[-1]['cells']),flush=True)
    f.close();csvfile(a.out/'matrix.csv',[c['summary'] for r in runs for c in r['cells']])
    tables=[[],[],[],[]]
    for run in runs:
        for target,rows in zip(tables,estimates(run,a.bootstrap)):target.extend(rows)
    for name,table in zip(['paired-contrasts','shapley','interactions','pareto'],tables):csvfile(a.out/(name+'.csv'),table)
    csvfile(a.out/'model-effects.csv',[x for r in runs for x in model_effects(r,a.bootstrap)])
    marginals=[x for r in runs for x in marginal_effects(r,a.bootstrap)]
    holdout_marginals=[x for x in marginals if x['stage']=='holdout']
    for x,p in zip(holdout_marginals,st.holm([x['p_exact'] for x in holdout_marginals])):x['p_holm_holdout_edges_all_os_models']=p
    csvfile(a.out/'context-marginals.csv',marginals)
    csvfile(a.out/'model-search-interactions.csv',[x for r in runs for x in model_search_interactions(r,a.bootstrap)])
    save(a.out/'holdout-confirmation.json',holdout_confirmation(runs,a.bootstrap))
    save(a.out/'selected-holdout.json',selected_holdout(runs,a.bootstrap))
    save(a.out/'analysis-manifest.json',{'analysis_script_sha256':sha(Path(__file__)),'statistics_sha256':sha(Path(__file__).with_name('factorial_stats.py')),
        'bootstrap_repetitions':a.bootstrap,'runs':[{k:v for k,v in r.items() if k!='cells'} for r in runs],
        'excluded_model_contrasts':{'ten_seconds':'model-specific diagnostic gain/loss positions are not a paired model sample'},
        'inference_limits':'screen interactions exploratory; holdout one primary pair; cluster bootstrap assumes sampled source-game units; fixed corpus is not a random population sample; no Elo promotion'})
    print('wrote',a.out,flush=True)
if __name__=='__main__':main()
