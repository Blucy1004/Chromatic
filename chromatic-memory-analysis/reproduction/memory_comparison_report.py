"""Audit legacy/recycled MCTS logs and generate static comparison artifacts.

Reads completed experiment logs only. It never builds an engine, starts a chess
search, changes a model, or edits a preserved release.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CASES=['start','before_Rg1_100ms','before_Rg1_1s','before_Ra6_10s']
LABEL={'start':'시작 위치 · 1.g4 직전','before_Rg1_100ms':'5.Rg1 직전',
       'before_Rg1_1s':'4.Rg1 직전(보조)','before_Ra6_10s':'2...Ra6 직전'}
EN={'start':'Start (before 1.g4)','before_Rg1_100ms':'Before 5.Rg1',
    'before_Rg1_1s':'Before 4.Rg1 (auxiliary)','before_Ra6_10s':'Before 2...Ra6'}
TARGET={'start':'g2g4','before_Rg1_100ms':'h1g1','before_Rg1_1s':'h1g1','before_Ra6_10s':'a8a6'}
SAN={'g2g4':'g4','h1g1':'Rg1','a8a6':'Ra6','c2c3':'c3','e7e6':'e6',
     'g1f3':'Nf3','g8f6':'Nf6','b1c3':'Nc3','e2e4':'e4','e4e5':'e5','f3h4':'Nh4'}
VARS=['legacy','recycled']
VN={'legacy':'기존','recycled':'수정'}
COL={'legacy':'#64748b','recycled':'#dc2626'}
MEMCOL={64:'#2563eb',512:'#e76f21'}
MIB=1024*1024


def move(x):return x.get('uci',x.get('move',''))
def visits(x):return int(x.get('visits',x.get('n',0)))
def elapsed(r):return r['result']['elapsed_ms']
def cn(r,k,default=0):return r['result']['telemetry']['counters'].get(k,default)
def mean_or_none(values):return st.mean(values) if values else None
def fmt(x,n=2):return '—' if x is None else f'{x:,.{n}f}'


def depths(hist):
    total=int(sum(hist))
    if total==0:return {'samples':0,'mean':None,'median':None,'max':None,'p90':None,'p99':None}
    def rank(k):
        acc=0
        for d,n in enumerate(hist):
            acc+=n
            if acc>=k:return d
        raise AssertionError('histogram rank')
    return {'samples':total,'mean':sum(d*n for d,n in enumerate(hist))/total,
            'median':(rank((total+1)//2)+rank(total//2+1))/2,
            'max':max(d for d,n in enumerate(hist) if n),
            'p90':rank(math.ceil(total*.9)),'p99':rank(math.ceil(total*.99))}


def norm(hist):
    total=sum(hist)
    return np.asarray(hist,dtype=float)/total if total else np.zeros(len(hist))


def trimmed(hist,extra=1):
    last=max((i for i,n in enumerate(hist) if n),default=0)
    end=min(len(hist),last+extra+1)
    return np.arange(end),np.asarray(hist[:end])


def csv_write(path,rows):
    if not rows:return
    names=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8-sig') as out:
        writer=csv.DictWriter(out,fieldnames=names);writer.writeheader();writer.writerows(rows)


def savefig(fig,path,name):
    fig.savefig(path/f'{name}.png',dpi=155,bbox_inches='tight',facecolor='white')
    fig.savefig(path/f'{name}.svg',bbox_inches='tight',facecolor='white')
    plt.close(fig)


def representative(group):
    median=st.median(r['result']['simulations'] for r in group)
    return min(group,key=lambda r:(abs(r['result']['simulations']-median),r.get('replicate',0)))


def top5(r):return sorted(r['result']['visits'],key=lambda c:(-visits(c),-c['prior']))[:5]
def root_by_move(r,uci):return next(c for c in r['result']['telemetry']['roots'] if move(c)==uci)


def validation_footer(out):
    metadata_path=out/'experiment-metadata.json'
    if not metadata_path.exists():return ''
    metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
    integrity=metadata.get('integrity_after',metadata['integrity_before'])
    native=(out/'native-tests.log').read_text(encoding='utf-8',errors='replace') if (out/'native-tests.log').exists() else ''
    asan=(out/'asan-tests.log').read_text(encoding='utf-8',errors='replace') if (out/'asan-tests.log').exists() else ''
    return f'''\n## 원본 보존·엔진 검증 자료

실험 종료 해시 검사에서 기존 릴리스 {integrity['original_files_checked']}개 파일과 이전 계측 버전 {integrity['depth_files_checked']}개 파일의 변경은 각각 {len(integrity['original_changed'])}개, {len(integrity['depth_changed'])}개였다. 모델 SHA-256은 `{integrity['model_sha256']}`다.

- [Native 테스트 로그](native-tests.log): 통과한 CTest 기록 {native.count('Test Passed.')}개
- [AddressSanitizer 테스트 로그](asan-tests.log): 통과한 CTest 기록 {asan.count('Test Passed.')}개
- [UCI·포화 전 통계 동등성·포화 뒤 완료 N 검증](validation-uci.json)

이 검증은 규칙·메모리 접근·회계·실행 동작을 확인하며 수의 객관적 우수성이나 Elo를 측정하지 않는다.
'''


def read(path):
    rows=[];seen=set();audit={'runs':0,'recycled_runs':0,'prune_events':0,'all_invariants_passed':True}
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():continue
        r=json.loads(line);key=r['run_id'];assert key not in seen,key;seen.add(key)
        s=r['result'];t=s['telemetry'];c=t['counters'];n=s['simulations'];h=t['leaf_depth'];d=depths(h)
        assert sum(h)==n and sum(visits(x) for x in s['visits'])==n,(key,'completed N')
        assert d['max']==s['max_depth'] and d['median']==t['depth']['median_depth'],(key,'summary')
        assert d['mean'] is None or abs(d['mean']-t['depth']['mean_depth'])<1e-7,(key,'mean')
        assert all(a+b==z for a,b,z in zip(t['pre_failure_leaf_depth'],t['post_failure_leaf_depth'],h)),(key,'phase partition')
        tail=n
        for depth,count in enumerate(h):
            assert t['traversal_depth'][depth]==tail,(key,'traversal',depth)
            tail-=count
        rm={move(x):x for x in t['roots']}
        for x in s['visits']:assert sum(rm[move(x)]['leaf_depth'])==visits(x),(key,'root partition')
        event=t['first_allocation_failure']
        if event['occurred']:
            assert sum(t['pre_failure_leaf_depth'])==min(n,event['simulation']-1),(key,'pre failure boundary')
        else:assert sum(t['post_failure_leaf_depth'])==0,(key,'post without failure')
        assert c['neural_evaluations']==s['neural_evals'],(key,'NN count')
        assert s['memory_bytes']<=r['memory_mib']*MIB,(key,'arena budget')
        previous=np.zeros(len(h),dtype=np.uint64)
        for snap in t['snapshots']:
            current=np.asarray(snap['leaf_depth'],dtype=np.uint64)
            window=np.asarray(snap['window_leaf_depth'],dtype=np.uint64)
            assert np.array_equal(previous+window,current),(key,'snapshot partition')
            assert sum(visits(x) for x in snap['roots'])==sum(current),(key,'snapshot visits')
            previous=current
        assert np.array_equal(previous,h),(key,'final snapshot')
        if r['variant']=='recycled':
            audit['recycled_runs']+=1
            assert c['node_creations']+1-c['nodes_evicted']==s['arena_nodes'],(key,'live node accounting')
            assert s['attempted_simulations']==n+s['aborted_simulations'],(key,'attempt accounting')
            assert c['non_retained_neural_evaluations']==0,(key,'unretained NN eliminated')
            assert c['resident_unexpanded_repeat']==0,(key,'resident unexpanded NN repeats eliminated')
            assert c['discarded_neural_evaluations']==0,(key,'no unstored NN value computed then discarded')
            assert c['neural_evaluations']==c['legal_expansions'],(key,'each computed NN value has stored legal edges')
            assert s['peak_memory_bytes']<=r['memory_mib']*MIB,(key,'peak budget')
            assert s['prune_count']==len(t['prunes']),(key,'prune count')
            for p in t['prunes']:
                assert p['success'] and p['root_statistics_preserved'],(key,'prune success')
                assert p['roots_before']==p['roots_after'],(key,'root statistics changed during prune')
                assert p['nodes_after']<=p['nodes_before'] and p['edges_after']<=p['edges_before'],(key,'prune size')
                audit['prune_events']+=1
        else:assert c['node_creations']+1==s['arena_nodes'],(key,'legacy live nodes')
        r['_depth']=d;r['_pre']=depths(t['pre_failure_leaf_depth']);r['_post']=depths(t['post_failure_leaf_depth'])
        rows.append(r)
    audit['runs']=len(rows)
    return rows,audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--validate-only',action='store_true');args=p.parse_args()
    rows,audit=read(args.input)
    if args.validate_only:print(json.dumps(audit));return
    assert rows,'No experiment rows'
    out=args.out.resolve();out.mkdir(exist_ok=True,parents=True);graphs=out/'graphs';graphs.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'svg.fonttype':'none'})
    cases=[c for c in CASES if any(r['case']==c for r in rows)]
    groups=defaultdict(list)
    for r in rows:
        budget=r['budget_ms'] if r['mode']=='time' else r['budget_simulations']
        groups[(r['case'],r['mode'],r['group'],budget,r['memory_mib'],r['variant'])].append(r)
    summaries=[];metrics=[];candidates=[];leafrows=[];pathrows=[];phaserows=[];windows=[];prunes=[];roottrace=[]
    for key,group in sorted(groups.items()):
        case,mode,kind,budget,memory,variant=key
        hist=np.sum([r['result']['telemetry']['leaf_depth'] for r in group],axis=0).tolist();d=depths(hist)
        firsts=[r['result']['telemetry']['first_allocation_failure'] for r in group]
        summary={'case':case,'mode':mode,'group':kind,'budget':budget,'memory_mib':memory,'variant':variant,
            'repeats':len(group),'representative_run':representative(group)['run_id'],
            'N_median':st.median(r['result']['simulations'] for r in group),
            'N_min':min(r['result']['simulations'] for r in group),'N_max':max(r['result']['simulations'] for r in group),
            'search_ms_median':st.median(elapsed(r) for r in group),'leaf_mean_pooled':d['mean'],'leaf_median_pooled':d['median'],
            'leaf_max_median':st.median(r['_depth']['max'] for r in group),
            'max_min':min(r['_depth']['max'] for r in group),'max_max':max(r['_depth']['max'] for r in group),
            'new_nodes_median':st.median(cn(r,'node_creations') for r in group),
            'resident_nodes_median':st.median(r['result']['arena_nodes'] for r in group),
            'nonroot_legal_expansions_median':st.median(cn(r,'legal_expansions')-cn(r,'root_legal_expansions') for r in group),
            'prunes_median':st.median(r['result'].get('prune_count',0) for r in group),
            'prune_duration_ms_median':st.median(sum(p['duration_ms'] for p in r['result']['telemetry'].get('prunes',[])) for r in group),
            'aborted_median':st.median(r['result'].get('aborted_simulations',0) for r in group),
            'allocation_failures_median':st.median(sum(cn(r,k) for k in ['child_fail_node_limit','child_fail_byte_capacity','expansion_fail_byte_capacity']) for r in group),
            'unretained_nn_median':st.median(cn(r,'non_retained_neural_evaluations') for r in group),
            'resident_unexpanded_repeat_median':st.median(cn(r,'resident_unexpanded_repeat') for r in group),
            'unstored_or_repeat_nn_median':st.median(cn(r,'non_retained_neural_evaluations')+cn(r,'resident_unexpanded_repeat') for r in group),
            'failure_runs':sum(f['occurred'] for f in firsts),
            'first_failure_ms_median':st.median(f['elapsed_ms'] for f in firsts if f['occurred']) if any(f['occurred'] for f in firsts) else None,
            'moves':';'.join(r['result']['bestmove'] for r in group)}
        summaries.append(summary)
    for r in rows:
        s=r['result'];t=s['telemetry'];c=t['counters'];n=s['simulations'];selected=next(x for x in s['visits'] if move(x)==s['bestmove'])
        base={k:r.get(k) for k in ['run_id','case','variant','mode','group','budget_ms','budget_simulations','memory_mib','replicate']}
        rootmap={move(x):x for x in t['roots']};ds=depths(rootmap[s['bestmove']]['leaf_depth'])
        metrics.append({**base,'completed_N':n,'attempts':s.get('attempted_simulations',n),'search_ms':elapsed(r),
            'process_wall_ms':r['process_wall_ms'],'bestmove':s['bestmove'],'selected_visits':visits(selected),
            'selected_q':selected['q'],'selected_policy':selected['prior'],'selected_visit_fraction':visits(selected)/n if n else None,
            **{f'leaf_{k}':v for k,v in r['_depth'].items()},**{f'selected_leaf_{k}':v for k,v in ds.items()},
            'selected_depth_le_2_fraction':sum(rootmap[s['bestmove']]['leaf_depth'][:3])/visits(selected) if visits(selected) else None,
            'arena_bytes':s['memory_bytes'],'peak_arena_bytes':s.get('peak_memory_bytes',s['memory_bytes']),
            'peak_algorithm_scratch_bytes':s.get('peak_auxiliary_bytes',0),'telemetry_bytes':t.get('telemetry_memory_bytes',0),
            'resident_nodes':s['arena_nodes'],'memory_exhausted':s.get('memory_exhausted',False),
            'prune_duration_ms':sum(p['duration_ms'] for p in t.get('prunes',[])),
            'prune_duration_fraction':sum(p['duration_ms'] for p in t.get('prunes',[]))/elapsed(r),
            **{f'pre_failure_{k}':v for k,v in r['_pre'].items()},**{f'post_failure_{k}':v for k,v in r['_post'].items()},**c})
        for rank,x in enumerate(top5(r),1):
            d=depths(rootmap[move(x)]['leaf_depth'])
            candidates.append({**base,'rank':rank,'uci':move(x),'selected':move(x)==s['bestmove'],'policy':x['prior'],
                'q':x['q'],'visits':visits(x),'visit_fraction':visits(x)/n if n else None,**{f'leaf_{k}':v for k,v in d.items()}})
        for depth,count in enumerate(t['leaf_depth']):
            if count:leafrows.append({**base,'depth':depth,'count':count,'fraction':count/n if n else None})
        for depth,count in enumerate(t['traversal_depth']):
            if count:pathrows.append({**base,'depth':depth,'visits':count,'fraction':count/n if n else None})
        for phase in ['pre_failure','post_failure']:
            h=t[f'{phase}_leaf_depth'];total=sum(h)
            for depth,count in enumerate(h):
                if count:phaserows.append({**base,'phase':phase,'depth':depth,'count':count,'fraction':count/total})
        for snap in t['snapshots']:
            b={**base,'snapshot_N':snap['simulation'],'snapshot_ms':snap['elapsed_ms'],'reason':snap['reason']}
            windows.append({**b,**snap['window_depth'],'resident_nodes':snap['nodes_size'],
                'cumulative_node_creations':snap['counters']['node_creations'],
                'prune_count':snap['counters'].get('prune_count',0),'aborted':snap['counters'].get('aborted_simulations',0),
                'non_retained_nn':snap['counters']['non_retained_neural_evaluations']})
            for x in snap['roots']:roottrace.append({**b,'uci':move(x),'prior':x['prior'],'q':x['q'],'visits':visits(x)})
        for index,event in enumerate(t.get('prunes',[]),1):
            prunes.append({**base,'prune_index':index,**{k:v for k,v in event.items() if k not in ['roots_before','roots_after']}})
    for name,data in [('comparison-summary.csv',summaries),('all-run-metrics.csv',metrics),('top5-candidates.csv',candidates),
        ('leaf-depth-histograms.csv',leafrows),('depth-path-visits.csv',pathrows),('pre-post-failure-histograms.csv',phaserows),
        ('depth-windows.csv',windows),('prune-events.csv',prunes),('root-candidate-snapshots.csv',roottrace)]:csv_write(out/name,data)
    time_s=[s for s in summaries if s['mode']=='time'];fixed_s=[s for s in summaries if s['mode']!='time']
    csv_write(out/'time-comparison.csv',time_s);csv_write(out/'fixed-simulation-comparison.csv',fixed_s)
    long={}
    for case in cases:
        for memory in [64,512]:
            for variant in VARS:
                g=[r for r in rows if r['case']==case and r['mode']=='time' and r['budget_ms']==10000 and r['memory_mib']==memory and r['variant']==variant]
                if g:long[(case,memory,variant)]=representative(g)

    # A compact figure suitable for the chat, with the full matrix in separate figures.
    fig,ax=plt.subplots(1,3,figsize=(14,4.5))
    for memory in [64,512]:
        for variant in VARS:
            part=sorted([s for s in time_s if s['case']=='start' and s['memory_mib']==memory and s['variant']==variant],key=lambda s:s['budget'])
            if part:ax[0].plot([s['budget'] for s in part],[s['leaf_mean_pooled'] for s in part],'-o' if variant=='recycled' else '--s',color=MEMCOL[memory],label=f'{variant}, {memory} MiB')
    ax[0].set_xscale('log');ax[0].set_xticks([100,1000,10000],['100ms','1s','10s']);ax[0].set_ylabel('Mean endpoint depth (plies)');ax[0].set_title('Start: all completed simulations');ax[0].legend(fontsize=8)
    for j,case in enumerate(['start','before_Ra6_10s'],1):
        for variant in VARS:
            r=long.get((case,64,variant))
            if not r:continue
            root=root_by_move(r,r['result']['bestmove']);h=root['leaf_depth'];d=depths(h);x,y=trimmed(norm(h))
            ax[j].step(x,y*100,where='mid',lw=2,color=COL[variant],label=f"{variant}: {SAN.get(move(root),move(root))}\nQ={root['q']:+.3f}, Visits={visits(root):,}\nmean={d['mean']:.2f}, max={d['max']}")
        ax[j].set_title(f'{EN[case]}: selected branch, 10s / 64 MiB');ax[j].set_xlabel('Endpoint depth (plies)');ax[j].set_ylabel('Conditional endpoint probability (%)');ax[j].legend(fontsize=7)
    fig.suptitle('Legacy vs memory recycling: same model, seed and search thread; time mode',fontsize=13);fig.tight_layout();savefig(fig,graphs,'overview')

    fig,axes=plt.subplots(len(cases),3,figsize=(14,3*len(cases)),squeeze=False)
    for i,case in enumerate(cases):
        for memory in [64,512]:
            for variant in VARS:
                part=sorted([s for s in time_s if s['case']==case and s['memory_mib']==memory and s['variant']==variant],key=lambda s:s['budget'])
                for j,(field,title) in enumerate([('leaf_mean_pooled','Mean'),('leaf_median_pooled','Median'),('leaf_max_median','Max')]):
                    axes[i,j].plot([s['budget'] for s in part],[s[field] for s in part],'-o' if variant=='recycled' else '--s',color=MEMCOL[memory],label=f'{variant} {memory} MiB')
                    axes[i,j].set_xscale('log');axes[i,j].set_xticks([100,1000,10000],['100ms','1s','10s']);axes[i,j].set_title(f'{EN[case]}\n{title} endpoint depth');axes[i,j].set_ylabel('Plies')
    axes[0,0].legend(fontsize=7);fig.suptitle('Time budget comparison: pooled mean/median; median run maximum',fontsize=13);fig.tight_layout();savefig(fig,graphs,'time-depth-comparison')

    fig,axes=plt.subplots(len(cases),3,figsize=(14,3*len(cases)),squeeze=False)
    for i,case in enumerate(cases):
        for memory in [64,512]:
            for variant in VARS:
                part=sorted([s for s in time_s if s['case']==case and s['memory_mib']==memory and s['variant']==variant],key=lambda s:s['budget'])
                for j,(field,title) in enumerate([('N_median','Completed simulations'),('new_nodes_median','Cumulative node creation'),('unstored_or_repeat_nn_median','Unretained + resident-repeat NN')]):
                    axes[i,j].plot([s['budget'] for s in part],[s[field] for s in part],'-o' if variant=='recycled' else '--s',color=MEMCOL[memory],label=f'{variant} {memory} MiB')
                    axes[i,j].set_xscale('log');axes[i,j].set_xticks([100,1000,10000],['100ms','1s','10s']);axes[i,j].set_title(f'{EN[case]}\n{title}');axes[i,j].set_ylabel('Count')
    axes[0,0].legend(fontsize=7);fig.suptitle('Time budget work: completed backups are separate from failed attempts',fontsize=13);fig.tight_layout();savefig(fig,graphs,'time-work-comparison')

    fig,axes=plt.subplots(len(cases),2,figsize=(12.5,3*len(cases)),squeeze=False)
    for i,case in enumerate(cases):
        for j,memory in enumerate([64,512]):
            for variant in VARS:
                part=sorted([s for s in fixed_s if s['case']==case and s['memory_mib']==memory and s['variant']==variant],key=lambda s:s['budget'])
                axes[i,j].plot([s['budget'] for s in part],[s['leaf_mean_pooled'] for s in part],'-o',color=COL[variant],label=f'{variant}: mean')
                axes[i,j].plot([s['budget'] for s in part],[s['leaf_median_pooled'] for s in part],'--',color=COL[variant],label=f'{variant}: median')
            axes[i,j].set_xscale('log');axes[i,j].set_title(f'{EN[case]} / {memory} MiB');axes[i,j].set_xlabel('Requested and completed N');axes[i,j].set_ylabel('Plies');axes[i,j].legend(fontsize=7)
    fig.suptitle('Fixed completed-N controls; memory effects separated from time throughput',fontsize=13);fig.tight_layout();savefig(fig,graphs,'fixed-N-depth-comparison')

    fig,axes=plt.subplots(2,2,figsize=(12.5,7.2),squeeze=False)
    for ax,case in zip(axes.flat,cases):
        for memory in [64,512]:
            paired=defaultdict(dict)
            for s in fixed_s:
                if s['case']==case and s['memory_mib']==memory:paired[(s['group'],s['budget'])][s['variant']]=s
            complete=sorted([(key,pair) for key,pair in paired.items() if all(v in pair for v in VARS)],key=lambda x:x[0][1])
            ax.plot([key[1] for key,pair in complete],[pair['recycled']['search_ms_median']/pair['legacy']['search_ms_median'] for key,pair in complete],'-o',color=MEMCOL[memory],label=f'{memory} MiB')
        ax.axhline(1,color='#64748b',ls='--',lw=1);ax.set_xscale('log');ax.set_title(EN[case]);ax.set_xlabel('Identical completed N');ax.set_ylabel('Recycled / legacy search time');ax.legend(fontsize=8)
    fig.suptitle('Fixed-N search cost: one run per condition; model load and JSON serialization excluded',fontsize=13);fig.tight_layout();savefig(fig,graphs,'fixed-N-search-cost')

    for name,phase in [('leaf-distribution-10s',None),('post-failure-depth-10s','post_failure')]:
        fig,axes=plt.subplots(len(cases),2,figsize=(12.5,3*len(cases)),squeeze=False)
        for i,case in enumerate(cases):
            for j,memory in enumerate([64,512]):
                for variant in VARS:
                    r=long.get((case,memory,variant))
                    if not r:continue
                    h=r['result']['telemetry'][f'{phase}_leaf_depth' if phase else 'leaf_depth']
                    if not sum(h):continue
                    x,y=trimmed(norm(h));d=depths(h)
                    axes[i,j].step(x,y*100,where='mid',color=COL[variant],lw=2,label=f"{variant}: mean {d['mean']:.2f}, median {d['median']:g}, max {d['max']}")
                axes[i,j].set_title(f'{EN[case]} / {memory} MiB');axes[i,j].set_xlabel('Endpoint depth (plies)');axes[i,j].set_ylabel('Endpoint probability (%)');axes[i,j].legend(fontsize=7)
        fig.suptitle('10s representative runs: '+('completed endpoints after first real allocation failure' if phase else 'all completed endpoint depths'),fontsize=13);fig.tight_layout();savefig(fig,graphs,name)

    fig,axes=plt.subplots(len(cases),2,figsize=(12.5,3*len(cases)),squeeze=False)
    memoryfig,memaxes=plt.subplots(len(cases),2,figsize=(12.5,3*len(cases)),squeeze=False)
    for i,case in enumerate(cases):
        for j,memory in enumerate([64,512]):
            for variant in VARS:
                r=long.get((case,memory,variant))
                if not r:continue
                t=r['result']['telemetry'];snaps=t['snapshots'];nonempty=[s for s in snaps if s['window_depth']['samples']]
                axes[i,j].plot([s['elapsed_ms']/1000 for s in nonempty],[s['window_depth']['mean_depth'] for s in nonempty],color=COL[variant],label=variant)
                event=t['first_allocation_failure']
                if event['occurred']:axes[i,j].axvline(event['elapsed_ms']/1000,color=COL[variant],ls='--',alpha=.6)
                for prune in t.get('prunes',[]):axes[i,j].axvline(prune['elapsed_ms']/1000,color=COL[variant],lw=.5,alpha=.12)
                memaxes[i,j].plot([s['elapsed_ms']/1000 for s in snaps],[s['nodes_size'] for s in snaps],color=COL[variant],label=f'{variant}: resident')
                memaxes[i,j].plot([s['elapsed_ms']/1000 for s in snaps],[s['counters']['node_creations'] for s in snaps],color=COL[variant],ls='--',label=f'{variant}: cumulative creations')
            axes[i,j].set_title(f'{EN[case]} / {memory} MiB');axes[i,j].set_xlabel('Search elapsed seconds');axes[i,j].set_ylabel('Nonempty-window mean depth');axes[i,j].legend(fontsize=7)
            memaxes[i,j].set_title(f'{EN[case]} / {memory} MiB');memaxes[i,j].set_xlabel('Search elapsed seconds');memaxes[i,j].set_ylabel('Node count');memaxes[i,j].legend(fontsize=7)
    fig.suptitle('10s depth timeline: no-data prune windows omitted; dashed line = first failure',fontsize=13);fig.tight_layout();savefig(fig,graphs,'depth-recycling-timeline')
    memoryfig.suptitle('10s memory reuse: current resident nodes versus cumulative node creations',fontsize=13);memoryfig.tight_layout();savefig(memoryfig,graphs,'resident-vs-cumulative-nodes')

    # Root-candidate traces show each version/cap in its own panel; no mixing of Q units.
    for case in cases:
        fig,axes=plt.subplots(3,4,figsize=(15,9),squeeze=False)
        for column,(memory,variant) in enumerate([(64,'legacy'),(64,'recycled'),(512,'legacy'),(512,'recycled')]):
            r=long.get((case,memory,variant))
            if not r:continue
            t=r['result']['telemetry'];chosen=r['result']['bestmove'];wanted=list(dict.fromkeys([move(x) for x in top5(r)]+[TARGET[case]]))
            for index,uci in enumerate(wanted):
                color=plt.get_cmap('tab10')(index%10)
                trace=[(s,next(x for x in s['roots'] if move(x)==uci)) for s in t['snapshots'] if s['simulation']>0]
                axes[0,column].plot([s['simulation'] for s,x in trace],[visits(x)/s['simulation']*100 for s,x in trace],color=color,label=SAN.get(uci,uci),lw=1.4)
                axes[1,column].plot([s['simulation'] for s,x in trace],[x['q'] for s,x in trace],color=color,label=SAN.get(uci,uci),lw=1.4)
            for row in [0,1]:
                axes[row,column].set_xscale('log');axes[row,column].set_xlabel('Completed N')
                event=t['first_allocation_failure']
                if event['occurred']:axes[row,column].axvline(event['simulation'],color='#dc2626',ls='--',alpha=.5)
            allh=t['leaf_depth'];h=root_by_move(r,chosen)['leaf_depth'];x,y=trimmed(norm(allh));axes[2,column].step(x,y*100,where='mid',color='#94a3b8',label='all endpoints')
            x,y=trimmed(norm(h));axes[2,column].step(x,y*100,where='mid',color='#dc2626',label=f'selected {SAN.get(chosen,chosen)}')
            axes[0,column].set_title(f'{variant} / {memory} MiB');axes[0,column].legend(fontsize=6)
            axes[2,column].set_xlabel('Endpoint depth (plies)');axes[2,column].legend(fontsize=6)
        axes[0,0].set_ylabel('Root visit share (%)');axes[1,0].set_ylabel('Root Q (W-L)');axes[2,0].set_ylabel('Endpoint probability (%)')
        fig.suptitle(f'{EN[case]}: 10s candidate Q, visits, and selected-branch depth',fontsize=13);fig.tight_layout();savefig(fig,graphs,f'candidate-diagnostics-{case}')

    # Tables and narrative use the exact historical N for interest moves.
    def compare_table(data,time_mode):
        lines=['| 위치 | '+('시간' if time_mode else '고정 N')+' | MiB | 버전 | 완료 N | 탐색 ms | 잎 평균/중앙값/최대 | 생성/확장/현재 노드 | 실패/회수 | 미보존/미확장 반복 NN | 선택 |',
               '|---|---:|---:|---|---:|---:|---|---:|---:|---:|---|']
        order={case:i for i,case in enumerate(cases)}
        for s in sorted(data,key=lambda s:(order[s['case']],s['budget'],s['memory_mib'],s['variant'],s['group'])):
            budget=f"{s['budget']/1000:g}s" if time_mode else f"{s['budget']:,}"
            names=' / '.join(SAN.get(x,x) for x in s['moves'].split(';'))
            lines.append(f"| {LABEL[s['case']]} | {budget} | {s['memory_mib']} | {VN[s['variant']]} | {s['N_median']:,.0f} | {fmt(s['search_ms_median'])} | {fmt(s['leaf_mean_pooled'])}/{fmt(s['leaf_median_pooled'],1)}/{s['leaf_max_median']:g} | {s['new_nodes_median']:,.0f}/{s['nonroot_legal_expansions_median']:,.0f}/{s['resident_nodes_median']:,.0f} | {s['allocation_failures_median']:,.0f}/{s['prunes_median']:g} | {s['unretained_nn_median']:,.0f}/{s['resident_unexpanded_repeat_median']:,.0f} | {names} |")
        return '\n'.join(lines)
    anchors=[r for r in rows if r['group']=='historical_anchor']
    interest=[];narrative=[];target_metrics=[]
    for case in cases:
        narrative.append(f'### {LABEL[case]}')
        for memory in [64,512]:
            for variant in VARS:
                match=[r for r in anchors if r['case']==case and r['memory_mib']==memory and r['variant']==variant]
                if not match:continue
                r=match[0];s=r['result'];t=s['telemetry'];target=root_by_move(r,TARGET[case]);selected=root_by_move(r,s['bestmove']);td=depths(target['leaf_depth']);sd=depths(selected['leaf_depth'])
                event=t['first_allocation_failure'];n=s['simulations'];h=target['leaf_depth']
                target_metrics.append({'case':case,'memory_mib':memory,'variant':variant,'run_id':r['run_id'],'completed_N':n,
                    'target':TARGET[case],'target_policy':target['prior'],'target_q':target['q'],'target_visits':visits(target),
                    **{f'target_leaf_{k}':v for k,v in td.items()},'selected':s['bestmove'],
                    **{f'selected_leaf_{k}':v for k,v in sd.items()},'target_le2_fraction':sum(h[:3])/sum(h) if sum(h) else None})
                narrative.append(f"- {VN[variant]} {memory}MiB, N={n:,}: 선택 `{SAN.get(s['bestmove'],s['bestmove'])}`, 관심 수 `{SAN.get(TARGET[case],TARGET[case])}` Policy={target['prior']*100:.5f}%, Q={target['q']:+.5f}, Visits={visits(target):,}. 관심 수의 잎 깊이 평균/중앙값/최대={fmt(td['mean'])}/{fmt(td['median'],1)}/{td['max']}, 깊이 2 이하 비중={fmt(sum(h[:3])/sum(h)*100 if sum(h) else None,4)}%다. 완료된 전체 분포는 {r['_depth']['mean']:.2f}/{r['_depth']['median']:g}/{r['_depth']['max']}다.")
                narrative.append(f"  실제 첫 할당 실패 "+(f"N={event['simulation']:,}, {event['elapsed_ms']:.2f}ms; " if event['occurred'] else '없음; ')+f"회수 {s.get('prune_count',0):,}회, 폐기 attempt {s.get('aborted_simulations',0):,}회, 미보존 NN={cn(r,'non_retained_neural_evaluations'):,}, 저장된 미확장 노드 재평가={cn(r,'resident_unexpanded_repeat'):,}." )
        narrative.append(f'![후보 Q·방문·깊이](graphs/candidate-diagnostics-{case}.png)')
    csv_write(out/'interest-move-comparison.csv',target_metrics)
    recycle_table=['| 위치 | MiB | 완료 N | 최초 실패 ms | 회수/폐기 attempt | 누적 회수 ms | 탐색 시간 중 회수 비중 | algorithm scratch MiB |',
                   '|---|---:|---:|---:|---:|---:|---:|---:|']
    for case in cases:
        for memory in [64,512]:
            r=long.get((case,memory,'recycled'))
            if not r:continue
            s=r['result'];t=s['telemetry'];event=t['first_allocation_failure'];duration=sum(p['duration_ms'] for p in t.get('prunes',[]))
            recycle_table.append(f"| {LABEL[case]} | {memory} | {s['simulations']:,} | {fmt(event['elapsed_ms'] if event['occurred'] else None)} | {s.get('prune_count',0)}/{s.get('aborted_simulations',0)} | {duration:.2f} | {duration/elapsed(r)*100:.3f}% | {s.get('peak_auxiliary_bytes',0)/MIB:.3f} |")
    top_tables=[]
    for case in cases:
        top_tables.append(f'### {LABEL[case]} · 10초 대표 실행')
        top_tables.append('| 버전 | MiB | 순위 | 수 | Policy | Q | Visits | 조건부 잎 평균/중앙값/최대 |\n|---|---:|---:|---|---:|---:|---:|---|')
        for memory in [64,512]:
            for variant in VARS:
                r=long.get((case,memory,variant))
                if not r:continue
                for rank,x in enumerate(top5(r),1):
                    d=depths(root_by_move(r,move(x))['leaf_depth'])
                    top_tables.append(f"| {VN[variant]} | {memory} | {rank} | `{SAN.get(move(x),move(x))}` {'★' if move(x)==r['result']['bestmove'] else ''} | {x['prior']*100:.5f}% | {x['q']:+.5f} | {visits(x):,} | {fmt(d['mean'])}/{fmt(d['median'],1)}/{d['max']} |")
    recycled=[r for r in rows if r['variant']=='recycled']
    fixed=[r for r in rows if r['mode']!='time']
    all_requested_complete=all(r['result']['simulations']==r['budget_simulations'] for r in fixed)
    recycled_unretained=sum(cn(r,'non_retained_neural_evaluations') for r in recycled)
    before_unretained=sum(cn(r,'non_retained_neural_evaluations') for r in rows if r['variant']=='legacy')
    recycled_repeats=sum(cn(r,'resident_unexpanded_repeat') for r in recycled)
    before_repeats=sum(cn(r,'resident_unexpanded_repeat') for r in rows if r['variant']=='legacy')
    maxscratch=max((r['result'].get('peak_auxiliary_bytes',0) for r in recycled),default=0)
    summary={'runs':len(rows),'time_runs':sum(r['mode']=='time' for r in rows),'fixed_runs':len(fixed),
        'variants':{v:sum(r['variant']==v for r in rows) for v in VARS},'invariants':audit,
        'fixed_requested_N_all_completed':all_requested_complete,'recycled_non_retained_NN_total':recycled_unretained,
        'legacy_non_retained_NN_total':before_unretained,'recycled_resident_repeat_NN_total':recycled_repeats,
        'legacy_resident_repeat_NN_total':before_repeats,'algorithm_scratch_peak_bytes':maxscratch,
        'comparisons':summaries,'representative_10s_runs':{f'{c}/{m}/{v}':r['run_id'] for (c,m,v),r in long.items()}}
    (out/'analysis-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    report=f'''# CHROMATIC 메모리 포화 탐색 수정 비교

동일 모델로 기존 탐색과 메모리 회수 탐색을 다시 비교했다. 분석한 실행은 **{len(rows)}회**이며 시간 기준 {summary['time_runs']}회, 동일 완료 N 기준 {len(fixed)}회다. 각 시간 조건은 3회 반복한다. 표의 N·노드 수·실패·회수 횟수는 반복 중앙값, 잎 평균·중앙값은 반복 전체 잎을 합친 분포, 최대 깊이는 실행별 최대의 중앙값이다. 그래프의 10초 대표 실행은 해당 조건에서 N 중앙값에 가장 가까운 실제 한 실행이다.

표의 생성/확장/현재 노드는 각각 누적 노드 생성(루트 제외·재생성 포함), 성공한 합법 수 목록 확장(루트 제외), 최종 보존 노드(루트 포함)다.

![요약](graphs/overview.png)

## 변경한 동작

기존 탐색은 노드 할당이 실패해도 임시 보드의 얕은 잎을 NN으로 평가하여 Q·Visits에 계속 백업했다. 수정본은 **공간을 확보하지 못한 attempt를 NN/backup 없이 폐기하고, 저방문 하위 가지를 정리·압축하여 기존 예약 슬롯을 재사용한 뒤 루트부터 다시 시도한다.** 루트·전체 루트 간선의 Policy/Q/Visits와 보존된 내부 통계는 유지한다. 일반적인 예산에서는 이미 생성한 루트 직계 자식도 보호한다. 제거된 하위 트리의 통계는 잃으므로 무제한 트리와 동일한 탐색이라고 부르지 않는다.

새로운 두 번째 트리 전체를 복사하지 않고 기존 Node/Edge 버퍼 안에서 압축한다. 바이트 예산과 `peak_memory_bytes`는 Node/Edge 예약 capacity 합계 기준이다. 초기 `vector::reserve` 재할당 중 일시적으로 공존하는 이전/새 할당이나 전체 프로세스 최대 메모리를 측정한 값은 아니다. 매핑·정렬용 임시 공간은 별도로 보고하며 이번 실행의 최대 algorithm scratch는 **{maxscratch/MIB:.3f}MiB**다. 선택 계측 자료와 출력 JSON 버퍼도 트리 예산 외의 메모리다. 전체 프로세스 메모리가 64MiB/512MiB라는 뜻은 아니다.

## 회계와 검증 결과

- 완료 N은 실제 백업을 마친 시뮬레이션만 센다. 실패 attempt는 Visits·잎 분포에 포함하지 않는다. 수정본의 `attempted = completed + aborted`와 `cumulative node creations + 1 − evicted nodes = resident nodes`를 검사했다.
- 동일 N 대조에서 요청한 N을 모두 완료했는가: **{all_requested_complete}**.
- 기존 버전의 미보존 잎 NN 호출 합계: **{before_unretained:,}**. 수정본: **{recycled_unretained:,}**.
- 기존 버전의 저장된 미확장 노드 NN 재평가 합계: **{before_repeats:,}**. 수정본: **{recycled_repeats:,}**. 노드 공간은 있으나 간선 목록을 저장하지 못한 경우에도 반복 평가가 생기므로 두 지표를 함께 본다. 이 두 값이 0이라는 것은 모든 동일 입력 NN 호출이 사라졌다는 뜻은 아니다. 정리된 하위 트리를 나중에 재생성하여 같은 체스 위치를 다시 평가할 수 있다.
- 회수 이벤트 {audit['prune_events']:,}개에서 root before/after Policy·Q·Visits가 같고 메모리 상한을 지키는지 검사했다.
- 잎 합=N, 루트 수별 잎 합=그 수 Visits, 깊이별 경로 방문=잎 분포 꼬리합, 전후 분포와 스냅샷 window 분할, 정확한 평균·중앙값·최대를 재계산했다.

모델·시드 2026·스레드 1·CPU affinity 2·SIMD/sparse·CPuct 1.5·FPU 0·noise/temperature/proof 0을 동일하게 사용했다. MaxNodes도 두 메모리와 두 버전에서 1,000,000으로 맞춰 바이트 상한 효과를 비교했다. 기존 엔진·모델·앞선 계측 버전의 보존 해시는 [실험 메타데이터](experiment-metadata.json)에 있다.

## 시간 기준 비교

{compare_table(time_s,True)}

![시간별 깊이](graphs/time-depth-comparison.png)

![완료 방문과 실제 확장량](graphs/time-work-comparison.png)

시간 모드의 N은 다르다. 기존 버전의 포화 뒤 얕은 미보존/미확장 반복 평가는 더 싸게 수행될 수 있으므로 단순 N/s 증가를 유효한 수읽기 증가로 해석하지 않는다. 수정본은 압축과 더 깊은 경로 탐색·실제 새 가지 확장의 비용을 부담한다. 서로 같은 작업량의 처리속도 비교가 아니므로 같은 시간의 방문수와 동일 N의 소요시간을 구분한다. 누적 회수 시간은 이벤트별 `duration_ms`의 합으로 따로 보존했으며 모든 소요시간 차이를 회수 비용으로 돌리지 않는다.

## 동일 완료 방문수 비교

고정 예산은 이전 계측의 64MiB 시간별 중앙 N 및 과거 관심 수가 나온 정확한 N을 사용한다. 두 버전·두 메모리에 동일하게 요청했다. 표에서 가까운 N이 두 번 나타나면 서로 다른 원자료 그룹이며 원시 로그의 `group`/`budget_label`로 구분한다.

{compare_table(fixed_s,False)}

![동일 N 깊이](graphs/fixed-N-depth-comparison.png)

![동일 N의 탐색 소요시간 비율](graphs/fixed-N-search-cost.png)

고정 N 소요시간은 조건당 1회이므로 실행 잡음을 포함한다. 시간비는 해당 동일 N 쌍의 탐색 소요시간을 나눈 값으로, 전체 하드웨어나 모든 위치의 일정한 성능 비용으로 일반화하지 않는다. 탐색 시간은 모델 로드·워밍업·JSON 직렬화를 제외하고 회수·정리 비용을 포함한다. 프로세스 전체 시간은 원시 지표에 별도 보존했다.

## 포화 뒤 깊이와 메모리 재사용

첫 포화 경계는 실제 할당 실패다. 수정본에서 그 attempt는 완료 N에 포함되지 않으므로 첫 실패의 `simulation`이 완료 N+1일 수 있다. 재시도 전에 시간이 끝나면 post-failure 분포가 비어 있을 수도 있다. 회수 스냅샷은 완료 N이 증가하지 않으며 잎 표본이 없는 window를 깊이 0으로 그리지 않고 제외했다.

아래는 수정본 10초 대표 실행의 회수 비용이다.

{chr(10).join(recycle_table)}

![전체 잎 깊이 분포](graphs/leaf-distribution-10s.png)

![첫 실패 뒤 완료된 잎 깊이](graphs/post-failure-depth-10s.png)

![깊이 시간 창과 회수 시점](graphs/depth-recycling-timeline.png)

![현재 보존 노드와 누적 새 노드](graphs/resident-vs-cumulative-nodes.png)

`node_creations`는 **누적 실제 노드 생성(재생성 포함)**이다. 슬롯이 재사용될 때 새로 보존한 노드도 다시 센다. 간선 공간 부족이 뒤이어 발생하면 노드는 생성됐어도 해당 attempt의 백업은 없을 수 있다. 실제 합법 수 목록 확장 성공은 `legal_expansions`로 따로 센다. 서로 다른 체스 포지션의 총 개수가 아니다. `non_retained_neural_evaluations`도 기존 버전에서 정확히 같은 NN 입력이 몇 번 반복되었는지를 뜻하지 않는다. `resident_unexpanded_repeat`는 이미 NN을 호출했지만 간선 목록을 보존하지 못한 상주 노드를 다시 평가한 횟수다. 수정본은 두 종류 모두 막지만, 퇴거된 위치의 정상적인 재생성·재평가는 허용한다.

## g4·Rg1·Ra6 선택 분석

다음 설명은 과거 N을 고정한 대조를 우선 사용한다. 전체 최대 깊이와 그 수의 조건부 잎 분포를 구분한다. Q는 모델 W−L의 백업 평균이며 객관적 승률이나 센티폰이 아니다. 수정본에서 관심 수가 다시 선택되거나 더 나빠 보이는 수를 선택해도 원자료 그대로 기록한다.

{chr(10).join(narrative)}

## 상위 5개 후보

각 실행의 전체 후보 표는 [top5-candidates.csv](top5-candidates.csv)에 있다. 아래는 10초 대표 실행의 상위 5개이며 ★가 최종 선택이다.

{chr(10).join(top_tables)}

## 해석 범위

이 수정은 포화 뒤 저장하지 못한 얕은 평가가 끝없이 백업되는 동작을 제거하고, 같은 메모리 예약 안에서 실제 저장하며 새 확장을 이어가는 동작을 검증한다. 모델 가중치·학습 데이터·정책·가치를 변경하지 않았다. 기존 버전과의 통계 동등성은 포화 전 구간에 대해 검사하며 포화 뒤 결과 변화는 의도한 변경이다. 제거된 하위 통계를 잃는 근사 탐색이므로 수 선택이 바뀌거나 깊이 통계가 좋아진 것만으로 Elo 향상을 주장하지 않는다. 평균 깊이 증가 자체도 탐색의 폭·모델 정확도를 대체하지 않는다. **수정본 64MiB의 평균 깊이가 512MiB보다 높더라도 64MiB가 더 강하다는 증거는 아니다.** 더 자주 낮은 방문 후손을 제거하며 탐색 분포가 집중되는 서로 다른 경로를 만들 수 있다. 기존 가치 함수의 일반화·일관성 문제는 남아 있다. 객관적인 강도 평가는 별도 대국이 필요하다.

## 원자료

- [전체 실행 지표](all-run-metrics.csv), [시간 비교](time-comparison.csv), [동일 N 비교](fixed-simulation-comparison.csv), [요약 JSON](analysis-summary.json)
- [관심 수 비교](interest-move-comparison.csv), [상위 5개 후보](top5-candidates.csv), [후보 스냅샷](root-candidate-snapshots.csv)
- [잎 깊이 빈도](leaf-depth-histograms.csv), [깊이별 경로 방문](depth-path-visits.csv), [포화 전후 빈도](pre-post-failure-histograms.csv), [빈 window를 포함한 원시 시간 창](depth-windows.csv)
- [회수 이벤트](prune-events.csv), [실험 JSONL]({args.input.name}), [조건·해시](experiment-metadata.json), 개별 명령과 stdout은 `raw/`
- 모든 그래프는 `graphs/`에 PNG와 SVG로 저장했다. 완성 전 부분 로그를 분석했다면 실행 수가 208보다 작을 수 있으며 보고서의 실제 실행 수를 기준으로 한다.
'''
    report+=validation_footer(out)
    (out/'MEMORY_REPORT.md').write_text(report,encoding='utf-8')
    print(json.dumps({'runs':len(rows),'time':summary['time_runs'],'fixed':len(fixed),'prunes':audit['prune_events'],
        'unretained_nn_after':recycled_unretained,'graphs':len(list(graphs.glob('*.png'))),'report':str(out/'MEMORY_REPORT.md')},ensure_ascii=False))


if __name__=='__main__':main()
