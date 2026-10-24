"""Generate auditable tables and static figures from optional MCTS telemetry.

Only reads experiment JSONL and writes the requested analysis directory. No
engine/model modifications and no chess searches are performed here.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


CASES = ['start', 'before_Rg1_100ms', 'before_Rg1_1s', 'before_Ra6_10s']
LABEL = {'start': '1.g4 직전 · 시작 위치', 'before_Rg1_100ms': '5.Rg1 직전 · 100ms 기보',
         'before_Rg1_1s': '4.Rg1 직전 · 1초 기보', 'before_Ra6_10s': '2...Ra6 직전 · 10초 기보'}
ENGLISH = {'start': 'Start (before 1.g4)', 'before_Rg1_100ms': 'Before 5.Rg1 (100ms game)',
           'before_Rg1_1s': 'Before 4.Rg1 (1s game)', 'before_Ra6_10s': 'Before 2...Ra6 (10s game)'}
TARGET = {'start': 'g2g4', 'before_Rg1_100ms': 'h1g1', 'before_Rg1_1s': 'h1g1', 'before_Ra6_10s': 'a8a6'}
SAN = {'g2g4': 'g4', 'h1g1': 'Rg1', 'a8a6': 'Ra6', 'c2c3': 'c3', 'b1c3': 'Nc3',
       'g1f3': 'Nf3', 'g8f6': 'Nf6', 'e2e4': 'e4', 'e7e6': 'e6'}
COLORS = {64: '#2563eb', 512: '#e76f21'}


def move(row):
    value = row.get('uci', row.get('move', ''))
    return value.get('uci', '') if isinstance(value, dict) else str(value)


def visits(row):
    return int(row.get('visits', row.get('n', 0)))


def depth_summary(hist):
    n = int(sum(hist))
    if not n:
        return {'samples': 0, 'mean': None, 'median': None, 'max': None,
                'p90': None, 'p95': None, 'p99': None}
    def rank(k):
        acc = 0
        for d, count in enumerate(hist):
            acc += count
            if acc >= k:
                return d
        raise AssertionError('histogram rank overflow')
    median = (rank((n+1)//2) + rank(n//2+1)) / 2
    return {'samples': n, 'mean': sum(d*c for d,c in enumerate(hist))/n,
            'median': median, 'max': max(d for d,c in enumerate(hist) if c),
            'p90': rank(math.ceil(.90*n)), 'p95': rank(math.ceil(.95*n)), 'p99': rank(math.ceil(.99*n))}


def read_runs(path):
    rows = []
    seen = set()
    for lineno, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        assert row['run_id'] not in seen, f'duplicate run_id line {lineno}'
        seen.add(row['run_id'])
        result = row['result']
        telemetry = result.get('telemetry')
        if not telemetry:
            continue  # Optional off controls belong in separate overhead reports.
        hist = telemetry['leaf_depth']
        n = result['simulations']
        summary = depth_summary(hist)
        assert summary['samples'] == n, (row['run_id'], 'one endpoint per simulation')
        assert summary['max'] == result['max_depth'], (row['run_id'], 'max depth')
        assert abs(summary['mean']-telemetry['depth']['mean_depth']) < 1e-6
        assert summary['median'] == telemetry['depth']['median_depth']
        assert telemetry['traversal_depth'][0] == n
        tail = n
        for d, c in enumerate(hist):
            assert telemetry['traversal_depth'][d] == tail, (row['run_id'], 'tail sum', d)
            tail -= c
        assert all(a+b == c for a,b,c in zip(telemetry['pre_failure_leaf_depth'],
                    telemetry['post_failure_leaf_depth'], hist))
        by_root = {move(item): item for item in telemetry['roots']}
        for item in result['visits']:
            assert sum(by_root[move(item)]['leaf_depth']) == visits(item)
        event = telemetry['first_allocation_failure']
        if event['occurred']:
            assert sum(telemetry['pre_failure_leaf_depth']) == event['simulation']-1
        else:
            assert sum(telemetry['post_failure_leaf_depth']) == 0
        row['_depth'] = summary
        row['_pre'] = depth_summary(telemetry['pre_failure_leaf_depth'])
        row['_post'] = depth_summary(telemetry['post_failure_leaf_depth'])
        rows.append(row)
    return rows


def elapsed(row):
    r = row['result']
    return r.get('elapsed_ms', r.get('elapsed_seconds', 0)*1000)


def best_candidates(row, count=5):
    return sorted(row['result']['visits'], key=lambda x: (-visits(x), -x['prior']))[:count]


def representative(rows):
    center = st.median(row['result']['simulations'] for row in rows)
    return min(rows, key=lambda row: (abs(row['result']['simulations']-center), row.get('replicate', 0)))


def fmt(value, decimals=2):
    return '—' if value is None else f'{value:,.{decimals}f}'


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_fig(fig, directory, name):
    fig.savefig(directory / f'{name}.png', dpi=155, facecolor='white', bbox_inches='tight')
    fig.savefig(directory / f'{name}.svg', facecolor='white', bbox_inches='tight')
    plt.close(fig)


def trim(hist):
    nonzero = [i for i,n in enumerate(hist) if n]
    return max(nonzero) if nonzero else 0


def normalize(hist):
    total = sum(hist)
    return np.asarray(hist, dtype=float)/total if total else np.zeros(len(hist))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    charts = out/'graphs'; charts.mkdir(exist_ok=True)
    telemetry_rows = read_runs(args.input)
    rows = [row for row in telemetry_rows if row.get('group') != 'overhead']
    assert rows, 'no telemetry runs found'
    cases = [case for case in CASES if any(row['case'] == case for row in rows)]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .18, 'svg.fonttype': 'none'})
    time_rows = [row for row in rows if row['mode'] == 'time']
    fixed_rows = [row for row in rows if row['mode'] != 'time']
    groups = defaultdict(list)
    for row in rows:
        groups[(row['case'], row['mode'], row['budget_ms'] if row['mode']=='time' else row['budget_simulations'],
                row['memory_mib'], row.get('group', ''))].append(row)

    all_metrics, candidate_rows, leaf_rows, path_rows, phase_rows, snapshot_rows, window_rows = [], [], [], [], [], [], []
    for row in rows:
        result = row['result']; telemetry = result['telemetry']; counters = telemetry['counters']
        depth = row['_depth']; selected = next(c for c in result['visits'] if move(c) == result['bestmove'])
        n = result['simulations']; root_map = {move(x):x for x in telemetry['roots']}
        event = telemetry['first_allocation_failure']; cap = telemetry['first_reserved_cap']
        base = {key:row.get(key) for key in ['run_id','case','mode','group','budget_ms','budget_simulations','memory_mib','replicate']}
        metric = {**base, 'simulations':n, 'elapsed_ms':elapsed(row),
                  'process_wall_ms':row.get('process_wall_ms'), 'bestmove':result['bestmove'],
                  'selected_visits':visits(selected), 'selected_visit_fraction':visits(selected)/n,
                  'selected_q':selected['q'], 'selected_policy':selected['prior'],
                  **{f'leaf_{key}':value for key,value in depth.items()},
                  'leaf_at_max_count':telemetry['leaf_depth'][depth['max']],
                  'leaf_at_max_fraction':telemetry['leaf_depth'][depth['max']]/n,
                  'arena_bytes':result['memory_bytes'], 'telemetry_bytes':telemetry.get('telemetry_memory_bytes'),
                  'reserved_cap_simulation':cap['simulation'] if cap['occurred'] else None,
                  'reserved_cap_elapsed_ms':cap['elapsed_ms'] if cap['occurred'] else None,
                  'failure_simulation':event['simulation'] if event['occurred'] else None,
                  'failure_elapsed_ms':event['elapsed_ms'] if event['occurred'] else None,
                  'pre_failure_mean':row['_pre']['mean'], 'pre_failure_median':row['_pre']['median'],
                  'pre_failure_max':row['_pre']['max'], 'pre_failure_samples':row['_pre']['samples'],
                  'post_failure_mean':row['_post']['mean'], 'post_failure_median':row['_post']['median'],
                  'post_failure_max':row['_post']['max'], 'post_failure_samples':row['_post']['samples'],
                  **counters}
        all_metrics.append(metric)
        for rank,candidate in enumerate(best_candidates(row),1):
            hist = root_map[move(candidate)]['leaf_depth']
            candidate_rows.append({**base, 'rank_visits':rank, 'uci':move(candidate),
                    'selected':move(candidate)==result['bestmove'], 'policy':candidate['prior'],
                    'q':candidate['q'], 'visits':visits(candidate), 'visit_fraction':visits(candidate)/n,
                    **{f'leaf_{k}':v for k,v in depth_summary(hist).items()}})
        for d,count in enumerate(telemetry['leaf_depth']):
            if count:
                leaf_rows.append({**base,'depth_plies':d,'leaf_endpoints':count,'fraction':count/n})
        for d,count in enumerate(telemetry['traversal_depth']):
            if count:
                path_rows.append({**base,'depth_plies':d,'path_visits':count,'fraction_of_simulations':count/n})
        for phase in ['pre_failure','post_failure']:
            hist = telemetry[f'{phase}_leaf_depth']; total=sum(hist)
            for d,count in enumerate(hist):
                if count:
                    phase_rows.append({**base,'phase':phase,'depth_plies':d,'leaf_endpoints':count,'phase_fraction':count/total})
        for snapshot in telemetry['snapshots']:
            b = {**base,'snapshot_simulation':snapshot['simulation'],'snapshot_elapsed_ms':snapshot['elapsed_ms'],
                 'snapshot_reason':snapshot['reason']}
            window_rows.append({**b,**snapshot['window_depth'],
                        'node_creations':snapshot['counters']['node_creations'],
                        'child_fail_byte_capacity':snapshot['counters']['child_fail_byte_capacity'],
                        'non_retained_neural_evaluations':snapshot['counters']['non_retained_neural_evaluations']})
            for candidate in snapshot['roots']:
                snapshot_rows.append({**b,'uci':move(candidate),'policy':candidate['prior'],
                    'visits':visits(candidate),'q':candidate['q'],
                    'visit_fraction':visits(candidate)/snapshot['simulation'] if snapshot['simulation'] else 0})
    for filename, table in [('all-run-metrics.csv',all_metrics),('top5-candidates.csv',candidate_rows),
                            ('leaf-depth-histograms.csv',leaf_rows),('depth-path-visits.csv',path_rows),
                            ('pre-post-failure-histograms.csv',phase_rows),('root-candidate-snapshots.csv',snapshot_rows),
                            ('depth-windows.csv',window_rows)]:
        write_csv(out/filename,table)

    summary_rows=[]
    for key, group in sorted(groups.items()):
        case,mode,budget,memory,kind=key
        rep=representative(group)
        hist=np.sum([row['result']['telemetry']['leaf_depth'] for row in group],axis=0).tolist()
        pooled=depth_summary(hist)
        caps=[row['result']['telemetry']['first_allocation_failure'] for row in group]
        summary_rows.append({'case':case,'mode':mode,'group':kind,'budget':budget,'memory_mib':memory,
                'repeats':len(group),'representative_run':rep['run_id'],
                'simulations_median':st.median(row['result']['simulations'] for row in group),
                'simulations_min':min(row['result']['simulations'] for row in group),
                'simulations_max':max(row['result']['simulations'] for row in group),
                'elapsed_ms_median':st.median(elapsed(row) for row in group),
                'leaf_mean_pooled':pooled['mean'],'leaf_median_pooled':pooled['median'],
                'leaf_max_median':st.median(row['_depth']['max'] for row in group),
                'leaf_max_min':min(row['_depth']['max'] for row in group),
                'leaf_max_max':max(row['_depth']['max'] for row in group),
                'bestmoves':';'.join(row['result']['bestmove'] for row in group),
                'node_creations_median':st.median(row['result']['telemetry']['counters']['node_creations'] for row in group),
                'legal_expansions_median':st.median(row['result']['telemetry']['counters']['legal_expansions'] for row in group),
                'non_root_legal_expansions_median':st.median(row['result']['telemetry']['counters']['legal_expansions']-row['result']['telemetry']['counters']['root_legal_expansions'] for row in group),
                'allocation_failures_median':st.median(sum(row['result']['telemetry']['counters'][k] for k in
                    ['child_fail_node_limit','child_fail_byte_capacity','expansion_fail_byte_capacity']) for row in group),
                'failure_runs':sum(event['occurred'] for event in caps),
                'failure_elapsed_ms_median':st.median(event['elapsed_ms'] for event in caps if event['occurred']) if any(event['occurred'] for event in caps) else None})
    write_csv(out/'comparison-summary.csv',summary_rows)
    time_summaries=[s for s in summary_rows if s['mode']=='time']
    fixed_summaries=[s for s in summary_rows if s['mode']!='time']
    write_csv(out/'time-comparison.csv',time_summaries)
    write_csv(out/'fixed-simulations-comparison.csv',fixed_summaries)

    # Time budgets: measured endpoint statistics and measured throughput are kept
    # separate from fixed simulation controls.
    fig,axes=plt.subplots(len(cases),3,figsize=(13.8,3.0*len(cases)),squeeze=False)
    for i,case in enumerate(cases):
        for memory in [64,512]:
            subset=sorted([s for s in time_summaries if s['case']==case and s['memory_mib']==memory],key=lambda s:s['budget'])
            if not subset: continue
            x=[s['budget'] for s in subset]
            for j,(field,title) in enumerate([('leaf_mean_pooled','Mean endpoint depth'),('leaf_median_pooled','Median endpoint depth'),('leaf_max_median','Max endpoint depth')]):
                axes[i,j].plot(x,[s[field] for s in subset],'-o',color=COLORS[memory],label=f'{memory} MiB')
                if j==2:
                    axes[i,j].fill_between(x,[s['leaf_max_min'] for s in subset],[s['leaf_max_max'] for s in subset],color=COLORS[memory],alpha=.14)
                axes[i,j].set_xscale('log');axes[i,j].set_xticks([100,1000,10000],['100ms','1s','10s'])
                axes[i,j].set_title(f'{ENGLISH[case]}\n{title}');axes[i,j].set_ylabel('Plies below root')
    axes[0,0].legend();fig.suptitle('Time limits: leaf endpoint depth (max bands = repeat range)',fontsize=14)
    fig.tight_layout();save_fig(fig,charts,'time-depth-summary')

    fig,axes=plt.subplots(len(cases),2,figsize=(12.8,2.9*len(cases)),squeeze=False)
    for i,case in enumerate(cases):
        for memory in [64,512]:
            subset=sorted([s for s in time_summaries if s['case']==case and s['memory_mib']==memory],key=lambda s:s['budget'])
            x=[s['budget'] for s in subset]
            axes[i,0].plot(x,[s['simulations_median'] for s in subset],'-o',color=COLORS[memory],label=f'{memory} MiB')
            axes[i,1].plot(x,[s['node_creations_median']/s['simulations_median'] for s in subset],'-o',color=COLORS[memory],label=f'{memory} MiB')
        axes[i,0].set_yscale('log');axes[i,0].set_ylabel('Completed simulations')
        axes[i,1].set_ylabel('New retained nodes / simulation');axes[i,1].set_ylim(0,1.05)
        for ax in axes[i]:
            ax.set_xscale('log');ax.set_xticks([100,1000,10000],['100ms','1s','10s']);ax.set_title(ENGLISH[case])
    axes[0,0].legend();fig.suptitle('Time limits: simulations and retained-node creation',fontsize=14)
    fig.tight_layout();save_fig(fig,charts,'time-expansion-summary')

    long_reps={}
    for case in cases:
        for memory in [64,512]:
            subset=[row for row in time_rows if row['case']==case and row['memory_mib']==memory and row.get('budget_ms')==10000]
            if subset:long_reps[(case,memory)]=representative(subset)
    fig,axes=plt.subplots(len(cases),2,figsize=(13.2,3.0*len(cases)),squeeze=False)
    fig2,axes2=plt.subplots(len(cases),1,figsize=(11.6,2.8*len(cases)),squeeze=False)
    for i,case in enumerate(cases):
        for memory in [64,512]:
            row=long_reps.get((case,memory))
            if not row: continue
            telemetry=row['result']['telemetry'];hist=telemetry['leaf_depth'];end=trim(hist);x=np.arange(end+1)
            prob=normalize(hist)[:end+1]
            axes[i,0].step(x,prob*100,where='mid',color=COLORS[memory],label=f'{memory} MiB; {row["result"]["bestmove"]}')
            axes[i,1].step(x,np.cumsum(prob)*100,where='post',color=COLORS[memory],label=f'{memory} MiB')
            axes2[i,0].step(x,np.asarray(telemetry['traversal_depth'][:end+1])/row['result']['simulations']*100,
                          where='post',color=COLORS[memory],label=f'{memory} MiB')
        axes[i,0].set_title(f'{ENGLISH[case]}\nLeaf endpoint probability')
        axes[i,1].set_title(f'{ENGLISH[case]}\nLeaf endpoint cumulative distribution')
        axes[i,0].set_ylabel('% of completed simulations');axes[i,1].set_ylabel('% endpoints at or below x depth')
        axes[i,1].set_ylim(0,101)
        for ax in axes[i]:ax.set_xlabel('Endpoint depth (plies)');ax.legend(fontsize=8)
        axes2[i,0].set_title(ENGLISH[case]);axes2[i,0].set_ylim(0,101)
        axes2[i,0].set_ylabel('% simulations reaching depth');axes2[i,0].set_xlabel('Path depth (plies)');axes2[i,0].legend()
    fig.suptitle('10s: endpoint distributions (representative run = median simulation count)',fontsize=13)
    fig.tight_layout();save_fig(fig,charts,'leaf-distribution-10s')
    fig2.suptitle('10s: traversal visits by depth; depth 0 = all simulations',fontsize=13)
    fig2.tight_layout();save_fig(fig2,charts,'path-depth-visits-10s')

    selected_depth_rows=[]
    fig,axes=plt.subplots(len(cases),2,figsize=(13.0,3.1*len(cases)),squeeze=False)
    selected_depth_text=[]
    for i,case in enumerate(cases):
        for j,memory in enumerate([64,512]):
            row=long_reps.get((case,memory));ax=axes[i,j]
            if not row:continue
            result=row['result'];telemetry=result['telemetry'];selected=result['bestmove']
            root=next(c for c in telemetry['roots'] if move(c)==selected)
            hist=root['leaf_depth'];conditional=depth_summary(hist);overall=row['_depth']
            shallow=sum(hist[:3])/sum(hist)
            selected_depth_rows.append({'case':case,'memory_mib':memory,'run_id':row['run_id'],
                'selected_move':selected,'selected_visits':visits(root),'selected_q':root['q'],
                'selected_policy':root['prior'],'selected_leaf_mean':conditional['mean'],
                'selected_leaf_median':conditional['median'],'selected_leaf_max':conditional['max'],
                'selected_leaf_at_depth_2':hist[2],'selected_leaf_depth_up_to_2_fraction':shallow,
                'whole_leaf_mean':overall['mean'],'whole_leaf_median':overall['median'],
                'whole_max_depth':overall['max']})
            end=trim(telemetry['leaf_depth']);x=np.arange(end+1)
            ax.step(x,normalize(telemetry['leaf_depth'])[:end+1]*100,where='mid',color='#64748b',label=f'All endpoints; mean={overall["mean"]:.2f}, max={overall["max"]}')
            ax.step(x,normalize(hist)[:end+1]*100,where='mid',color='#dc2626',label=f'{selected} endpoints; mean={conditional["mean"]:.2f}, max={conditional["max"]}')
            ax.set_title(f'{ENGLISH[case]} / {memory} MiB\nSelected {selected}: {shallow*100:.3f}% endpoints at depth <=2')
            ax.set_xlabel('Endpoint depth below root (plies)');ax.set_ylabel('Conditional endpoint probability (%)')
            ax.legend(fontsize=7)
            if memory==64 and selected==TARGET[case]:
                selected_depth_text.append(f"- {LABEL[case]} 64MiB 10초 대표 실행: 전체 최대 깊이는 **{overall['max']}**였지만 선택된 `{selected}` 가지의 잎 평균/중앙값/최대는 **{conditional['mean']:.3f}/{conditional['median']:g}/{conditional['max']}**, 깊이 2 이하 비중은 **{shallow*100:.4f}%**였다. 이 가지의 방문수는 {visits(root):,}, Q는 {root['q']:+.5f}다.")
    fig.suptitle('10s: selected root move endpoints versus all endpoints\nA deep global maximum does not imply deep analysis of the selected move',fontsize=13)
    fig.tight_layout();save_fig(fig,charts,'selected-vs-whole-depth-10s')
    write_csv(out/'selected-depth-comparison.csv',selected_depth_rows)

    # Compact overview for the chat; all detailed figures remain available.
    fig,axes=plt.subplots(1,3,figsize=(14.0,4.5))
    for memory in [64,512]:
        subset=sorted([s for s in time_summaries if s['case']=='start' and s['memory_mib']==memory],key=lambda s:s['budget'])
        x=[s['budget'] for s in subset]
        axes[0].plot(x,[s['leaf_mean_pooled'] for s in subset],'-o',color=COLORS[memory],label=f'{memory} MiB mean')
        axes[0].plot(x,[s['leaf_median_pooled'] for s in subset],'--s',color=COLORS[memory],alpha=.75,label=f'{memory} MiB median')
    axes[0].set_xscale('log');axes[0].set_xticks([100,1000,10000],['100ms','1s','10s'])
    axes[0].set_title('Start: pooled leaf endpoint depth');axes[0].set_ylabel('Plies below root');axes[0].legend(fontsize=7.5)
    for j,case in enumerate(['start','before_Ra6_10s'],1):
        for memory in [64,512]:
            row=long_reps.get((case,memory))
            if not row:continue
            result=row['result'];root=next(c for c in result['telemetry']['roots'] if move(c)==result['bestmove'])
            hist=root['leaf_depth'];d=depth_summary(hist);end=trim(hist)
            label=f'{memory} MiB: {SAN.get(move(root),move(root))}\nQ={root["q"]:+.3f}, Visits={visits(root):,}\nmean={d["mean"]:.2f}, max={d["max"]}'
            axes[j].step(np.arange(end+2),normalize(hist)[:end+2]*100,where='mid',color=COLORS[memory],lw=1.7,label=label)
        axes[j].set_title(('Start' if case=='start' else 'Before 2...Ra6')+': selected branch at 10s')
        axes[j].set_xlabel('Endpoint depth below root (plies)');axes[j].set_ylabel('% of selected-branch endpoints')
        axes[j].legend(fontsize=7.5,loc='upper right');axes[j].set_ylim(-2,105)
    fig.suptitle('MCTS time-mode depth: 3 repeats per setting; 10s distributions use the run with median N',fontsize=12)
    fig.tight_layout();save_fig(fig,charts,'overview')

    fig,axes=plt.subplots(len(cases),2,figsize=(13.2,3.1*len(cases)),squeeze=False)
    timeline,taxes=plt.subplots(len(cases),2,figsize=(13.2,3.2*len(cases)),squeeze=False)
    for i,case in enumerate(cases):
        for j,memory in enumerate([64,512]):
            row=long_reps.get((case,memory));ax=axes[i,j];tx=taxes[i,j]
            if not row: continue
            telemetry=row['result']['telemetry'];event=telemetry['first_allocation_failure']
            for field,label,color in [('pre_failure_leaf_depth','Before first allocation failure','#2563eb'),
                                      ('post_failure_leaf_depth','From failing simulation onward','#dc2626')]:
                hist=telemetry[field];n=sum(hist)
                if n:ax.step(np.arange(len(hist)),normalize(hist)*100,where='mid',label=f'{label}; N={n:,}',color=color)
            ax.set_xlim(0,trim(telemetry['leaf_depth'])+1);ax.set_ylabel('% of phase endpoints');ax.set_xlabel('Endpoint depth (plies)')
            status=f"failure at sim {event['simulation']:,}" if event['occurred'] else 'No allocation failure'
            ax.set_title(f'{ENGLISH[case]} / {memory} MiB\n{status}');ax.legend(fontsize=7)
            snapshots=[s for s in telemetry['snapshots'] if s['window_depth']['samples']]
            xs=[s['elapsed_ms']/1000 for s in snapshots]
            tx.plot(xs,[s['window_depth']['mean_depth'] for s in snapshots],label='Window mean depth',color='#2563eb')
            tx.plot(xs,[s['window_depth']['median_depth'] for s in snapshots],label='Window median depth',color='#16a34a',alpha=.85)
            if event['occurred']:tx.axvline(event['elapsed_ms']/1000,color='#dc2626',ls='--',label='First allocation failure')
            tx.set_title(f'{ENGLISH[case]} / {memory} MiB');tx.set_xlabel('Search elapsed seconds');tx.set_ylabel('Plies in completed window')
            tx.legend(fontsize=7)
    fig.suptitle('10s: endpoint distribution before and after actual allocation failure',fontsize=13)
    fig.tight_layout();save_fig(fig,charts,'pre-post-saturation')
    timeline.suptitle('10s: depth windows and exact first allocation-failure event',fontsize=13)
    timeline.tight_layout();save_fig(timeline,charts,'saturation-depth-timeline')

    candidate_figs=[]
    for case in cases:
        fig,axes=plt.subplots(3,2,figsize=(13.0,10.4),squeeze=False)
        for j,memory in enumerate([64,512]):
            row=long_reps.get((case,memory))
            if not row: continue
            telemetry=row['result']['telemetry'];top=best_candidates(row);ids=[move(c) for c in top]
            if TARGET[case] not in ids:ids.append(TARGET[case])
            colors=plt.get_cmap('tab10')(np.arange(len(ids)))
            rootmap={move(c):c for c in row['result']['visits']}
            for uci,color in zip(ids,colors):
                points=[]
                for snap in telemetry['snapshots']:
                    c=next((c for c in snap['roots'] if move(c)==uci),None)
                    if c and snap['simulation']>0:points.append((snap['simulation'],c))
                if not points:continue
                p=rootmap[uci]['prior'];label=f'{SAN.get(uci,uci)} (P={p*100:.3g}%)'
                xs=[n for n,c in points]
                axes[0,j].plot(xs,[visits(c)/n*100 for n,c in points],label=label,color=color)
                axes[1,j].plot(xs,[c['q'] if visits(c) else np.nan for n,c in points],label=label,color=color)
                rh=next((r['leaf_depth'] for r in telemetry['roots'] if move(r)==uci),None)
                if rh and sum(rh):axes[2,j].step(np.arange(trim(rh)+1),normalize(rh)[:trim(rh)+1]*100,where='mid',label=label,color=color)
            axes[0,j].set_title(f'{memory} MiB / selected {row["result"]["bestmove"]}')
            axes[0,j].set_ylabel('Root visit share (%)');axes[0,j].set_ylim(0,101)
            axes[1,j].set_ylabel('Root Q (mean backed-up W-L)');axes[1,j].set_ylim(-1.05,1.05)
            axes[2,j].set_ylabel('Conditional endpoint distribution (%)');axes[2,j].set_xlabel('Endpoint depth under root move (plies)')
            for k in [0,1]:
                axes[k,j].set_xscale('log');axes[k,j].set_xlabel('Completed simulations');axes[k,j].legend(fontsize=7)
            axes[2,j].legend(fontsize=7)
            event=telemetry['first_allocation_failure']
            if event['occurred']:
                for k in [0,1]:axes[k,j].axvline(event['simulation'],color='#dc2626',ls='--',alpha=.6)
        fig.suptitle(f'{ENGLISH[case]}: 10s root candidates, Q and depth\nTop five by final visits plus historical target; red dash = first allocation failure',fontsize=13)
        fig.tight_layout();name=f'candidate-progress-{case}';save_fig(fig,charts,name);candidate_figs.append(name)

    fig,axes=plt.subplots(len(cases),2,figsize=(13.0,3.0*len(cases)),squeeze=False)
    for i,case in enumerate(cases):
        for memory in [64,512]:
            unique={}
            for s in fixed_summaries:
                if s['case']==case and s['memory_mib']==memory:unique[s['budget']]=s
            subset=[unique[n] for n in sorted(unique)];x=[s['budget'] for s in subset]
            axes[i,0].plot(x,[s['leaf_mean_pooled'] for s in subset],'-o',color=COLORS[memory],label=f'{memory} MiB mean')
            axes[i,0].plot(x,[s['leaf_median_pooled'] for s in subset],'--s',color=COLORS[memory],alpha=.7,label=f'{memory} MiB median')
            axes[i,1].plot(x,[s['leaf_max_median'] for s in subset],'-o',color=COLORS[memory],label=f'{memory} MiB')
        for j in [0,1]:
            axes[i,j].set_xscale('log');axes[i,j].set_xlabel('Identical requested simulations (per position)')
            axes[i,j].set_ylabel('Plies');axes[i,j].legend(fontsize=8)
            axes[i,j].set_title(f'{ENGLISH[case]}\n'+('Endpoint mean / median' if j==0 else 'Max endpoint depth'))
    fig.suptitle('Fixed-simulation controls: memory capacity compared at identical N',fontsize=13)
    fig.tight_layout();save_fig(fig,charts,'fixed-simulation-depth-summary')

    raw_rows=[json.loads(line) for line in args.input.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    perf=[row for row in raw_rows if row.get('group')=='overhead']
    performance={}
    performance_text='성능 대조 로그가 아직 입력되지 않았다.'
    if perf:
        paired=defaultdict(dict)
        for row in perf:paired[row['replicate']][row['variant']]=row
        complete=[(rep,d) for rep,d in sorted(paired.items()) if all(k in d for k in ['pristine','off','on'])]
        performance['complete_paired_rounds']=len(complete)
        if complete:
            def signature(result):
                return {k:result[k] for k in ['bestmove','simulations','value','neural_evals','max_depth','visits']}
            for rep,group in complete:
                common=['bestmove','simulations','max_depth']
                assert {k:group['off']['result'][k] for k in common}=={k:group['pristine']['result'][k] for k in common}, ('off/pristine common-field parity',rep)
                assert signature(group['on']['result'])==signature(group['off']['result']), ('on parity',rep)
            rng=np.random.default_rng(20261024)
            performance['off_pristine_common_fields_equal']=['bestmove','simulations','max_depth']
            performance['off_on_full_root_statistics_equal']=True
            for name,first,second in [('off_vs_pristine','off','pristine'),('on_vs_off','on','off')]:
                logratios=np.log([elapsed(group[first])/elapsed(group[second]) for rep,group in complete])
                boot=np.exp(rng.choice(logratios,size=(20000,len(logratios)),replace=True).mean(axis=1))
                performance[name]={'time_ratio_geometric_mean':float(np.exp(logratios.mean())),
                                   'time_ratio_bootstrap_95_percentile_ci':np.quantile(boot,[.025,.975]).tolist()}
            perf_table=[]
            for variant in ['pristine','off','on']:
                relevant=[group[variant] for rep,group in complete]
                perf_table.append({'variant':variant,'paired_rounds':len(complete),
                    'median_search_ms':st.median(elapsed(row) for row in relevant),
                    'min_search_ms':min(elapsed(row) for row in relevant),
                    'max_search_ms':max(elapsed(row) for row in relevant),
                    'median_simulations_per_second':st.median(row['result']['simulations']/elapsed(row)*1000 for row in relevant)})
            write_csv(out/'performance-summary.csv',perf_table)
            fig,ax=plt.subplots(figsize=(9.5,5.2))
            for rep,group in complete:
                baseline=elapsed(group['pristine'])
                ax.plot([0,1,2],[elapsed(group[k])/baseline*100 for k in ['pristine','off','on']],'-o',color='#64748b',alpha=.35,lw=1)
            medians=[st.median(elapsed(group[k])/elapsed(group['pristine'])*100 for rep,group in complete) for k in ['pristine','off','on']]
            ax.plot([0,1,2],medians,'-o',color='#2563eb',lw=2.8,label='Median normalized paired time')
            ax.axhline(100,color='#111827',ls='--',lw=1)
            ax.set_xticks([0,1,2],['Pristine source','Telemetry OFF','Telemetry ON'])
            ax.set_ylabel('Search time / paired pristine time (%)')
            ax.set_title(f'Optional telemetry cost: {len(complete)} randomized paired rounds\nStart, N=65,536, 512 MiB; model load and JSON serialization excluded')
            ax.legend();fig.tight_layout();save_fig(fig,charts,'instrumentation-overhead')
            lines=['| 버전 | 반복 | 탐색 시간 중앙값(ms) | 범위(ms) | 중앙값 sims/s |',
                   '|---|---:|---:|---|---:|']
            for item in perf_table:
                lines.append(f"| {item['variant']} | {item['paired_rounds']} | {item['median_search_ms']:.2f} | {item['min_search_ms']:.2f}–{item['max_search_ms']:.2f} | {item['median_simulations_per_second']:,.0f} |")
            lines.append('')
            for label,key in [('OFF/기존 소스','off_vs_pristine'),('ON/OFF','on_vs_off')]:
                item=performance[key];low,high=item['time_ratio_bootstrap_95_percentile_ci']
                lines.append(f"- {label} 시간비 기하평균 **{item['time_ratio_geometric_mean']:.4f}배**, 대응 반복의 95% 부트스트랩 구간 {low:.4f}–{high:.4f}배. 1배는 같은 소요시간을 뜻한다.")
            lines.append('시드 20261024, 20,000회 대응 로그비 재표집의 백분위 구간이다. 이 한 위치·한 예산의 측정 불확실성이며 모든 기기·위치에 대한 비용 상한은 아니다. ON/OFF 결과 수·방문수·Q·prior·value는 모든 대응 반복에서 정확히 일치했다. pristine 성능 probe는 간소화된 출력이므로 OFF/pristine의 비교는 공통 bestmove·simulations·max_depth 세 필드 일치만 확인했다. 기존 배포 엔진과 새 엔진의 전체 통계 동등성은 별도의 UCI 검증으로 구분한다.')
            lines.append('![선택적 계측 비용](graphs/instrumentation-overhead.png)')
            performance_text='\n'.join(lines)
    uci_text='별도 UCI 검증 로그는 이 분석 시점에 입력되지 않았다.'
    uci_path=out/'validation-uci.json'
    if uci_path.exists():
        uci=json.loads(uci_path.read_text(encoding='utf-8'))
        assert uci['status']=='passed'
        response=uci['uci_100ms_response_summary']
        write_csv(out/'uci-response-summary.csv',response)
        performance['uci_response_100ms']=response
        performance['production_new_off_on_uci_equivalence_cases']={case:{
            'production_vs_new_off':value['production_vs_new_off'],
            'new_off_vs_new_on':value['new_off_vs_new_on']}
            for case,value in uci['cases'].items()}
        lines=['[별도 UCI 검증](validation-uci.json)에서 4개 위치의 배포 엔진/새 OFF/새 ON 전체 root 통계를 절대 오차 1e-8 이내에서 비교해 통과했다. ON/OFF는 정확히 일치했고 배포 엔진 비교의 미소 차이는 직렬화 정밀도 범위였다. stop/infinite 이후 후속 탐색도 통과했다.',
               '', 'UCI 응답 시간은 탐색 내부 시간에 JSON 직렬화·출력·수신 비용까지 더한 사용자 관측값이다. 아래는 별도 100ms 요청 각 3회다. 고정 N 네이티브 비용과 구분한다.',
               '', '| 계측 | 반복 | 요청(ms) | bestmove 도착 중앙값(ms) | 범위(ms) | info 이후 bestmove 도착 지연 중앙값(ms) |',
               '|---|---:|---:|---:|---|---:|']
        for item in response:
            lines.append(f"| {'ON' if item['enabled'] else 'OFF'} | {item['repeats']} | 100 | {item['median_elapsed_to_bestmove_ms']:.3f} | {item['min_elapsed_to_bestmove_ms']:.3f}–{item['max_elapsed_to_bestmove_ms']:.3f} | {item['median_after_info_delivery_ms']:.3f} |")
        lines.append('')
        lines.append('ON은 탐색 종료 후 상세 telemetry JSON을 bestmove 앞에서 전송하므로 그 직렬화·전송 및 수신 비용이 응답 지연에 포함된다. 계측 기본값은 OFF다. 이번 네이티브 시간 비교는 JSON 직렬화·전송을 제외하므로 위 응답 시간과 같은 수치로 해석하지 않는다. 긴 실행일수록 스냅샷이 많아 상세 출력 크기도 달라진다.')
        uci_text='\n'.join(lines)

    def table(summary, time_mode):
        lines=['| 위치 | '+('시간' if time_mode else '고정 N')+' | MiB | N 중앙값 | 잎 평균 / 중앙값 | 최대 깊이 중앙값 [범위] | 새 노드 / 수 목록 확장 | 확장 실패 | 최초 실패(ms) | 선택 수 |',
               '|---|---:|---:|---:|---|---|---:|---:|---:|---|']
        order={c:i for i,c in enumerate(cases)}
        for s in sorted(summary,key=lambda s:(order[s['case']],s['budget'],s['memory_mib'],s['group'])):
            budget=(f"{s['budget']/1000:g}s" if time_mode else f"{s['budget']:,}")
            best=' / '.join(SAN.get(v,v) for v in s['bestmoves'].split(';'))
            lines.append(f"| {LABEL[s['case']]} | {budget} | {s['memory_mib']} | {s['simulations_median']:,.0f} | {s['leaf_mean_pooled']:.2f} / {s['leaf_median_pooled']:g} | {s['leaf_max_median']:g} [{s['leaf_max_min']}–{s['leaf_max_max']}] | {s['node_creations_median']:,.0f} / {s['non_root_legal_expansions_median']:,.0f} | {s['allocation_failures_median']:,.0f} | {fmt(s['failure_elapsed_ms_median'])} | {best} |")
        return '\n'.join(lines)

    anchor_rows=[row for row in fixed_rows if row.get('group')=='historical_anchor']
    explanations=[]
    for case in cases:
        explanations.append(f'### {LABEL[case]}')
        relevant=sorted([r for r in anchor_rows if r['case']==case],key=lambda r:r['memory_mib'])
        if not relevant: relevant=[long_reps[(case,m)] for m in [64,512] if (case,m) in long_reps]
        for row in relevant:
            result=row['result'];t=result['telemetry'];n=result['simulations'];target=next(c for c in result['visits'] if move(c)==TARGET[case])
            selected=next(c for c in result['visits'] if move(c)==result['bestmove']);rank=1+sum(c['prior']>target['prior'] for c in result['visits'])
            d=row['_depth'];event=t['first_allocation_failure'];hist=t['leaf_depth'];m=d['max']
            mode='동일 N 과거 예산 대조' if row.get('group')=='historical_anchor' else '10초 대표 실행'
            explanations.append(f"- {row['memory_mib']}MiB, {mode}, N={n:,}: 선택 `{result['bestmove']}`(방문 {visits(selected):,}, Q={selected['q']:+.5f}, Policy={selected['prior']*100:.5f}%). 관심 수 `{TARGET[case]}`의 Policy 순위는 {rank}/{len(result['visits'])}, Policy={target['prior']*100:.5f}%, 방문 {visits(target):,}({visits(target)/n*100:.2f}%), Q={target['q']:+.5f}다.")
            explanations.append(f"  잎 깊이 평균 {d['mean']:.2f}, 중앙값 {d['median']:g}, 90백분위 {d['p90']}, 최대 {m}. 최대 깊이의 잎은 {hist[m]:,}회({hist[m]/n*100:.4f}%)다. "+
                (f"최초 할당 실패는 {event['simulation']:,}회/{event['elapsed_ms']:.2f}ms이며 전/후 평균 깊이는 {fmt(row['_pre']['mean'])}/{fmt(row['_post']['mean'])}다." if event['occurred'] else '할당 실패는 없었다.'))
            selected_hist=next(c['leaf_depth'] for c in t['roots'] if move(c)==result['bestmove'])
            sd=depth_summary(selected_hist)
            explanations.append(f"  최종 선택 수의 조건부 잎 깊이는 평균 {sd['mean']:.4f}, 중앙값 {sd['median']:g}, 최대 {sd['max']}이며, 깊이 2 이하에서 끝난 비중은 {sum(selected_hist[:3])/sum(selected_hist)*100:.4f}%다. 전체 트리 최대 깊이와 이 가지의 실제 평가 깊이를 구분해야 한다.")
            trace=[]
            for snap in t['snapshots']:
                item=next((c for c in snap['roots'] if move(c)==TARGET[case]),None)
                if item and visits(item)>0:trace.append((snap,item))
            if trace:
                first,item=trace[0]
                narrative=f"  스냅샷에서 관심 수의 최초 방문 관측: N={first['simulation']:,}, Visits={visits(item):,}, Q={item['q']:+.5f}."
                positive=next(((s,c) for s,c in trace if visits(c)>1 and c['q']>0),None)
                leader=next(((s,c) for s,c in trace if visits(c)==max(visits(other) for other in s['roots'])),None)
                if positive:
                    s,c=positive;narrative+=f" 두 번 이상 방문 후 양의 Q가 처음 관측된 스냅샷은 N={s['simulation']:,}, Q={c['q']:+.5f}."
                if leader:
                    s,c=leader;narrative+=f" 방문수 공동/단독 1위로 처음 관측된 스냅샷은 N={s['simulation']:,}, Visits={visits(c):,}, Q={c['q']:+.5f}."
                else:narrative+=' 저장된 스냅샷에서는 관심 수가 방문수 1위가 된 적이 없다.'
                explanations.append(narrative+' 이는 샘플링된 관측 시점이며 최초 전환의 정확한 시뮬레이션 번호가 아니다.')
        explanations.append(f'![후보 방문·Q·깊이](graphs/candidate-progress-{case}.png)')

    top_tables=[]
    for case in cases:
        top_tables.append(f'### {LABEL[case]} · 10초 대표 실행')
        top_tables.append('| MiB | 순위 | 수(UCI) | Policy | Q | Visits | 방문 비중 | 해당 수 잎 평균/중앙값/최대 |\n|---:|---:|---|---:|---:|---:|---:|---|')
        for memory in [64,512]:
            row=long_reps.get((case,memory))
            if not row:continue
            roots={move(c):c for c in row['result']['telemetry']['roots']}
            for rank,c in enumerate(best_candidates(row),1):
                d=depth_summary(roots[move(c)]['leaf_depth'])
                top_tables.append(f"| {memory} | {rank} | `{move(c)}` {'★' if move(c)==row['result']['bestmove'] else ''} | {c['prior']*100:.5f}% | {c['q']:+.5f} | {visits(c):,} | {visits(c)/row['result']['simulations']*100:.2f}% | {fmt(d['mean'])}/{fmt(d['median'],1)}/{d['max']} |")
    conditions={key:sorted({json.dumps(r.get(key),ensure_ascii=False) for r in rows}) for key in ['seed','threads','max_nodes']}
    summary={'runs_analyzed':len(rows),'time_runs':len(time_rows),'fixed_simulation_runs':len(fixed_rows),
             'conditions':conditions,'comparisons':summary_rows,
             'histogram_invariants_checked':True,'performance':performance,
             'representative_10s_runs':{f'{c}/{m}':r['run_id'] for (c,m),r in long_reps.items()},
             'definitions':{'leaf_endpoint':'one completed simulation termination depth; root initialization/proof excluded',
                 'traversal_depth':'root depth0=N, at depth d tail-sum of endpoint histogram',
                 'maximum':'deepest single completed simulation, not nominal minimax depth',
                 'post_failure':'starts at first failing simulation, not first reserved-memory cap event'}}
    (out/'analysis-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    report=f'''# CHROMATIC MCTS 깊이 정밀 계측

![시간 기준 깊이와 선택 가지 요약](graphs/overview.png)

## 측정 범위와 정의

분석한 실행은 총 {len(rows)}개(시간 기준 {len(time_rows)}개, 동일 시뮬레이션 수 기준 {len(fixed_rows)}개)다. 배포 엔진과 모델을 보존한 별도 계측 버전의 로그를 읽어 표와 그래프를 생성했다.

- 모든 깊이는 현재 루트 아래 **플라이**다. 최대 깊이는 가장 깊은 한 번의 시뮬레이션 경로이며 미니맥스 완전 탐색 깊이가 아니다.
- 잎 깊이는 완료된 시뮬레이션당 도착점 하나를 센다. 신규 보존 노드, 캐시된 종료 노드, 메모리 부족으로 보존되지 못한 잎, 경로 제한에 도착한 경우를 포함한다. 루트 초기 평가와 별도 메이트 증명은 잎 표본에 포함하지 않는다.
- 깊이별 경로 방문은 해당 깊이까지 도달한 시뮬레이션 수다. 깊이 0은 전체 N이며, 깊이 d의 방문수는 잎 깊이 히스토그램에서 깊이≥d인 횟수의 합이다. 보존되지 못한 잎도 해당 위치까지 이동해 평가했으므로 경로 도달에 포함한다. 이는 고유 보존 노드 수나 잎 도착점 분포와 구분한다.
- 새 노드는 실제 보존 노드 생성(루트 제외)이며, 합법 수 목록 확장 성공은 별도 카운터다. 실패는 child 노드 수 한도, child 바이트 한도, 간선 목록 바이트 한도별로 기록한다. 최대 메모리 예약 도달과 실제 첫 할당 실패는 별도 사건이다.
- Policy는 루트 합법 수 사이의 사전 확률, Q는 그 수의 역관점 백업 W−L 평균, Visits는 선택 횟수다. **Q는 객관적 승률·센티폰이 아니다.**
- `5.Rg1`은 100ms 기보의 9번째 플라이 위치다. 이전 1초 기보의 `4.Rg1`은 별도 보조 위치로 측정했다.

시간 모드에서는 같은 100ms/1초/10초를 주어 실제 완료 N을 측정했다. 반복의 N 중앙값·범위와 최대 깊이 중앙값·범위를 표시한다. 평균·중앙값 잎 깊이는 반복의 모든 잎을 합친 분포에서 계산한다. 그래프의 대표 실행은 해당 조건의 N 중앙값에 가장 가까운 실행이다. 시간 모드에서는 N이 달라지므로 메모리 효과와 처리량 효과를 함께 받는다.

동일 N 모드는 같은 위치·같은 모델·시드·스레드·요청 N을 고정하여 메모리 상한의 영향을 비교한다. 같은 벽시계 시간의 강도 비교로 해석하지 않는다. 조건 값: `{json.dumps(conditions,ensure_ascii=False)}`. 구체적 설정·해시·명령은 원시 JSONL과 실행 로그에 기록되어 있다.

## 시간 기준 비교

{table(time_summaries,True)}

확장 실패는 세 실패 카운터의 합이다. 같은 시뮬레이션에서 child 실패 뒤 NN 평가가 발생할 수 있으므로 실패 횟수를 서로 다른 위치 수로 부르지 않는다. 전용 CSV에는 실패 원인과 미보존 NN 평가를 따로 기록했다.

![시간별 깊이](graphs/time-depth-summary.png)

![시뮬레이션과 실제 새 노드 생성](graphs/time-expansion-summary.png)

## 동일 시뮬레이션 수 비교

{table(fixed_summaries,False)}

![동일 N 깊이 비교](graphs/fixed-simulation-depth-summary.png)

## 깊이 분포와 포화 전후

평균·중앙값과 분포의 대부분이 얕은데 극소수 경로만 깊으면 최대 깊이는 대표적인 수읽기 양을 과장할 수 있다. 다음 그림은 개별 최대값 대신 모든 잎 도착점의 분포, 그 누적분포, 각 깊이까지 도달한 실제 경로 방문 비중을 보여준다.

![10초 잎 깊이 분포](graphs/leaf-distribution-10s.png)

![깊이별 실제 경로 방문](graphs/path-depth-visits-10s.png)

포화 전후는 **실제 첫 할당 실패**를 경계로 나눈다. 실패한 시뮬레이션부터 후반에 포함한다. 메모리 예약이 상한에 도달해도 남은 예약 슬롯을 사용하여 실패 없이 확장할 수 있으므로 예약 도달 시간을 포화 실패 시간으로 대체하지 않는다. 예약 도달 사건은 예약 바이트가 정확히 상한과 같을 때만 발생한다. 자료형 크기/반올림으로 소량의 공간이 남은 상태에서도 다음 할당이 실패할 수 있으므로 이 사건이 false라고 포화 실패가 없었다고 해석하지 않는다. 실패가 없으면 후반 분포는 비어 있다.

![실제 포화 전후 분포](graphs/pre-post-saturation.png)

시간 창 그래프는 마지막 스냅샷 이후 새로 완료된 잎들로 계산한다. 누적 평균과 달리 포화 뒤 깊이 변화가 드러나며, 첫 실패는 정확한 사건 시점으로 표시한다. 후보 추이·리더 전환은 스냅샷 해상도에서 관찰한 것이므로 매 시뮬레이션의 정확한 전환 시점으로 해석하지 않는다.

![포화 전후 깊이 시간 창](graphs/saturation-depth-timeline.png)

## 관심 수가 선택된 과정

다음은 과거 탐색 횟수를 고정한 대조를 우선 사용한다. 관심 수가 이번 예산에서 선택되지 않았으면 그 사실을 그대로 표시한다. 높은 Q와 방문 집중은 모델 가치에 따른 탐색 선택의 근거이며 수 자체가 좋다는 증거가 아니다. 반대로 Policy가 작더라도 반복 백업 Q가 높으면 PUCT가 해당 가지로 방문을 집중할 수 있다.

**전체 트리의 최대 깊이와 최종 선택 가지의 실제 잎 깊이는 다를 수 있다.** 아래 수치는 시간 모드 10초 대표 실행이며, 뒤의 동일 N 대조와 구분한다.

{chr(10).join(selected_depth_text)}

![선택된 수와 전체 잎 깊이 비교](graphs/selected-vs-whole-depth-10s.png)

{chr(10).join(explanations)}

## 상위 5개 후보 · Policy/Q/Visits

각 조건의 모든 실행에 대한 상위 5개 후보는 [top5-candidates.csv](top5-candidates.csv)에 있다. 아래는 10초 대표 실행이다. ★는 최종 선택이다. 해당 수의 깊이는 그 루트 수를 선택한 시뮬레이션들만 조건부로 집계했다.

{chr(10).join(top_tables)}

## 해석의 한계와 검증

히스토그램 합=N, 경로 방문=잎 히스토그램 꼬리합, 루트 수별 잎 합=해당 수 Visits, 포화 전후 합=전체 분포, 평균·중앙값·최대 재계산 일치를 모든 분석 실행에서 검사했다. 이는 로그 회계 검증이며 수의 정답성이나 Elo 검증이 아니다.

64MiB와 512MiB에서 최대 깊이가 늘거나 선택 수가 자연스러워졌다는 사실만으로 강도 향상을 선언하지 않는다. 방문 분포, 실제 신규 확장량, Q의 출처인 가치 함수 품질, 독립 대국 결과를 함께 확인해야 한다. 이전 원인 분석에서 확인된 가치 함수의 일관성 문제는 별도로 남아 있으며, 이번 계측은 그 모델을 교체하지 않는다.

계측 기능은 선택적으로 활성화한다. off 경로는 히스토그램·스냅샷·추가 시계 읽기를 실행하지 않도록 구현했다. on의 실제 비용은 고정 N 소요시간으로 측정해야 하며 시간 모드 N만으로 분리하지 않는다. 탐색 소요시간은 모델 로드·워밍업·JSON 직렬화를 제외하며 탐색 반환 직전 계측 정리 비용은 포함한다. 프로세스 전체 소요시간은 별도 필드로 보존한다.

## 선택적 계측 비용

{performance_text}

## UCI 동등성 및 실제 응답 지연

{uci_text}

## 원자료와 재현 가능한 표

- [전체 실행 지표](all-run-metrics.csv), [조건별 비교 표](comparison-summary.csv), [분석 요약 JSON](analysis-summary.json)
- [잎 깊이 원시 빈도](leaf-depth-histograms.csv), [깊이별 경로 방문](depth-path-visits.csv), [포화 전후 빈도](pre-post-failure-histograms.csv)
- [후보별 스냅샷](root-candidate-snapshots.csv), [깊이 시간 창](depth-windows.csv), [상위 5개 후보](top5-candidates.csv)
- [선택된 수와 전체 깊이 비교](selected-depth-comparison.csv)
- [원시 실행 JSONL]({args.input.name}), [실험 조건·해시](experiment-metadata.json), 개별 명령·표준출력은 `raw/`에 보존한다.
- [네이티브 성능 표](performance-summary.csv), [UCI 응답 표](uci-response-summary.csv), [UCI 동등성·프로토콜 검증](validation-uci.json)
- 그래프는 `graphs/`에 PNG와 SVG를 함께 저장했다. 원시 실험 JSONL은 분석 입력 파일과 개별 실행 로그를 보존한다.
'''
    (out/'DEPTH_REPORT.md').write_text(report,encoding='utf-8')
    print(json.dumps({'runs':len(rows),'time':len(time_rows),'fixed':len(fixed_rows),
                      'graphs_png':len(list(charts.glob('*.png'))),'report':str(out/'DEPTH_REPORT.md')},ensure_ascii=False))


if __name__=='__main__':
    main()
