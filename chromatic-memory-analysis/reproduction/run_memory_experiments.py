from pathlib import Path
import hashlib
import json
import random
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/chromatic-memory-analysis'
NEW=ROOT/'outputs/project-chromatic-memory'
OLD=ROOT/'outputs/project-chromatic-depth'
sys.dont_write_bytecode=True
import chess

CASES={
 'start':([],722781,'1.g4 직전'),
 'before_Rg1_100ms':(['c2c3','d7d6','a2a3','g8f6','g1f3','b7b6','e2e4','e7e6'],6363,'5.Rg1 직전'),
 'before_Rg1_1s':(['c2c3','g8f6','a2a3','d7d6','g1f3','e7e6'],52995,'4.Rg1 직전(보조)'),
 'before_Ra6_10s':(['g2g4','a7a5','c2c3'],717076,'2...Ra6 직전')}
EXES={'legacy':OLD/'build/depth_probe.exe','recycled':NEW/'build/depth_probe.exe'}
MODEL=NEW/'dist/default.zero'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

OUT.mkdir(exist_ok=True);(OUT/'raw').mkdir(exist_ok=True)
previous_manifest=json.loads((ROOT/'outputs/chromatic-depth-analysis/delivery-manifest.json').read_text(encoding='utf-8'))
original_manifest=json.loads((ROOT/'outputs/project-chromatic/release-manifest.json').read_text(encoding='utf-8'))
def preserve():
    original_changed=[r['path'] for r in original_manifest['files'] if digest(ROOT/'outputs/project-chromatic'/r['path'])!=r['sha256']]
    depth_files=[r for r in previous_manifest['files'] if r['path'].startswith('outputs/project-chromatic-depth/')]
    depth_changed=[r['path'] for r in depth_files if digest(ROOT/r['path'])!=r['sha256']]
    assert not original_changed and not depth_changed
    assert digest(MODEL)==digest(ROOT/'outputs/project-chromatic/dist/default.zero')
    return {'original_files_checked':len(original_manifest['files']),'depth_files_checked':len(depth_files),
            'original_changed':original_changed,'depth_changed':depth_changed,'model_sha256':digest(MODEL),
            'probe_sha256':{k:digest(v) for k,v in EXES.items()}}

initial=preserve()
if (OUT/'experiment-metadata.json').exists():
    previous=json.loads((OUT/'experiment-metadata.json').read_text(encoding='utf-8'))
    assert previous['integrity_before']['probe_sha256']==initial['probe_sha256'],'Cannot mix builds in resumed experiment'
metadata={'integrity_before':initial,'cases':{},'variants':{'legacy':'preserved depth telemetry version; unretained NN backups',
          'recycled':'visit-threshold subtree recycling; failed attempts have no NN/backup'},
          'settings':{'seed':2026,'threads':1,'affinity_mask':2,'backend':'SIMD','sparse':True,'cpuct':1.5,
          'fpu_reduction':0,'noise':0,'temperature':0,'proof_nodes':0,'max_nodes':1000000,
          'telemetry_interval':4096,'warmup_evaluations':32},'time_repeats':3,'fixed_repeats':1,
          'matched_budget_source':'previous depth analysis matched-budgets.json; completed N identical for both variants/caps',
          'search_elapsed_excludes_model_load_warmup_and_json_serialization':True}
for case,(moves,anchor,label) in CASES.items():
    board=chess.Board()
    for m in moves:board.push_uci(m)
    assert board.halfmove_clock==0
    metadata['cases'][case]={'moves':moves,'fen':board.fen(en_passant='fen'),'anchor_simulations':anchor,'label':label}
save(OUT/'experiment-metadata.json',metadata)
existing={}
if (OUT/'runs.jsonl').exists():
    for line in (OUT/'runs.jsonl').read_text(encoding='utf-8').splitlines():
        row=json.loads(line);existing[row['run_id']]=row

def validate(result,variant,mode,budget):
    t=result['telemetry'];hist=t['leaf_depth'];n=result['simulations'];c=t['counters']
    assert sum(hist)==n and sum(row['n'] for row in result['visits'])==n
    assert t['depth']['samples']==n and t['depth']['max_depth']==result['max_depth']
    assert all(a+b==d for a,b,d in zip(t['pre_failure_leaf_depth'],t['post_failure_leaf_depth'],hist))
    if variant=='recycled':
        assert c['node_creations']+1-c['nodes_evicted']==result['arena_nodes']
        assert c['non_retained_neural_evaluations']==0
        assert result['attempted_simulations']==n+result['aborted_simulations']
        assert not result['memory_exhausted']
        assert all(p['root_statistics_preserved'] and p['success'] for p in t['prunes'])
        assert result['peak_memory_bytes']<=result['memory_bytes']
    if mode=='simulations':assert n==budget
    for root in t['roots']:assert sum(root['leaf_depth'])==root['visits']
    for d in range(len(hist)):assert t['traversal_depth'][d]==sum(hist[d:])

def run(case,memory,variant,mode,budget,rep,group,label):
    identity=f'{variant}-{group}-{case}-{memory}MiB-{mode}-{budget}-r{rep}'
    if identity in existing:return existing[identity]
    args=[str(EXES[variant]),'--model',str(MODEL),'--fen',metadata['cases'][case]['fen'],'--mode',mode,
          '--budget',str(budget),'--memory',str(memory),'--max-nodes','1000000','--seed','2026','--threads','1',
          '--affinity','2','--telemetry','true','--interval','4096','--warmup','true']
    start=time.perf_counter()
    proc=subprocess.run(args,capture_output=True,text=True,timeout=(budget/1000+60 if mode=='time' else 180))
    wall=(time.perf_counter()-start)*1000
    if proc.returncode:raise RuntimeError(f'{identity}: {proc.stderr}')
    raw=json.loads(proc.stdout);result=raw['result'];result['elapsed_ms']=raw['elapsed_ms']
    validate(result,variant,mode,budget)
    row={'run_id':identity,'case':case,'case_label':metadata['cases'][case]['label'],'fen':metadata['cases'][case]['fen'],
         'moves':metadata['cases'][case]['moves'],'variant':variant,'mode':mode,'memory_mib':memory,'replicate':rep,
         'group':group,'budget_label':label,'budget_ms':budget if mode=='time' else None,
         'budget_simulations':budget if mode!='time' else None,'seed':2026,'threads':1,'affinity_mask':2,
         'max_nodes':1000000,'process_wall_ms':wall,'probe_sha256':initial['probe_sha256'][variant],
         'command_argv':args,'stderr':proc.stderr,'result':result}
    (OUT/'raw'/f'{identity}.stdout.json').write_text(proc.stdout,encoding='utf-8')
    save(OUT/'raw'/f'{identity}.request.json',{k:v for k,v in row.items() if k!='result'})
    with (OUT/'runs.jsonl').open('a',encoding='utf-8') as file:file.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    existing[identity]=row
    c=result['telemetry']['counters'];summary={'run':identity,'ms':round(result['elapsed_ms'],3),'N':result['simulations'],
       'best':result['bestmove'],'mean':round(result['telemetry']['depth']['mean_depth'],3),
       'median':result['telemetry']['depth']['median_depth'],'max':result['max_depth'],
       'prunes':result.get('prune_count',0),'unretained_nn':c['non_retained_neural_evaluations']}
    print(json.dumps(summary),flush=True)
    return row

rng=random.Random(20261025)
for rep in range(1,4):
    cells=[(case,ms) for case in CASES for ms in [100,1000,10000]];rng.shuffle(cells)
    for case,ms in cells:
        configs=[(m,v) for m in [64,512] for v in ['legacy','recycled']];rng.shuffle(configs)
        for memory,variant in configs:run(case,memory,variant,'time',ms,rep,'time_matrix',f'{ms}ms')

budgets=json.loads((ROOT/'outputs/chromatic-depth-analysis/matched-budgets.json').read_text(encoding='utf-8'))
for group,cells in [('matched_simulations',[(b['case'],b['simulations'],f"previous_median64_{b['from_ms']}ms") for b in budgets]),
                    ('historical_anchor',[(case,anchor,'previous_observed_N') for case,(_,anchor,_) in CASES.items()])]:
    for case,n,label in cells:
        configs=[(m,v) for m in [64,512] for v in ['legacy','recycled']];rng.shuffle(configs)
        for memory,variant in configs:run(case,memory,variant,'simulations',n,1,group,label)

metadata['integrity_after']=preserve();metadata['run_count']=len(existing)
save(OUT/'experiment-metadata.json',metadata)
assert len(existing)==208
print('MEMORY_EXPERIMENTS_COMPLETE '+str(len(existing)),flush=True)
