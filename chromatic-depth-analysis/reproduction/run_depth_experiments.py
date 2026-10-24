from pathlib import Path
import collections
import hashlib
import json
import random
import shutil
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/chromatic-depth-analysis'
PROJECT=ROOT/'outputs/project-chromatic-depth'
sys.path.insert(0,str(ROOT/'outputs/project-chromatic'))
sys.dont_write_bytecode=True
import chess

CASES={
 'start':([],722781,'1.g4 직전'),
 'before_Rg1_100ms':(['c2c3','d7d6','a2a3','g8f6','g1f3','b7b6','e2e4','e7e6'],6363,'5.Rg1 직전'),
 'before_Rg1_1s':(['c2c3','g8f6','a2a3','d7d6','g1f3','e7e6'],52995,'4.Rg1 직전(1초 기보)'),
 'before_Ra6_10s':(['g2g4','a7a5','c2c3'],717076,'2...Ra6 직전')}

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

OUT.mkdir(exist_ok=True);(OUT/'raw').mkdir(exist_ok=True)
MODEL=PROJECT/'dist/default.zero'
EXE=PROJECT/'build/depth_probe.exe'
BASE=PROJECT/'build/pristine_probe.exe'
manifest=json.loads((ROOT/'outputs/project-chromatic/release-manifest.json').read_text())
def preserve():
    differences=[r['path'] for r in manifest['files'] if digest(ROOT/'outputs/project-chromatic'/r['path'])!=r['sha256']]
    assert not differences,differences
    assert digest(MODEL)==digest(ROOT/'outputs/project-chromatic/dist/default.zero')
    return {'original_release_files':len(manifest['files']),'changed':differences,
            'original_engine_sha256':digest(ROOT/'outputs/project-chromatic/dist/chromatic.exe'),
            'model_sha256':digest(MODEL),'probe_sha256':digest(EXE),'pristine_probe_sha256':digest(BASE)}

metadata={'integrity_before':preserve(),'cases':{},'settings':{'seed':2026,'threads':1,'affinity_mask':2,
          'backend':'SIMD','sparse':True,'cpuct':1.5,'fpu_reduction':0,'noise':0,'temperature':0,
          'proof_nodes':0,'max_nodes':1000000,'telemetry_interval':4096,'warmup_evaluations':32},
          'time_repeats':3,'matched_repeats':1,'performance_rounds':9,
          'time_comparison_has_instrumentation_overhead':True,'search_elapsed_excludes_model_load_and_json_serialization':True}
for name,(moves,anchor,label) in CASES.items():
    board=chess.Board()
    for move in moves:board.push_uci(move)
    assert board.halfmove_clock==0 # no repetition history survives these exact roots
    metadata['cases'][name]={'moves':moves,'fen':board.fen(en_passant='fen'),'anchor_simulations':anchor,'label':label}
save(OUT/'experiment-metadata.json',metadata)
existing={}
if (OUT/'runs.jsonl').exists():
    for line in (OUT/'runs.jsonl').read_text(encoding='utf-8').splitlines():
        row=json.loads(line);existing[row['run_id']]=row

def validate(result):
    if not result.get('telemetry_enabled'):return
    t=result['telemetry'];h=t['leaf_depth'];n=result['simulations']
    assert sum(h)==n
    assert sum(r['n'] for r in result['visits'])==n
    assert sum(t['pre_failure_leaf_depth'])+sum(t['post_failure_leaf_depth'])==n
    assert t['counters']['node_creations']+1==result['arena_nodes']
    assert t['counters']['neural_evaluations']==result['neural_evals']
    assert t['depth']['max_depth']==result['max_depth']
    for d in range(len(h)):
        assert t['traversal_depth'][d]==sum(h[d:])
        assert sum(r['leaf_depth'][d] for r in t['roots'])==h[d]
    for r in t['roots']:assert sum(r['leaf_depth'])==r['visits']

def run(case,memory,mode,budget,replicate,group,variant='on',label=None):
    identity=f'{group}-{case}-{memory}MiB-{mode}-{budget}-r{replicate}-{variant}'
    if identity in existing:return existing[identity]
    args=[str(BASE if variant=='pristine' else EXE),'--model',str(MODEL),'--fen',metadata['cases'][case]['fen'],
          '--mode',mode,'--budget',str(budget),'--memory',str(memory),'--max-nodes','1000000','--seed','2026',
          '--threads','1','--affinity','2','--telemetry','true' if variant=='on' else 'false',
          '--interval','4096','--warmup','true']
    started=time.perf_counter()
    process=subprocess.run(args,capture_output=True,text=True,timeout=max(60,budget/1000+60) if mode=='time' else 120)
    wall_ms=(time.perf_counter()-started)*1000
    if process.returncode:raise RuntimeError(f'{identity}: {process.stderr}')
    payload=json.loads(process.stdout);result=payload['result'];result['elapsed_ms']=payload['elapsed_ms']
    validate(result)
    row={'run_id':identity,'case':case,'case_label':metadata['cases'][case]['label'],
         'fen':metadata['cases'][case]['fen'],'moves':metadata['cases'][case]['moves'],
         'mode':mode,'memory_mib':memory,'max_nodes':1000000,'seed':2026,'threads':1,'affinity_mask':2,
         'replicate':replicate,'group':group,'variant':variant,'budget_label':label,
         'budget_ms':budget if mode=='time' else None,'budget_simulations':budget if mode=='simulations' else None,
         'process_wall_ms':wall_ms,'result':result,'command_argv':args,'stderr':process.stderr,
         'probe_sha256':metadata['integrity_before']['pristine_probe_sha256' if variant=='pristine' else 'probe_sha256']}
    (OUT/'raw'/f'{identity}.stdout.json').write_text(process.stdout,encoding='utf-8')
    save(OUT/'raw'/f'{identity}.request.json',{k:v for k,v in row.items() if k!='result'})
    with (OUT/'runs.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    existing[identity]=row
    brief={'run':identity,'elapsed_ms':round(result['elapsed_ms'],3),'N':result['simulations'],'bestmove':result['bestmove'],'max':result['max_depth']}
    if variant=='on':
        brief.update({k:round(result['telemetry']['depth'][k],3) for k in ['mean_depth','median_depth']})
        brief['first_failure_N']=result['telemetry']['first_allocation_failure']['simulation'] or None
    print(json.dumps(brief,ensure_ascii=False),flush=True)
    return row

rng=random.Random(20261024)
# Complete time matrix first, with each paired memory order counterbalanced/randomized.
for rep in range(1,4):
    cells=[(case,ms) for case in CASES for ms in [100,1000,10000]];rng.shuffle(cells)
    for case,ms in cells:
        memories=[64,512];rng.shuffle(memories)
        for memory in memories:run(case,memory,'time',ms,rep,'time_matrix',label=f'{ms}ms')

# The 64 MiB measured-N median defines each budget, then exactly the same N is used at both caps.
matched=[]
for case in CASES:
    for ms in [100,1000,10000]:
        trials=[row for row in existing.values() if row['group']=='time_matrix' and row['case']==case and row['memory_mib']==64 and row['budget_ms']==ms]
        budget=int(statistics.median(row['result']['simulations'] for row in trials))
        matched.append({'case':case,'from_ms':ms,'simulations':budget})
        memories=[64,512];rng.shuffle(memories)
        for memory in memories:run(case,memory,'simulations',budget,1,'matched_simulations',label=f'median64_{ms}ms')
save(OUT/'matched-budgets.json',matched)

# Reproduce the exact earlier observed counts separately from current hardware/time throughput.
for case,(_,anchor,_) in CASES.items():
    for memory in [64,512]:run(case,memory,'simulations',anchor,1,'historical_anchor',label='previous_observed_N')

# Native-only, paired randomized rounds: baseline and telemetry off/on all share build flags and process affinity.
perf=[]
for rep in range(1,10):
    variants=['pristine','off','on'];rng.shuffle(variants)
    for variant in variants:perf.append(run('start',512,'simulations',65536,rep,'overhead',variant=variant))

metadata['integrity_after']=preserve()
metadata['run_count']=len(existing)
save(OUT/'experiment-metadata.json',metadata)
save(OUT/'performance-runs.json',[{k:v for k,v in row.items() if k not in ['result']}|{'result':{k:v for k,v in row['result'].items() if k!='telemetry'}} for row in perf])
print('DEPTH_EXPERIMENTS_COMPLETE '+str(len(existing)),flush=True)
