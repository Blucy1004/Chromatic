#!/usr/bin/env python3
"""Model/search factorial diagnostic with process-attested loaded tensor hashes."""
import argparse,csv,gzip,json,os,platform,struct,subprocess,time
from pathlib import Path
import chess,psutil
from experiment import Engine,ROOT,CONFIGS,sha,save
MODELS={'zero_initial':ROOT.parent/'chromatic-shard-trainer/champion/default.zero',
        'lichess_10000':ROOT.parent/'chromatic-shard-trainer/candidates/positions-10000.zero'}
ARMS={name:{'Threads':1,'CPuct':1.5,'FPUReduction':0,'InferenceBackend':'SIMD','SparsePolicyHead':True,
 'MateProofNodes':0,'MateProofPlies':1,'TreeMemoryMB':64,'MaxNodes':100000,'VerificationNodes':2048,
 'VerificationDepth':4,'VerificationCandidates':4,'SharedTree':True,
 'CounterexampleVerify':False,'AdaptiveBudget':False,'ForcingSearch':False,'TacticalVerify':False,**CONFIGS[name]}
 for name in ['disabled','tactical','combination']}
def load_fixtures():
 path=ROOT.parent/'chromatic-arc-trainer/fixtures/official-puzzles.csv';rows=[]
 with path.open(newline='') as f:
  for row in csv.DictReader(f):
   board=chess.Board(row['FEN']);moves=row['Moves'].split();board.push_uci(moves[0]);expected=moves[1]
   if not board.is_valid() or chess.Move.from_uci(expected) not in board.legal_moves:raise ValueError('invalid published fixture')
   rows.append({'puzzle_id':row['PuzzleId'],'fen':board.fen(en_passant='fen'),'expected_uci':expected})
 if len(rows)!=100:raise ValueError('Expected the same 100 puzzles')
 return path,rows
def identity(engine,model):
 engine.send('zero model');record=json.loads(engine.until(lambda s:s.startswith('{'))[-1]);expected=sha(model)
 if not record['loaded'] or record['sha256']!=expected:raise AssertionError('Loaded tensor identity mismatch')
 return record
def main():
 p=argparse.ArgumentParser();p.add_argument('--engine',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--repeats',type=int,default=3);p.add_argument('--ms',type=int,default=100);p.add_argument('--legacy',action='store_true');p.add_argument('--require-windows',action='store_true');a=p.parse_args()
 if a.require_windows and platform.system()!='Windows':raise RuntimeError('Native Windows execution required')
 if a.out.exists() and any(a.out.iterdir()):raise FileExistsError('Use a fresh matrix directory')
 a.out.mkdir(parents=True,exist_ok=True)
 if a.repeats<1 or a.ms!=100:raise ValueError('This audit requires >=1 repeats and 100ms')
 process=psutil.Process();allowed=process.cpu_affinity();process.cpu_affinity(allowed[:2])
 fixture,positions=load_fixtures();exe_hash=sha(a.engine)
 source_hashes={str(path.relative_to(ROOT)):sha(path) for folder in ['src','include'] for path in sorted((ROOT/folder).rglob('*')) if path.is_file()}
 metadata={'os':platform.system(),'platform':platform.platform(),'machine':platform.machine(),'processor':platform.processor(),
  'logical_cpus':psutil.cpu_count(),'cpu_affinity':process.cpu_affinity(),'python':platform.python_version(),'python_chess':chess.__version__,
  'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'engine_path':str(a.engine.resolve()),
  'engine_sha256':exe_hash,'source_sha256':source_hashes,'fixture_path':str(fixture.resolve()),'fixture_sha256':sha(fixture),
  'seed':1,'threads':1,'milliseconds':100,'repeats':a.repeats,'arms':ARMS,'legacy_no_memory_attestation':a.legacy,
  'model_paths':{k:str(v.resolve()) for k,v in MODELS.items()},'model_sha256':{k:sha(v) for k,v in MODELS.items()}}
 save(a.out/'metadata.json',metadata);summaries=[]
 for repeat in range(a.repeats):
  # Rotate complete cells to reduce systematic ordering effects; all cells use
  # the same positions/order, CPU set, seed, worker count and movetime.
  cells=[(m,c) for m in MODELS for c in ARMS];offset=repeat%len(cells);cells=cells[offset:]+cells[:offset]
  for model_name,arm in cells:
   if a.legacy and model_name!='zero_initial':continue
   model=MODELS[model_name];before=sha(model);rows=[]
   with Engine(a.engine,ARMS[arm],model=model) as engine:
    loaded=None if a.legacy else identity(engine,model)
    uci_options=[s for s in engine.trace if s.startswith(('id ','option name '))]
    for position in positions:
     r=engine.search(position['fen'],ms=100);move=chess.Move.from_uci(r['move'])
     if move not in chess.Board(position['fen']).legal_moves:raise AssertionError('Illegal move')
     rows.append({**position,'move':r['move'],'hit':r['move']==position['expected_uci'],'elapsed_ms':r['elapsed_ms'],'detail':r['detail']})
    after_identity=None if a.legacy else identity(engine,model)
    if loaded!=after_identity:raise AssertionError('Loaded model changed during a cell')
   if sha(model)!=before or sha(a.engine)!=exe_hash:raise AssertionError('Model or binary changed')
   result={'model':model_name,'arm':arm,'repeat':repeat,'correct':sum(r['hit'] for r in rows),'positions':len(rows),
    'model_path':str(model.resolve()),'model_sha256':before,'loaded_model':loaded,'loaded_model_after':after_identity,
    'engine_sha256':exe_hash,'options':ARMS[arm],'milliseconds':100,'threads':1,'seed':1,'uci_advertisement':uci_options,'results':rows}
   payload=json.dumps(result,indent=2,allow_nan=False).encode();(a.out/f'{model_name}-{arm}-r{repeat}.json.gz').write_bytes(gzip.compress(payload,mtime=0))
   summaries.append({k:result[k] for k in ['model','arm','repeat','correct','positions','model_sha256','engine_sha256','milliseconds','threads','seed']})
   save(a.out/'summary.json',summaries);print(summaries[-1],flush=True)
 with (a.out/'summary.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(summaries[0]),lineterminator='\n');w.writeheader();w.writerows(summaries)
 save(a.out/'completed.json',{'cells':len(summaries),'searches':sum(r['positions'] for r in summaries),'all_moves_legal':True,'os':platform.system()})
if __name__=='__main__':main()
