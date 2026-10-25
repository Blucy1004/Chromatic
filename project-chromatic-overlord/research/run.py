#!/usr/bin/env python3
"""Bounded, resumable factorial research. Every cell has an immutable identity."""
import argparse,csv,gzip,hashlib,itertools,json,os,platform,random,statistics,struct,subprocess,sys,time
from pathlib import Path
import chess,psutil
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parent
sys.path.insert(0,str(ROOT/'tools'))
from experiment import Engine,game,sha,save
MODELS={'zero128':REPO/'chromatic-shard-trainer/champion/default.zero',**{f'lichess{n}':REPO/f'chromatic-shard-trainer/candidates/positions-{n}.zero' for n in [100,1000,10000]}}
CORE=['zero128','lichess10000'];BITS=['C','A','F','T','P']
OPENINGS=[s.split() for s in [
'e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6','e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4 c8f5',
'e2e4 d7d6 d2d4 g8f6 b1c3 g7g6','d2d4 g8f6 c2c4 e7e6 b1c3 f8b4',
'd2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6','d2d4 g8f6 c2c4 g7g6 b1c3 d7d5',
'd2d4 g8f6 c2c4 e7e6 g2g3 d7d5 f1g2','e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4']]
def options(mask,metrics=True):
 return {'MateProofNodes':0,'Threads':2 if mask&16 else 1,'SharedTree':True,'CPuct':1.5,'FPUReduction':0,
 'InferenceBackend':'SIMD','SparsePolicyHead':True,'TreeMemoryMB':64,'MaxNodes':100000,
 'VerificationNodes':2048,'VerificationDepth':4,'VerificationCandidates':4,'SearchTelemetry':False,
 'CounterexampleVerify':bool(mask&1),'AdaptiveBudget':bool(mask&2),'ForcingSearch':bool(mask&4),'TacticalVerify':bool(mask&8),'ResearchMetrics':metrics}
def load_positions(path):
 rows=[]
 with path.open(newline='') as f:
  for row in csv.DictReader(f):
   b=chess.Board(row['FEN']);moves=row['Moves'].split();b.push_uci(moves[0]);expected=moves[1]
   if not b.is_valid() or chess.Move.from_uci(expected) not in b.legal_moves:raise ValueError('invalid puzzle')
   rows.append({'id':row['PuzzleId'],'fen':b.fen(en_passant='fen'),'expected':expected,'rating':int(row['Rating']),
    'game':row['GameUrl'].split('#')[0].replace('/black','').replace('/white','')})
 return rows
OLD=REPO/'chromatic-arc-trainer/fixtures/official-puzzles.csv';HOLDOUT=ROOT/'research/fixtures/holdout.csv'
def atomic(path,data):save(Path(path),data)
def read(path):return json.loads(Path(path).read_text())
def identity(e,model):
 e.send('zero model');r=json.loads(e.until(lambda s:s.startswith('{'))[-1]);assert r['loaded'] and r['sha256']==sha(model);return r
def bench(e,p,mode,budget):
 e.position(p['fen']);before=psutil.Process(e.p.pid).cpu_times();start=time.perf_counter_ns()
 e.send(f'zero bench {mode} {budget} 1');r=json.loads(e.until(lambda s:s.startswith('{'),max(30,budget/1000+5))[-1]);wall=time.perf_counter_ns()-start
 after=psutil.Process(e.p.pid).cpu_times()
 if 'error' in r:raise RuntimeError(r['error'])
 move=chess.Move.from_uci(r['bestmove']);assert move in chess.Board(p['fen']).legal_moves
 return {**p,'hit':r['bestmove']==p['expected'],'wall_ns':wall,'cpu_seconds':after.user+after.system-before.user-before.system,
  'rss_after_bytes':psutil.Process(e.p.pid).memory_info().rss,'result':r}
class Paused(Exception):pass
class Runner:
 def __init__(self,a):
  self.a=a;self.out=a.out;self.out.mkdir(parents=True,exist_ok=True);self.started=time.monotonic();self.new=0
  self.positions=load_positions(OLD);self.holdout=load_positions(HOLDOUT);assert len(self.positions)==100 and len(self.holdout)==200
  allowed=psutil.Process().cpu_affinity();existing=read(self.out/'manifest.json') if (self.out/'manifest.json').exists() else None
  self.cpus=existing['engine_cpu_affinity'] if existing else allowed[:2]
  self.driver_cpu=existing['driver_cpu_affinity'] if existing else ([allowed[-1]] if len(allowed)>2 else allowed)
  psutil.Process().cpu_affinity(self.driver_cpu)
  self.manifest={'schema':1,'os':platform.system(),'platform':platform.platform(),'processor':platform.processor(),'host':platform.node(),'logical_cpus':psutil.cpu_count(),'runner_sha256':sha(Path(__file__)),'engine_path':str(a.engine.resolve()),'engine_sha256':sha(a.engine),
  'cpp_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for d in ['src','include'] for p in sorted((ROOT/d).rglob('*')) if p.is_file()},
  'models':{k:{'path':str(p.resolve()),'sha256':sha(p),'architecture':list(struct.unpack('<4I',p.read_bytes()[8:24]))} for k,p in MODELS.items()},
  'old_fixture_sha256':sha(OLD),'holdout_sha256':sha(HOLDOUT),'seed':1,'cell_shuffle_seed':20261025,'engine_cpu_affinity':self.cpus,'driver_cpu_affinity':self.driver_cpu,
  'budget_seconds':a.max_seconds,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()}
  path=self.out/'manifest.json'
  if path.exists():
   old=read(path)
   for key in ['engine_sha256','cpp_sha256','models','old_fixture_sha256','holdout_sha256','seed','engine_cpu_affinity','host','processor','runner_sha256']:
    if old[key]!=self.manifest[key]:raise ValueError('Resume identity mismatch: '+key)
   self.manifest=old
  else:atomic(path,self.manifest)
  self.lock=self.out/'.runner.lock'
  if self.lock.exists():
   owner=read(self.lock);live=owner.get('host')==platform.node() and psutil.pid_exists(owner.get('pid',-1))
   if live:
    try:live=abs(psutil.Process(owner['pid']).create_time()-owner['born'])<.01
    except psutil.Error:live=False
   if live:raise RuntimeError('Another runner owns this experiment')
   self.lock.unlink()
  with self.lock.open('x') as f:json.dump({'pid':os.getpid(),'born':psutil.Process().create_time(),'host':platform.node()},f)
  self.used=read(self.out/'usage.json').get('active_seconds',0) if (self.out/'usage.json').exists() else 0
 def checkpoint(self):
  elapsed=self.used+time.monotonic()-self.started;atomic(self.out/'usage.json',{'active_seconds':elapsed,'records_added_this_process':self.new})
  if elapsed>self.a.max_seconds or self.a.stop_after and self.new>=self.a.stop_after:raise Paused('bounded checkpoint')
 def event(self,kind,**kw):
  with (self.out/'events.jsonl').open('a') as f:f.write(json.dumps({'event':kind,'utc_epoch':time.time(),**kw})+'\n');f.flush();os.fsync(f.fileno())
 def rows(self,path):
  gz=path.with_suffix('.jsonl.gz')
  if gz.exists():return [json.loads(x) for x in gzip.decompress(gz.read_bytes()).decode().splitlines()]
  if not path.exists():return []
  raw=path.read_bytes();lines=raw.splitlines(keepends=True);good=[];offset=0
  for i,line in enumerate(lines):
   try:row=json.loads(line)
   except json.JSONDecodeError:
    if i!=len(lines)-1:raise ValueError('Corrupt middle journal')
    path.with_suffix('.interrupted-tail').write_bytes(raw[offset:]);path.write_bytes(raw[:offset]);break
   good.append(row);offset+=len(line)
  return good
 def cell(self,stage,model,mask,mode,budget,positions,repeat=0,extra=None):
  directory=self.out/stage;directory.mkdir(parents=True,exist_ok=True)
  key=f'{model}-m{mask:02d}-{mode}{budget}-r{repeat}'+('-'+extra['label'] if extra else '')
  path=directory/(key+'.jsonl');meta=directory/(key+'.meta.json');rows=self.rows(path);opts=options(mask)
  if extra:opts.update(extra.get('options',{}))
  expected={'model':model,'mask':mask,'mode':mode,'budget':budget,'repeat':repeat,'options':opts,'ids':[p['id'] for p in positions],
   'model_sha256':sha(MODELS[model]),'engine_sha256':self.manifest['engine_sha256']}
  if meta.exists():
   old=read(meta)
   for k,v in expected.items():
    if old[k]!=v:raise ValueError('Cell identity mismatch '+key+' '+k)
  else:atomic(meta,expected)
  if [r['id'] for r in rows]!=expected['ids'][:len(rows)]:raise ValueError('Journal ordering/duplicates')
  if len(rows)==len(positions):return rows
  self.event('cell_start',stage=stage,key=key,remaining=len(positions)-len(rows));start=time.monotonic();prefix_rows=len(rows)
  with Engine(self.a.engine,opts,model=MODELS[model]) as e:
   psutil.Process(e.p.pid).cpu_affinity(self.cpus);loaded=identity(e,MODELS[model]);assert loaded['sha256']==expected['model_sha256']
   atomic(meta,{**expected,'loaded_model':loaded})
   e.position(chess.STARTING_FEN)
   for _ in range(3):e.send('zero evallegal');e.until(lambda s:s.startswith('{'))
   with path.open('a') as f:
    for p in positions[len(rows):]:
     self.checkpoint();row=bench(e,p,mode,budget);row.update({'model':model,'mask':mask,'mode':mode,'budget':budget,'repeat':repeat,'cell':key,'stage':stage})
     f.write(json.dumps(row,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno());rows.append(row);self.new+=1
   assert identity(e,MODELS[model])==loaded and sha(self.a.engine)==expected['engine_sha256']
  payload=path.read_bytes();compressed=gzip.compress(payload,compresslevel=3,mtime=0);path.with_suffix('.jsonl.gz').write_bytes(compressed);path.unlink()
  atomic(meta,{**expected,'loaded_model':loaded,'loaded_model_after':loaded,'resumed_prefix_rows':prefix_rows,'complete':True,'raw_gzip_sha256':hashlib.sha256(compressed).hexdigest()})
  self.event('cell_complete',stage=stage,key=key,seconds=time.monotonic()-start,hits=sum(r['hit'] for r in rows));self.checkpoint()
  print(stage,key,len(rows),sum(r['hit'] for r in rows),flush=True);return rows
 def subset(self,n):
  indices=sorted(range(100),key=lambda i:(self.positions[i]['rating'],i));return [self.positions[indices[min(99,int((j+.5)*100/n))]] for j in range(n)]
 def screen(self):
  cells=list(itertools.product(MODELS,range(32)));random.Random(20261025).shuffle(cells)
  for model,mask in cells:self.cell('screen',model,mask,'time',100,self.positions)
  for model in MODELS:self.cell('legacy_proof',model,0,'time',100,self.positions,extra={'label':'proof256','options':{'MateProofNodes':256}})
 def selection(self):
  scores={};cost={}
  for model in MODELS:
   for mask in range(32):
    rows=self.rows(self.out/'screen'/f'{model}-m{mask:02d}-time100-r0.jsonl');assert len(rows)==100
    scores[model,mask]=sum(r['hit'] for r in rows);cost[model,mask]=sum(r['cpu_seconds'] for r in rows)
  best={model:min(range(16),key=lambda m:(-scores[model,m],cost[model,m],m.bit_count(),m)) for model in MODELS}
  pairs=list(itertools.combinations(range(4),2))
  def synergy(pair):
   a,b=map(lambda x:1<<x,pair);return sum(scores[m,a|b]-scores[m,a]-scores[m,b]+scores[m,0] for m in CORE)
  positive=max(pairs,key=lambda p:(synergy(p),tuple(-x for x in p)));negative=min(pairs,key=lambda p:(synergy(p),p))
  primary=(1<<positive[0])|(1<<positive[1]);bad=(1<<negative[0])|(1<<negative[1])
  rank=sorted(range(32),key=lambda mask:(-sum(scores[m,mask] for m in CORE),sum(cost[m,mask] for m in CORE),mask))
  stage2=list(dict.fromkeys([0,1,2,4,8,15]+rank))[:8]
  plan={'schema':1,'old_fixture_sha256':sha(OLD),'holdout_sha256':sha(HOLDOUT),'cpp_sha256':self.manifest['cpp_sha256'],
   'selection_os':'Linux','runner_sha256':self.manifest['runner_sha256'],'best_serial':best,'primary_pair':list(positive),'primary_pair_mask':primary,'negative_pair_mask':bad,
   'stage2_masks':stage2,'screen_hits':{m:{str(k):scores[m,k] for k in range(32)} for m in MODELS},
   'criterion':'screen-only: best serial accuracy then measured CPU cost; one primary serial pair selected by summed core-model synergy; holdout is never consulted'}
  atomic(self.out/'selection-plan.json',plan);return plan
 def get_plan(self):
  if self.a.plan and self.a.plan.exists():return read(self.a.plan)
  if platform.system()=='Linux':return self.selection()
  # Keep the same native host while waiting for the independently chosen plan.
  deadline=time.monotonic()+900
  while time.monotonic()<deadline:
   try:
    data=subprocess.check_output(['gh','api','repos/Blucy1004/Chromatic/contents/project-chromatic-overlord/research/selection-plan.json?ref=operation-overlord','--jq','.content'],stderr=subprocess.DEVNULL)
    import base64
    plan=json.loads(base64.b64decode(data));assert plan['cpp_sha256']==self.manifest['cpp_sha256'] and plan['holdout_sha256']==sha(HOLDOUT) and plan['runner_sha256']==self.manifest['runner_sha256']
    atomic(self.out/'selection-plan.json',plan);return plan
   except (subprocess.CalledProcessError,ValueError,AssertionError,json.JSONDecodeError):time.sleep(15)
  raise Paused('Linux selection plan was not published within 15min')
 def publish_plan(self,plan):
  if not self.a.publish_plan:return
  path=ROOT/'research/selection-plan.json';atomic(path,plan)
  if subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).strip():raise RuntimeError('Unrelated staged changes prevent plan commit')
  subprocess.run(['git','add',str(path.relative_to(REPO))],cwd=REPO,check=True)
  if subprocess.run(['git','diff','--cached','--quiet','--',str(path.relative_to(REPO))],cwd=REPO).returncode==0:return
  subprocess.run(['git','commit','-m','Lock factorial follow-up selection before holdout [skip ci]'],cwd=REPO,check=True)
  subprocess.run(['git','push','origin','HEAD:operation-overlord'],cwd=REPO,check=True)
 def followup(self,plan):
  for model in CORE:
   for mask in sorted(set([0,8,15,plan['best_serial'][model]])):
    for repeat in [1,2]:self.cell('anchors',model,mask,'time',100,self.positions,repeat)
  for model in MODELS:
   for mask in plan['stage2_masks']:self.cell('one_second',model,mask,'time',1000,self.subset(24))
  for model in CORE:
   masks=sorted(set([0,15,plan['best_serial'][model],plan['negative_pair_mask']]))
   for mask in masks:self.cell('five_seconds',model,mask,'time',5000,self.subset(16))
   bestrows=self.rows(self.out/'screen'/f'{model}-m{plan["best_serial"][model]:02d}-time100-r0.jsonl')
   base=self.rows(self.out/'screen'/f'{model}-m00-time100-r0.jsonl')
   selected=[i for i in range(100) if base[i]['hit'] and not bestrows[i]['hit']][:2]+[i for i in range(100) if not base[i]['hit'] and bestrows[i]['hit']][:2]
   selected=list(dict.fromkeys(selected+list(range(100))))[:4]
   for mask in sorted(set([0,15,plan['best_serial'][model]])):self.cell('ten_seconds',model,mask,'time',10000,[self.positions[i] for i in selected])
  for mode in ['nodes','nn']:
   for model in MODELS:
    for mask in range(32):self.cell('fixed_'+mode,model,mask,mode,512,self.subset(24))
  for model in MODELS:
   masks=sorted(set([0,15,plan['best_serial'][model],1<<plan['primary_pair'][0],1<<plan['primary_pair'][1],plan['primary_pair_mask']]))
   for mask in masks:self.cell('holdout',model,mask,'time',100,self.holdout)
 def diagnostics(self):
  for model in MODELS:
   path=self.out/f'numeric-{model}.jsonl.gz'
   if path.exists():continue
   rows=[]
   with Engine(self.a.engine,options(0),model=MODELS[model]) as e:
    psutil.Process(e.p.pid).cpu_affinity(self.cpus);loaded=identity(e,MODELS[model])
    for backend in ['SIMD','Scalar']:
     e.set({'InferenceBackend':backend})
     for p in self.positions:
      e.position(p['fen']);e.send('zero evallegal');r=json.loads(e.until(lambda x:x.startswith('{'))[-1])
      rows.append({'id':p['id'],'backend':backend,'wdl':r['wdl'],'legal':[x['uci'] for x in r['legal']],
       'logits':[r['policy'][x['action']] for x in r['legal']],'loaded_sha256':loaded['sha256']})
   path.write_bytes(gzip.compress(('\n'.join(json.dumps(r) for r in rows)+'\n').encode(),mtime=0))
   # Scalar fixed-work probes discriminate numeric/tie sensitivity from timing.
   for mask in [0,15,16,31]:self.cell('scalar_fixed',model,mask,'nodes',512,self.subset(24),extra={'label':'scalar','options':{'InferenceBackend':'Scalar'}})
 def matches(self,plan):
  directory=self.out/'matches';directory.mkdir(exist_ok=True)
  series=[(m,plan['best_serial'][m]) for m in CORE]
  if plan['best_serial']['zero128']!=15:series.append(('zero128',15))
  for model,mask in series:
   target=directory/f'{model}-m{mask:02d}.json';records=read(target) if target.exists() else []
   with Engine(self.a.engine,options(mask),model=MODELS[model]) as candidate,Engine(self.a.engine,options(0),model=MODELS[model]) as baseline:
    for e in [candidate,baseline]:psutil.Process(e.p.pid).cpu_affinity(self.cpus);identity(e,MODELS[model])
    for pair,moves in enumerate(OPENINGS):
     for color in [chess.WHITE,chess.BLACK]:
      index=pair*2+int(not color)
      if index<len(records):continue
      self.checkpoint();opening={'name':f'factorial-{pair}','fen':chess.STARTING_FEN,'moves':moves}
      result,pgn=game(candidate,baseline,opening,color,100,200)
      result.update(pair=pair,model=model,mask=mask,candidate_white=color,pgn=pgn,model_sha256=sha(MODELS[model]),engine_sha256=self.manifest['engine_sha256'],threads=1,movetime_ms=100)
      records.append(result);atomic(target,records)
      target.with_suffix('.pgn').write_text(''.join(x['pgn'] for x in records).rstrip()+'\n')
      self.event('match_complete',model=model,mask=mask,pair=pair,color=color,result=result['result']);print('match',model,mask,pair,color,result['result'],flush=True)
 def pipeline(self):
  self.event('pipeline_start');self.diagnostics();self.screen();plan=self.get_plan();self.publish_plan(plan);self.followup(plan);self.matches(plan);self.checkpoint();self.event('pipeline_complete');atomic(self.out/'completed.json',{'completed':True,'active_seconds':self.used+time.monotonic()-self.started})
def main():
 p=argparse.ArgumentParser();p.add_argument('--engine',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',choices=['pipeline','screen','followup','diagnostics','matches'],default='pipeline');p.add_argument('--plan',type=Path);p.add_argument('--publish-plan',action='store_true');p.add_argument('--max-seconds',type=int,default=5400);p.add_argument('--stop-after',type=int,default=0);p.add_argument('--require-windows',action='store_true');a=p.parse_args()
 if a.require_windows and platform.system()!='Windows':raise RuntimeError('Native Windows required')
 runner=Runner(a)
 try:
  if a.stage=='pipeline':runner.pipeline()
  elif a.stage=='screen':runner.screen()
  elif a.stage=='diagnostics':runner.diagnostics()
  else:getattr(runner,a.stage)(runner.get_plan())
 except Paused as exc:runner.event('paused',reason=str(exc));print('PAUSED:',exc,flush=True);sys.exit(75)
 finally:
  if runner.lock.exists():runner.lock.unlink()
if __name__=='__main__':main()
