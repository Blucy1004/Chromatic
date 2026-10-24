#!/usr/bin/env python3
"""Reproducible UCI ablations. Never adjudicates unfinished games as draws."""
import argparse,csv,gzip,hashlib,io,json,math,os,platform,queue,statistics,subprocess,threading,time
from pathlib import Path
import chess,chess.pgn
ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT.parent/'project-chromatic-memory/dist/default.zero'
CONFIGS={
 'disabled':{},'counterexample':{'CounterexampleVerify':True},
 'adaptive':{'AdaptiveBudget':True},'forcing':{'ForcingSearch':True},
 'tactical':{'TacticalVerify':True},
 'parallel':{'Threads':2},
 'combination':{'CounterexampleVerify':True,'AdaptiveBudget':True,'ForcingSearch':True,'TacticalVerify':True},
}
# Hand-authored diagnostic positions, not a claim of a representative puzzle set.
TACTICS=[
 ('mate_white','7k/8/5KQ1/8/8/8/8/8 w - - 0 1','mate'),
 ('mate_black','8/8/8/8/8/5kq1/8/7K b - - 0 1','mate'),
 ('back_rank','6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1','mate'),
 ('win_queen_white','6k1/8/8/8/8/8/3q4/3Q2K1 w - - 0 1',['d1d2']),
 ('win_queen_black','3q2k1/3Q4/8/8/8/8/8/6K1 b - - 0 1',['d8d7']),
 ('promotion','8/P6k/8/8/8/8/8/7K w - - 0 1',['a7a8q','a7a8r']),
 ('ep','4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1',['e5d6']),
 ('in_check','4k3/8/8/8/8/8/4r3/4K3 w - - 0 1',['e1e2']),
]
OPENINGS=[
 {'name':'Italian','fen':chess.STARTING_FEN,'moves':['e2e4','e7e5','g1f3','b8c6','f1c4','f8c5']},
 {'name':'QueensGambit','fen':chess.STARTING_FEN,'moves':['d2d4','d7d5','c2c4','e7e6','b1c3','g8f6']},
 {'name':'RookEnding','fen':'8/5pk1/6p1/3R4/8/5KP1/7P/5r2 w - - 0 1','moves':[]},
 {'name':'PawnEnding','fen':'8/5pk1/6p1/3p4/3P4/5KP1/5P1P/8 w - - 0 1','moves':[]},
]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):
 path=Path(path);content=json.dumps(data,indent=2,allow_nan=False)+'\n'
 if path.name.endswith('-games.json'):
  path.with_suffix('.json.gz').write_bytes(gzip.compress(content.encode(),compresslevel=3,mtime=0))
 else:path.write_text(content)
class Engine:
 def __init__(self,path,options=None,model=MODEL):
  self.p=subprocess.Popen([str(path),'--model',str(model)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
  self.lines=queue.Queue();self.trace=[]
  def consume():
   for line in self.p.stdout:self.lines.put(line.strip())
   self.lines.put(None)
  self.reader=threading.Thread(target=consume,daemon=True);self.reader.start()
  self.send('uci');lines=self.until(lambda s:s=='uciok')
  self.options={s.split(' type ')[0][len('option name '):] for s in lines if s.startswith('option name ')}
  self.set({'MateProofNodes':0,**(options or {})});self.send('isready')
  ready=self.until(lambda s:s=='readyok')
  if any('error' in s for s in ready):raise RuntimeError(ready)
 def send(self,s):self.p.stdin.write(s+'\n');self.p.stdin.flush()
 def until(self,predicate,timeout=10):
  end=time.monotonic()+timeout;lines=[]
  while True:
   try:s=self.lines.get(timeout=max(.001,end-time.monotonic()))
   except queue.Empty:raise TimeoutError('UCI timeout')
   if s is None:raise RuntimeError('UCI EOF: '+self.p.stderr.read())
   lines.append(s);self.trace.append(s)
   if predicate(s):return lines
 def set(self,options):
  for name,value in options.items():
   if name not in self.options:raise ValueError('unknown option '+name)
   if isinstance(value,bool):value=str(value).lower()
   self.send(f'setoption name {name} value {value}')
 def position(self,fen,moves=()):self.send('position fen '+fen+(' moves '+' '.join(moves) if moves else ''))
 def search(self,fen,moves=(),ms=30,nodes=None):
  self.position(fen,moves);t=time.perf_counter();self.send('go '+(f'nodes {nodes}' if nodes else f'movetime {ms}'))
  lines=self.until(lambda s:s.startswith('bestmove '),max(5,ms/1000+2));elapsed=(time.perf_counter()-t)*1000
  detail=next((json.loads(s[len('info string overlord '):]) for s in lines if s.startswith('info string overlord ')),None)
  info=next((s for s in lines if s.startswith('info nodes ')),None)
  return {'move':lines[-1].split()[1],'elapsed_ms':elapsed,'detail':detail,'info':info}
 def zero(self,fen,moves=(),nodes=512):
  self.position(fen,moves);self.send(f'zero search {nodes} 2026 0 0')
  return json.loads(self.until(lambda s:s.startswith('{'))[-1])
 def close(self):
  if self.p.poll() is None:
   self.send('quit')
   try:self.p.wait(timeout=5)
   except subprocess.TimeoutExpired:self.p.kill();self.p.wait()
  self.reader.join(timeout=2)
 def __enter__(self):return self
 def __exit__(self,*args):self.close()
def stats(games):
 finished=[g for g in games if g['score'] is not None];units={}
 for g in games:units.setdefault(g['pair'],[]).append(g['score'])
 pairs=[statistics.mean(scores) for scores in units.values() if len(scores)==2 and None not in scores]
 mean=statistics.mean(g['score'] for g in finished) if finished else None
 # Distribution-free bound on bounded opening-pair scores. Unlike a normal
 # interval this retains uncertainty even if every observed pair has one score.
 if pairs:
  margin=math.sqrt(math.log(40)/(2*len(pairs)));interval=[max(0,statistics.mean(pairs)-margin),min(1,statistics.mean(pairs)+margin)]
 else:interval=[0,1]
 return {'games':len(games),'finished':len(finished),'unfinished':len(games)-len(finished),
   'wins':sum(g['score']==1 for g in finished),'draws':sum(g['score']==.5 for g in finished),'losses':sum(g['score']==0 for g in finished),
   'score':mean,'complete_opening_pairs':len(pairs),'paired_score_interval_95_conservative':interval,
   'interval_method':'Hoeffding bound on complete opening pairs; assumes independent units; no population generalization from this fixed suite','elo_claim':None,'promote':False,'decision':'insufficient_evidence'}
def game(candidate,reference,opening,color,ms,maxplies):
 board=chess.Board(opening['fen']);moves=list(opening['moves'])
 for m in moves:board.push_uci(m)
 game=chess.pgn.Game();game.setup(chess.Board(opening['fen']));node=game
 for m in moves:node=node.add_variation(chess.Move.from_uci(m))
 game.headers.update({'Event':'Overlord paired ablation','White':'candidate' if color else 'reference','Black':'reference' if color else 'candidate','TimeControl':f'{ms}ms/move','Opening':opening['name']})
 trace=[];error=None
 candidate.send('ucinewgame');reference.send('ucinewgame')
 while not board.is_game_over(claim_draw=True) and len(trace)<maxplies:
  engine=candidate if board.turn==color else reference
  try:
   result=engine.search(opening['fen'],moves,ms);move=chess.Move.from_uci(result['move'])
   if move not in board.legal_moves:raise ValueError('illegal move '+result['move'])
  except Exception as exc:error=str(exc);break
  trace.append({'ply':len(moves),'candidate':board.turn==color,**result})
  board.push(move);moves.append(move.uci());node=node.add_variation(move)
 outcome=board.outcome(claim_draw=True)
 score=(.5 if outcome.winner is None else float(outcome.winner==color)) if outcome else None
 game.headers['Result']=outcome.result() if outcome else '*'
 game.headers['Termination']=str(outcome.termination.name) if outcome else ('error: '+error if error else 'ply_limit_unfinished')
 return {'score':score,'result':game.headers['Result'],'termination':game.headers['Termination'],'plies':len(trace),'moves':moves,'trace':trace,'error':error},str(game)+'\n\n'
def main():
 if hasattr(os,'sched_getaffinity'):
  os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
 p=argparse.ArgumentParser();p.add_argument('--engine',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
 p.add_argument('--out',type=Path,required=True);p.add_argument('--ms',type=int,default=30);p.add_argument('--max-plies',type=int,default=240)
 p.add_argument('--configs',nargs='+',default=list(CONFIGS));p.add_argument('--skip-matches',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 metadata={'driver_sha256':sha(Path(__file__)), 'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'cpu_affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,'search_seed':1,'platform':platform.platform(),'python':platform.python_version(),'model':str(MODEL),'model_sha256':sha(MODEL),
  'engine_sha256':sha(a.engine),'baseline_sha256':sha(a.baseline),'time_control':f'{a.ms}ms per move','max_plies':a.max_plies,'proof_nodes':0,'configs':CONFIGS,'openings':OPENINGS,'tactics':TACTICS}
 save(a.out/'metadata.json',metadata)
 # Disabled-module exact parity with the unmodified memory-recycling engine.
 parity=[]
 with Engine(a.engine) as e,Engine(a.baseline) as b:
  for opening in OPENINGS:
   x=e.zero(opening['fen'],opening['moves']);y=b.zero(opening['fen'],opening['moves'])
   keys=['bestmove','simulations','value','neural_evals','visits','memory_bytes','arena_nodes','max_depth']
   parity.append({'opening':opening['name'],'equal':all(x[k]==y[k] for k in keys),'candidate':x,'baseline':y})
 save(a.out/'disabled-parity.json',parity)
 if not all(r['equal'] for r in parity):raise AssertionError('disabled search differs from baseline')
 tactics=[];bench=[];summaries={}
 for name in a.configs:
  options=CONFIGS[name]
  with Engine(a.engine,options) as e:
   for label,fen,expected in TACTICS:
    board=chess.Board(fen);r=e.search(fen,ms=a.ms);move=chess.Move.from_uci(r['move'])
    legal=move in board.legal_moves;board.push(move)
    passed=legal and (board.is_checkmate() if expected=='mate' else move.uci() in expected)
    tactics.append({'config':name,'label':label,'passed':passed,**r})
   for opening in OPENINGS:
    for repeat in range(3):bench.append({'config':name,'opening':opening['name'],'repeat':repeat,**e.search(opening['fen'],opening['moves'],a.ms)})
  save(a.out/'tactics.json',tactics);save(a.out/'search-bench.json',bench)
  if a.skip_matches:continue
  games=[]
  # Old engine has one search worker. The two-worker shared-tree arm instead
  # compares with two independent original-algorithm trees (equal worker count
  # and total tree budget), explicitly a distinct control in the metadata.
  parallel=name=='parallel';refpath=a.engine if parallel else a.baseline
  refopts={'Threads':2,'SharedTree':False} if parallel else {}
  with Engine(a.engine,options) as e,Engine(refpath,refopts) as ref, (a.out/f'{name}.pgn').open('w') as pgn:
   for index,opening in enumerate(OPENINGS):
    for color in [chess.WHITE,chess.BLACK]:
     r,text=game(e,ref,opening,color,a.ms,a.max_plies);r.update({'pair':index,'opening':opening['name'],'candidate_color':'white' if color else 'black','reference':'two_independent_legacy_trees' if parallel else 'original_memory_engine','threads_per_side':2 if parallel else 1})
     games.append(r);pgn.write(text);pgn.flush();save(a.out/f'{name}-games.json',games)
     print(name,opening['name'],r['candidate_color'],r['result'],r['termination'],flush=True)
  summaries[name]=stats(games);save(a.out/'summary.json',summaries)
 with (a.out/'summary.csv').open('w',newline='') as f:
  fields=['config','games','finished','unfinished','wins','draws','losses','score','complete_opening_pairs','promote']
  writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
  for k,v in summaries.items():writer.writerow({'config':k,**v})
 save(a.out/'tactics-summary.json',{name:{'passed':sum(r['passed'] for r in tactics if r['config']==name),'total':sum(r['config']==name for r in tactics)} for name in a.configs})
if __name__=='__main__':main()
