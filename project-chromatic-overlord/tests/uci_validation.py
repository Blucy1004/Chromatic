"""Black-box UCI/history/inference checks against python-chess."""
import argparse,json,random,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from experiment import Engine,MODEL,save
import chess,chess.pgn

def run(engine,report):
 results=[]
 with Engine(engine) as e:
  for name in ['CounterexampleVerify','AdaptiveBudget','ForcingSearch','TacticalVerify','SharedTree','Threads']:
   assert name in e.options,name
  rng=random.Random(2026);board=chess.Board();moves=[]
  for index in range(100):
   if board.is_game_over(claim_draw=True):board=chess.Board();moves=[]
   e.position(chess.STARTING_FEN,moves);e.send('zero inspect')
   r=json.loads(e.until(lambda s:s.startswith('{'))[-1])
   assert {m['uci'] for m in r['legal']}=={m.uci() for m in board.legal_moves}
   assert r['repetitions']==(3 if board.is_repetition(3) else 2 if board.is_repetition(2) else 1)
   chosen=rng.choice(list(board.legal_moves));board.push(chosen);moves.append(chosen.uci())
  results.append({'name':'100 seeded legal/history differential positions','passed':True})
  e.position(chess.STARTING_FEN);e.send('zero eval');full=json.loads(e.until(lambda s:s.startswith('{'))[-1])
  e.send('zero evallegal');sparse=json.loads(e.until(lambda s:s.startswith('{'))[-1])
  assert all(full['policy'][m['action']]==sparse['policy'][m['action']] for m in full['legal'])
  e.set({'InferenceBackend':'Scalar'});e.send('zero evallegal');scalar=json.loads(e.until(lambda s:s.startswith('{'))[-1])
  error=max(abs(scalar['policy'][m['action']]-sparse['policy'][m['action']]) for m in full['legal'])
  assert error<1e-3,error
  results.append({'name':'existing model sparse exactness / scalar-SIMD','passed':True,'max_logit_error':error})
  e.set({'InferenceBackend':'SIMD','Threads':4,'CounterexampleVerify':True,'ForcingSearch':True,'TacticalVerify':True})
  for ms in [1,5,25,100]:
   r=e.search(chess.STARTING_FEN,ms=ms)
   assert chess.Move.from_uci(r['move']) in chess.Board().legal_moves
   assert r['elapsed_ms']<ms+500,r
   results.append({'name':f'parallel all modules movetime {ms}','passed':True,'elapsed_ms':r['elapsed_ms']})
  e.position(chess.STARTING_FEN);t=time.perf_counter();e.send('go infinite');time.sleep(.03);e.send('stop');e.until(lambda s:s.startswith('bestmove '),2)
  results.append({'name':'parallel go infinite / stop','passed':True,'elapsed_ms':1000*(time.perf_counter()-t)})
  e.send('isready');e.until(lambda s:s=='readyok')
  e.send('setoption name CounterexampleVerify value invalid');e.send('isready');r=e.until(lambda s:s=='readyok')
  assert any('requires true or false' in s for s in r)
  e.set({'CounterexampleVerify':False,'ForcingSearch':False,'TacticalVerify':False,'Threads':2,'SharedTree':False})
  r=e.search(chess.STARTING_FEN,nodes=101);assert r['detail']['simulations']==101
  assert sum(v['n'] for v in r['detail']['visits'])==101
  results.append({'name':'two-worker independent reference exact aggregate N','passed':True})
  repeat=['g1f3','g8f6','f3g1','f6g8']*2
  r=e.search(chess.STARTING_FEN,repeat,nodes=32);assert r['move']=='0000'
  results.append({'name':'UCI history threefold terminal','passed':True})
 save(report,{'passed':True,'checks':results})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--engine',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args();run(a.engine,a.report)
