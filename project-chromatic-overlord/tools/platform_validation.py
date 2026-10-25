#!/usr/bin/env python3
"""Fail closed on wrong OS, wrong model, protocol failures or missing recycling."""
import argparse,json,platform,struct,subprocess,time
from pathlib import Path
import chess
from experiment import Engine,ROOT,sha,save
from model_matrix import MODELS,ARMS,identity

def main():
 p=argparse.ArgumentParser();p.add_argument('--engine',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--require-windows',action='store_true');a=p.parse_args()
 if a.require_windows and (platform.system()!='Windows' or struct.calcsize('P')!=8):raise RuntimeError('Requires native Windows x64')
 a.out.mkdir(parents=True,exist_ok=True)
 result={'os':platform.system(),'platform':platform.platform(),'native_windows':platform.system()=='Windows','engine_sha256':sha(a.engine),'checks':[],'passed':False}
 def record(name,**data):result['checks'].append({'name':name,'passed':True,**data});save(a.out/'protocol.json',result)
 for model_name,model in MODELS.items():
  with Engine(a.engine,ARMS['disabled'],model=model) as e:
   record('loaded_model_'+model_name,identity=identity(e,model),file_sha256=sha(model))
   e.send('isready');e.until(lambda s:s=='readyok')
   record('uci_initialization_'+model_name)
   for ms in [100,1000,10000]:
    r=e.search(chess.STARTING_FEN,ms=ms)
    if chess.Move.from_uci(r['move']) not in chess.Board().legal_moves:raise AssertionError('illegal timed move')
    if r['elapsed_ms']>ms+1000:raise AssertionError('excessive time overrun')
    record(f'timed_{model_name}_{ms}ms',requested_ms=ms,elapsed_ms=r['elapsed_ms'],move=r['move'])
   for name in ['CounterexampleVerify','AdaptiveBudget','ForcingSearch','TacticalVerify']:
    for enabled in [True,False]:
     e.set({name:enabled});r=e.search(chess.STARTING_FEN,ms=100)
     if chess.Move.from_uci(r['move']) not in chess.Board().legal_moves:raise AssertionError('illegal option move')
     record(f'{model_name}_{name}_{enabled}',elapsed_ms=r['elapsed_ms'])
   for threads in [2,1]:
    e.set({'Threads':threads});r=e.search(chess.STARTING_FEN,ms=100)
    if threads==2 and r['detail']['peak_inflight']<2:raise AssertionError('no shared-tree reservations')
    record(f'{model_name}_Threads_{threads}',detail=r['detail'])
   e.set({'Threads':2,'MaxNodes':16,'TreeMemoryMB':1,'SearchTelemetry':True})
   r=e.search(chess.STARTING_FEN,nodes=1024);d=r['detail']
   if d['prune_count']<=0 or d['simulations']!=1024:raise AssertionError('recycling did not execute')
   if any(not x['root_statistics_preserved'] for x in d['telemetry']['prunes']):raise AssertionError('root stats lost')
   record('recycling_'+model_name,prune_count=d['prune_count'],nodes_evicted=d['nodes_evicted'],memory_bytes=d['memory_bytes'])
   e.set({'SearchTelemetry':False,'MaxNodes':100000,'TreeMemoryMB':64,**ARMS['combination'],'Threads':2})
   r=e.search(chess.STARTING_FEN,ms=100);record('all_options_parallel_'+model_name,detail=r['detail'])
   e.position(chess.STARTING_FEN);e.send('go infinite');time.sleep(.1);t=time.perf_counter();e.send('stop')
   e.until(lambda s:s.startswith('bestmove '),3);record('stop_'+model_name,stop_latency_ms=1000*(time.perf_counter()-t))
   e.position(chess.STARTING_FEN);e.send('go infinite');time.sleep(.05)
   e.send('quit');e.p.wait(timeout=3)
   if e.p.returncode!=0:raise AssertionError('quit failed')
   record('quit_during_search_'+model_name,returncode=e.p.returncode)
 for depth,expected in [(3,8902),(4,197281),(5,4865609)]:
  r=subprocess.run([str(a.engine),'--perft',str(depth)],text=True,capture_output=True,check=True,timeout=60)
  if int(r.stdout.strip())!=expected:raise AssertionError('perft mismatch')
  record(f'perft_{depth}',nodes=expected)
 result['passed']=True;save(a.out/'protocol.json',result)
 print(json.dumps({'passed':True,'checks':len(result['checks']),'os':result['os']}))
if __name__=='__main__':main()
