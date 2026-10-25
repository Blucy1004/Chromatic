"""Validate all model/checkpoint bindings and safe journal resume before research."""
import argparse,hashlib,json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
import psutil,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'research'))
from audit_models import export_digest
from run import Runner,Paused,MODELS,identity,Engine,options,sha,load_positions,OLD,HOLDOUT

def main():
 p=argparse.ArgumentParser();p.add_argument('--engine',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();catalog={}
 for name,path in MODELS.items():
  pt=path.with_suffix('.pt') if name!='zero128' else path.parent/'model.pt'
  state=torch.load(pt,map_location='cpu',weights_only=False);digest=export_digest(state);assert digest==sha(path)
  with Engine(a.engine,options(0),model=path) as e:loaded=identity(e,path)
  catalog[name]={'model_path':str(path.resolve()),'sha256':digest,'pt_path':str(pt.resolve()),'pt_sha256':sha(pt),'architecture':state['config'],
   'checkpoint_format':state['format'],'updates':state['step'],'training_positions':128 if name=='zero128' else state['progress']['positions_processed'],
   'initial_checkpoint_sha256':state.get('initial_model_sha256'),'loaded_tensors':loaded}
 old=load_positions(OLD);new=load_positions(HOLDOUT);assert len(new)==200 and not ({r['id'] for r in old}&{r['id'] for r in new});assert not ({r['game'] for r in old}&{r['game'] for r in new})
 for row in new:assert not __import__('chess').Board(row['fen']).is_game_over(claim_draw=True)
 affinity=psutil.Process().cpu_affinity()
 with tempfile.TemporaryDirectory() as tmp:
  args=SimpleNamespace(out=Path(tmp),engine=a.engine,max_seconds=60,stop_after=2)
  r=Runner(args)
  try:r.cell('journal','zero128',15,'nodes',8,r.positions[:3])
  except Paused:pass
  finally:r.lock.unlink(missing_ok=True)
  journal=next((Path(tmp)/'journal').glob('*.jsonl'))
  with journal.open('ab') as f:f.write(b'{"interrupted":')
  args.stop_after=0;r=Runner(args)
  try:
   rows=r.cell('journal','zero128',15,'nodes',8,r.positions[:3]);assert len(rows)==3 and len({x['id'] for x in rows})==3
   assert list((Path(tmp)/'journal').glob('*.interrupted-tail'))
   try:Runner(args)
   except RuntimeError:pass
   else:raise AssertionError('Concurrent writer was accepted')
  finally:r.lock.unlink(missing_ok=True);psutil.Process().cpu_affinity(affinity)
 a.out.write_text(json.dumps({'models':catalog,'resume_test':'partial record recovered, 3 unique records, matching identities, concurrent writer rejected','holdout_disjoint':True},indent=2)+'\n')
 print('Four model bindings, C++ loading, 200 disjoint puzzles and interruption/resume verified')
if __name__=='__main__':main()
