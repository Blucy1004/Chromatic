"""Fixed-work observer-effect and historical-engine behavior check."""
import argparse,json,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'research'));sys.path.insert(0,str(ROOT/'tools'))
from run import Engine,MODELS,options,load_positions,OLD,sha

def execute(e,p,old=False,nodes=128):
 e.position(p['fen']);e.send(f'zero search {nodes} 1 0 0' if old else f'zero bench nodes {nodes} 1')
 return json.loads(e.until(lambda s:s.startswith('{'),30)[-1])
def main():
 p=argparse.ArgumentParser();p.add_argument('--engine',type=Path,required=True);p.add_argument('--old',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();positions=load_positions(OLD);results=[]
 keys=['bestmove','simulations','value','neural_evals','visits','memory_bytes','arena_nodes','max_depth']
 for model in ['zero128','lichess10000']:
  for mask in range(32):
   old_opts=options(mask);old_opts.pop('ResearchMetrics')
   with Engine(a.old,old_opts,model=MODELS[model]) as old,Engine(a.engine,options(mask),model=MODELS[model]) as new:
    for index in [0,17]:
     x=execute(old,positions[index],True);y=execute(new,positions[index]);assert all(x[k]==y[k] for k in keys),(model,mask,index)
   results.append({'model':model,'mask':mask,'positions_checked':2,'identical_fixed_work':True})
 overhead=[]
 for model in ['zero128','lichess10000']:
  for mask in [0,15,16,31]:
   with Engine(a.engine,options(mask),model=MODELS[model]) as e:
    for repeat in range(3):
     for index in [0,17,41,73]:
      measurements={};records={}
      for enabled in ([False,True] if repeat%2==0 else [True,False]):
       e.set({'ResearchMetrics':enabled});r=execute(e,positions[index],nodes=512);records[enabled]=r;measurements[enabled]=r['search_ns']
      assert all(records[False][k]==records[True][k] for k in keys)
      overhead.append({'model':model,'mask':mask,'puzzle':positions[index]['id'],'repeat':repeat,'off_ns':measurements[False],'on_ns':measurements[True],'ratio':measurements[True]/max(1,measurements[False])})
 a.out.write_text(json.dumps({'old_engine_sha256':sha(a.old),'research_engine_sha256':sha(a.engine),'fixed_work_parity':results,'observer_effect':overhead,'scope':'512 completed visits, no wall cutoff; search call includes cleanup/PV metrics but excludes JSON serialization'},indent=2)+'\n')
 print('128 historical comparisons and 96 paired observer-effect checks passed')
if __name__=='__main__':main()
