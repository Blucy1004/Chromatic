"""Two bounded CPU residual-MLP configurations; no strength claim from model size."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zero.train import train
from zero.verify import verify
from zero.data import atomic_json
from bench import benchmark

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--engine',required=True);p.add_argument('--bench',required=True)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=200)
    args=p.parse_args();root=Path(args.output);reports={}
    if not 1<=args.steps<=1000:raise ValueError('scaling experiment must be bounded to 1..1000 steps')
    for name,width,blocks in [('tiny',64,4),('small',96,6)]:
        directory=root/name;directory.mkdir(parents=True,exist_ok=True);model=directory/'model.pt'
        training=train([args.data],model,steps=args.steps,batch_size=32,save_every=100,
                       width=width,blocks=blocks,seed=2026,validation_fraction=.15)
        variants={}
        for backend in ('Scalar','SIMD'):
            parity=verify(args.engine,model,directory/(backend.lower()+'-parity.json'),positions=100,backend=backend)
            timing=benchmark(args.bench,model.with_suffix('.zero'),directory/(backend.lower()+'-bench.json'),
                             repeats=3,evaluations=1000,simulations=128,backend=backend.lower())
            variants[backend]={'parity':parity,'medians':timing['medians']}
        reports[name]={'architecture':'residual_mlp','width':width,'residual_blocks':blocks,
                       'training':training,'inference':variants,'strength_evidence':'not inferred from size or loss'}
    atomic_json(root/'comparison.json',reports)
    print(json.dumps({key:{'width':r['width'],'blocks':r['residual_blocks'],
                          'scalar_nps':r['inference']['Scalar']['medians']['nodes_per_second'],
                          'simd_nps':r['inference']['SIMD']['medians']['nodes_per_second'],
                          'validation':r['training']['final']['validation']} for key,r in reports.items()},indent=2))
