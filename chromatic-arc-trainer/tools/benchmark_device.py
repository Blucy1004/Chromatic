"""Run a bounded encoded-batch benchmark, without downloading or replacing a model."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from shardtrain.benchmarks import run_matrix,compile_environment
from shardtrain.storage import atomic_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initial-checkpoint',type=Path,default=ROOT/'champion/model.pt')
    p.add_argument('--records',type=Path,default=ROOT/'fixtures/benchmark-records.jsonl')
    p.add_argument('--output',type=Path,default=ROOT/'reports/benchmarks')
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--steps',type=int,default=15)
    p.add_argument('--warmup',type=int,default=2);p.add_argument('--deadline-seconds',type=float,default=240)
    p.add_argument('--cpu-only',action='store_true');p.add_argument('--no-ablations',action='store_true')
    args=p.parse_args()
    report=run_matrix(args.initial_checkpoint,args.records,args.output,repeats=args.repeats,steps=args.steps,
        warmup=args.warmup,include_xpu=not args.cpu_only,ablations=not args.no_ablations,deadline_seconds=args.deadline_seconds)
    atomic_json(args.output/'compile-environment.json',compile_environment())
    print('Recommendation:',report['recommendation'])
    return 0
if __name__=='__main__':raise SystemExit(main())
