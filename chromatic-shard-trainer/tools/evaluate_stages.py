"""Bounded 100/1,000/10,000-position evaluation and common-holdout comparison."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from shardtrain.evaluate import evaluate_run,common_holdout_report
from shardtrain.storage import atomic_json


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-root',type=Path,default=ROOT.parent/'chromatic-shard-validation')
    parser.add_argument('--engine',type=Path,default=ROOT.parent/'project-chromatic-memory/dist/chromatic-memory.exe')
    parser.add_argument('--pairs',type=int,default=2)
    parser.add_argument('--milliseconds',type=int,default=10)
    parser.add_argument('--max-plies',type=int,default=128)
    args=parser.parse_args(argv);base=args.validation_root.resolve()
    runs={str(size):(base/f'positions-{size}'/'data',base/f'positions-{size}'/'candidate') for size in [100,1000,10000]}
    for name,(_,candidate) in runs.items():
        status=json.loads((candidate/'status.json').read_text(encoding='utf-8'))
        if status.get('status')!='completed':raise ValueError(name+' training must finish before fixed-time evaluation')
    summary={'format':'chromatic-stage-evaluations-v1','stages':{}}
    for name,(data,candidate) in runs.items():
        report=evaluate_run(data,candidate,args.engine,ROOT/'champion/model.pt',ROOT/'champion/default.zero',
                            ROOT/'fixtures',pairs=args.pairs,milliseconds=args.milliseconds,max_plies=args.max_plies)
        summary['stages'][name]={'path':str(candidate/'evaluation-report.json'),'elapsed_seconds':report['elapsed_seconds'],
                                'candidate_step':report['candidate_step'],'promotion':report['promotion']}
        atomic_json(base/'evaluation-summary.json',summary)
        print(json.dumps({'stage':name,**summary['stages'][name]},ensure_ascii=False),flush=True)
    common=common_holdout_report(runs['10000'][0],runs,ROOT/'champion/model.pt',base/'common-holdout.json')
    summary['common_holdout']=str(base/'common-holdout.json');atomic_json(base/'evaluation-summary.json',summary)
    print(json.dumps({'common_holdout':summary['common_holdout'],'models':list(common['models'])},ensure_ascii=False),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
