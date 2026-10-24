"""Fresh, bounded v0.1 reconstruction. Refuses cached output for timing integrity."""
from pathlib import Path
import argparse
import json
import sys
import time
import torch
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'baseline/source'))
from zero.train import train
from zero.selfplay import selfplay
from zero.checkpoint import load
from zero.data import atomic_json

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    args=p.parse_args();directory=Path(args.output).resolve()
    if directory.exists() and any(directory.iterdir()):raise SystemExit('Choose an empty output directory; cached timings are not generation timings')
    directory.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    training=train([root/'runs/bootstrap/teacher.jsonl'],directory/'model.pt',steps=200,batch_size=32,
                   save_every=50,width=64,blocks=2,seed=2026)
    train_seconds=time.perf_counter()-start
    actual,_=load(directory/'model.pt');expected,_=load(root/'runs/bootstrap/model.pt')
    same=all(torch.equal(actual['model'][k],expected['model'][k]) for k in actual['model'])
    if not same:raise RuntimeError('baseline bootstrap weights did not reproduce')
    start=time.perf_counter()
    games=selfplay(root/'baseline/project_zero.exe',root/'baseline/default.zero',directory/'games',
                   games=4,simulations=64,max_plies=512,seed=2026)
    elapsed=time.perf_counter()-start
    metas=[json.loads(path.read_text()) for path in (directory/'games').glob('game_*.json')]
    draws=sum(row['accepted'] and row['result']=='1/2-1/2' for row in metas)
    report={'fresh_output':True,'training':training,'train_seconds':train_seconds,
            'train_samples_per_second_including_io':6400/train_seconds,
            'bootstrap_weights_equal':same,'selfplay':games,'selfplay_seconds':elapsed,
            'accepted_positions_per_minute':games['records']*60/elapsed,
            'draw_rate':draws/games['accepted'] if games['accepted'] else None}
    atomic_json(directory/'report.json',report);print(json.dumps(report,indent=2))
