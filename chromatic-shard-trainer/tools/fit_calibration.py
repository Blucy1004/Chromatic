"""Fit the small independent CP/WDL calibration; never trains the chess model."""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import sys
import time

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
sys.dont_write_bytecode=True
import chess
import chess.pgn
import torch
from shardtrain.calibration import fit_calibration,save_calibration
from shardtrain.quality import family_for

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic_json(path,obj):
    from shardtrain.storage import atomic_json as save
    save(path,obj)

def reserve_opening_families(fixtures,max_ply=12):
    families=set();lines=0;files={};maximum=0
    for letter in 'abcde':
        path=fixtures/f'official-openings-{letter}.tsv'
        files[path.name]={'sha256':digest(path),'bytes':path.stat().st_size}
        with path.open(encoding='utf-8',newline='') as stream:
            for record in csv.DictReader(stream,delimiter='\t'):
                game=chess.pgn.read_game(io.StringIO(record['pgn']))
                if game is None or game.errors:raise ValueError(f'invalid official opening: {record}')
                board=game.board();families.add(family_for(board));lines+=1
                for ply,move in enumerate(game.mainline_moves(),1):
                    if ply>max_ply:break
                    board.push(move);families.add(family_for(board));maximum=max(maximum,ply)
    return families,{'source_files':files,'opening_rows':lines,'families':len(families),'maximum_protected_ply':maximum}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,default=PROJECT)
    parser.add_argument('--steps',type=int,default=300)
    parser.add_argument('--min-holdout-games',type=int,default=10)
    args=parser.parse_args()
    started=time.perf_counter();root=args.project.resolve();fixtures=root/'fixtures';reports=root/'reports'
    reports.mkdir(exist_ok=True);torch.set_num_threads(1)
    reserved,opening_report=reserve_opening_families(fixtures)
    path=fixtures/'official-calibration-rows.jsonl'
    source=json.loads(Path(str(path)+'.source.json').read_text(encoding='utf-8'))
    assert digest(path)==source['local_sha256']
    rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
    overlap_games={r['game_id'] for r in rows if r['family'] in reserved}
    training=[r for r in rows if r['game_id'] not in overlap_games]
    assert not {r['family'] for r in training}&reserved
    fit_started=time.perf_counter()
    artifact=fit_calibration(training,seed=2026,min_holdout_games=args.min_holdout_games,steps=args.steps)
    fit_seconds=time.perf_counter()-fit_started
    artifact['opening_anchor_exclusion']={**opening_report,'input_rows':len(rows),'input_games':len({r['game_id'] for r in rows}),
        'removed_games':sorted(overlap_games),'removed_rows':len(rows)-len(training),'remaining_rows':len(training),
        'remaining_games':len({r['game_id'] for r in training}),
        'reserved_family_sha256':hashlib.sha256('\n'.join(sorted(reserved)).encode()).hexdigest(),
        'method':'exclude whole calibration game if any selected calibration row has a protected opening family'}
    artifact=save_calibration(fixtures/'calibration.json',artifact)
    atomic_json(reports/'calibration-report.json',{'status':artifact['status'],'artifact_sha256':artifact['sha256'],
        'artifact':'fixtures/calibration.json','input_source':source,'opening_exclusion':artifact['opening_anchor_exclusion'],
        'fit_seconds':fit_seconds,'total_seconds':time.perf_counter()-started,'parameters':artifact['parameters'],
        'group_counts':{k:len(v) for k,v in artifact['groups'].items()},'reports':artifact['reports'],
        'constant_prior_reports':artifact['constant_prior_reports'],'reasons':artifact['reasons'],
        'neural_network_training_performed':False,'meaning':'independent empirical game-outcome calibration, not perfect-play WDL'})
    atomic_json(reports/'reserved-opening-families.json',{'families':sorted(reserved),**opening_report})
    print(json.dumps({'status':artifact['status'],'groups':{k:len(v) for k,v in artifact['groups'].items()},
        'removed_anchor_overlap_games':len(overlap_games),'rows':len(training),'fit_seconds':fit_seconds,
        'test':artifact['reports']['test'],'reasons':artifact['reasons']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
