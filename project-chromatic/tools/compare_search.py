import argparse
import json
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import chess
from zero.native import Native
from zero.data import atomic_json

def compare(baseline,candidate,model,output,positions=20,simulations=128):
    rng=random.Random(2026);board=chess.Board();mismatches=[]
    with Native(baseline,model) as old,Native(candidate,model) as new:
        if 'InferenceBackend' in new.options:new.configure({'InferenceBackend':'Scalar','MateProofNodes':0})
        for index in range(positions):
            for engine in (old,new):engine.position(board)
            before=old.search(simulations,2026,0,0);after=new.search(simulations,2026,0,0)
            counts=lambda row:[(v['uci'],v['action'],v['n']) for v in row['visits']]
            if (before['bestmove'],counts(before),before['value'])!=(after['bestmove'],counts(after),after['value']):
                mismatches.append({'index':index,'fen':board.fen(),'baseline':before,'candidate':after})
            for _ in range(3):
                if board.is_game_over():board=chess.Board()
                board.push(rng.choice(list(board.legal_moves)))
    result={'positions':positions,'simulations':simulations,'identical':not mismatches,'mismatches':mismatches}
    atomic_json(output,result)
    if mismatches:raise AssertionError(f'{len(mismatches)} searches differ')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for arg in ('baseline','candidate','model','output'):p.add_argument('--'+arg,required=True)
    p.add_argument('--positions',type=int,default=20);p.add_argument('--simulations',type=int,default=128)
    print(json.dumps(compare(**vars(p.parse_args())),indent=2))
