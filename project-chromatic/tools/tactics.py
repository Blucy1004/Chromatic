"""Small independently proved suite. Accuracy is not an Elo estimate."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import chess
from zero.encoding import outcome
from zero.native import Native
from zero.data import atomic_json
from zero.checkpoint import digest

def solutions(board,plies):
    cache={};nodes=0
    def forced(b,depth,attacker):
        nonlocal nodes
        nodes+=1
        if nodes>250000:raise RuntimeError('oracle node budget exceeded')
        if b.is_checkmate():return b.turn!=attacker
        if depth==0 or outcome(b):return False
        key=(b.board_fen(),b.turn,depth,attacker)
        if key in cache:return cache[key]
        moves=list(b.legal_moves);moves.sort(key=lambda m:b.gives_check(m),reverse=True)
        result=b.turn!=attacker
        for move in moves:
            b.push(move);win=forced(b,depth-1,attacker);b.pop()
            if b.turn==attacker and win:result=True;break
            if b.turn!=attacker and not win:result=False;break
        cache[key]=result;return result
    attacker=board.turn
    if plies>1 and forced(board,plies-2,attacker):raise AssertionError('mate depth is not minimal')
    moves=[]
    for move in list(board.legal_moves):
        board.push(move);yes=forced(board,plies-1,attacker);board.pop()
        if yes:moves.append(move.uci())
    return sorted(moves),nodes

def verify_case(case):
    board=chess.Board(case['fen'])
    if not board.is_valid():raise AssertionError('invalid tactical FEN: '+case['id'])
    for text in case.get('moves',[]):board.push_uci(text)
    if case['oracle']=='mate':
        expected,nodes=solutions(board,case['plies'])
        if set(expected)!=set(case['expected']):raise AssertionError(f"{case['id']} expected {expected}, provided {case['expected']}")
    elif case['oracle']=='only_legal_capture':
        legal=list(board.legal_moves);nodes=1
        if len(legal)!=1 or not board.is_capture(legal[0]) or [legal[0].uci()]!=case['expected']:
            raise AssertionError('not a forced legal capture')
    else:
        nodes=1
        if not board.is_repetition(3) or outcome(board).winner is not None:raise AssertionError('not a repeated draw')
    return board,nodes

def tactics(executable,model,suite,output,milliseconds=25,nodes=None,options=None):
    cases=json.loads(Path(suite).read_text(encoding='utf-8'))['cases']
    verified=[verify_case(case) for case in cases];rows=[]
    with Native(executable,model) as engine:
        if options:engine.configure(options)
        for case,(board,proof_nodes) in zip(cases,verified):
            result=engine.play(board,nodes=nodes) if nodes else engine.play(board,milliseconds=milliseconds)
            passed=result['bestmove'] in case['expected']
            rows.append({'id':case['id'],'category':case['category'],'passed':passed,
                         'chosen':result['bestmove'],'expected':case['expected'],
                         'elapsed_seconds':result['elapsed_seconds'],'visits':result['info'].get('nodes'),
                         'oracle_nodes':proof_nodes,'failure_category':None if passed else case['category']})
    report={'suite_sha256':digest(suite),'engine_sha256':digest(executable),'model_sha256':digest(model) if model else None,
            'budget':{'milliseconds':milliseconds if not nodes else None,'nodes':nodes},'options':options or {},
            'correct':sum(row['passed'] for row in rows),'total':len(rows),
            'accuracy':sum(row['passed'] for row in rows)/len(rows),'cases':rows}
    atomic_json(output,report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--executable',required=True);p.add_argument('--model')
    p.add_argument('--suite',default='benchmarks/tactics.json');p.add_argument('--output',required=True)
    p.add_argument('--milliseconds',type=int,default=25);p.add_argument('--nodes',type=int);p.add_argument('--options',default='{}')
    args=vars(p.parse_args());args['options']=json.loads(args['options'])
    print(json.dumps(tactics(**args),indent=2))
