"""Bounded teacher-free games with durable unfinished trajectories."""
import base64
import hashlib
import json
from pathlib import Path
import random
import time
import zlib
import chess
from .checkpoint import digest
from .data import atomic_json,position_record,write_records
from .encoding import action,features,outcome,restore_board
from .native import Native


def selfplay(executable,model,output,games=8,simulations=128,seed=1,max_plies=512,
             start_fen=chess.STARTING_FEN,noise=.25,temperature_plies=30,
             temperature=1.,late_temperature=0.,opening_plies=0,starting_fens=None,
             engine_options=None,resume_truncated=False,save_every_plies=32):
    if games<1 or games>10000 or simulations<1 or not 1<=max_plies<=4096 or not 0<=noise<=1 or save_every_plies<1:
        raise ValueError('invalid bounded selfplay settings')
    if not 0<=opening_plies<=64 or min(temperature,late_temperature)<0:raise ValueError('invalid exploration settings')
    output=Path(output);output.mkdir(parents=True,exist_ok=True);pending=output/'pending';pending.mkdir(exist_ok=True)
    if model is None:
        adjacent=Path(executable).resolve().parent/'default.zero'
        if adjacent.exists():model=adjacent
    model_hash=digest(model) if model else 'uniform-baseline'
    settings={'format':2,'model':model_hash,'executable':digest(executable),'simulations':simulations,'seed':seed,
              'max_plies':max_plies,'start_fen':start_fen,'starting_fens':starting_fens,'noise':noise,
              'temperature_plies':temperature_plies,'temperature':temperature,'late_temperature':late_temperature,
              'opening_plies':opening_plies,'engine_options':engine_options or {}}
    manifest=output/'settings.json'
    if manifest.exists():
        old=json.loads(manifest.read_text(encoding='utf-8'))
        same=dict(old,max_plies=max_plies)==settings
        if not (old==settings or (resume_truncated and same and old['max_plies']<=max_plies)):
            raise ValueError('selfplay output belongs to different settings/model')
    atomic_json(manifest,settings)
    identity_settings={k:v for k,v in settings.items() if k!='max_plies'}
    run_id=hashlib.sha256(json.dumps(identity_settings,sort_keys=True).encode()).hexdigest()
    accepted=truncated=records_count=draws=generated=0;started=time.perf_counter()
    with Native(executable,model) as engine:
        if engine_options:engine.configure(engine_options)
        for game in range(games):
            path=output/f'game_{game:06d}.jsonl';meta=path.with_suffix('.json');partial=pending/path.name
            info=json.loads(meta.read_text(encoding='utf-8')) if meta.exists() else None
            reuse=info is not None and info.get('accepted') is not None and path.exists() and (info['accepted'] or not resume_truncated)
            if reuse:
                if info['sha256']!=digest(path):raise ValueError('selfplay shard checksum mismatch')
            else:
                records=[];rng=random.Random(seed+game*1000003)
                if partial.exists():
                    records=[json.loads(line) for line in partial.read_text(encoding='utf-8').splitlines()]
                    if any(row['game_id']!=run_id+':'+str(game) for row in records):raise ValueError('pending trajectory provenance mismatch')
                if records:
                    board=restore_board(records[-1]);board.push_uci(records[-1]['selected_uci'])
                    initial_opening=len(records[0]['moves'])
                else:
                    board=chess.Board(rng.choice(starting_fens) if starting_fens else start_fen)
                    if not board.is_valid() or outcome(board):raise ValueError('selfplay start must be valid and nonterminal')
                    for _ in range(opening_plies):
                        moves=list(board.legal_moves);rng.shuffle(moves)
                        for move in moves:
                            board.push(move)
                            if not outcome(board):break
                            board.pop()
                        else:break
                    initial_opening=len(board.move_stack)
                def durable_progress(reason):
                    write_records(partial,records)
                    progress={'game':game,'accepted':None,'records':0,'plies':len(records),'termination':reason,
                              'moves':[m.uci() for m in board.move_stack],'pending_sha256':digest(partial)}
                    atomic_json(meta,progress)
                for ply in range(len(records),max_plies):
                    if outcome(board):break
                    engine.position(board);search_seed=seed+game*1000003+ply
                    try:result=engine.search(simulations,search_seed,noise,temperature if ply<temperature_plies else late_temperature)
                    except Exception:
                        durable_progress('ENGINE_ERROR');raise
                    if result['terminal']:raise RuntimeError('C++ and Python termination disagreement')
                    move=chess.Move.from_uci(result['bestmove'])
                    if move not in board.legal_moves:raise RuntimeError('native illegal selfplay move')
                    total=sum(item['n'] for item in result['visits'])
                    if total!=result['simulations'] or total<1:raise RuntimeError('invalid MCTS visits')
                    policy=[[item['action'],item['n']/total] for item in result['visits'] if item['n']]
                    record=position_record(board,policy,'selfplay',schema=2,model=model_hash,game=game,
                        game_id=run_id+':'+str(game),simulations=total,search_budget=simulations,
                        random_seed=search_seed,selected_action=action(move),selected_uci=move.uci(),
                        legal_actions=sorted(action(m) for m in board.legal_moves),
                        encoded_position=base64.b64encode(zlib.compress(features(board).astype('<f4').tobytes())).decode('ascii'))
                    records.append(record);board.push(move);generated+=1
                    if len(records)%save_every_plies==0:durable_progress('IN_PROGRESS')
                result=outcome(board)
                if result:
                    for row in records:
                        turn=restore_board(row).turn
                        row['wdl']=[0,1,0] if result.winner is None else [1,0,0] if result.winner==turn else [0,0,1]
                        row['result']=result.result();row['termination']=result.termination.name
                    kept=records
                else:
                    # Keep policy targets durably, without fabricating an outcome.
                    durable_progress('TRUNCATED');kept=[]
                write_records(path,kept)
                info={'game':game,'accepted':result is not None,'records':len(kept),'plies':len(records),
                      'opening_plies':initial_opening,'result':result.result() if result else '*',
                      'termination':result.termination.name if result else 'TRUNCATED',
                      'moves':[m.uci() for m in board.move_stack],'sha256':digest(path)}
                atomic_json(meta,info)
            accepted+=bool(info['accepted']);truncated+=not info['accepted'];records_count+=info['records']
            draws+=bool(info['accepted'] and info['result']=='1/2-1/2')
    elapsed=time.perf_counter()-started
    summary={'games':games,'accepted':accepted,'truncated':truncated,'records':records_count}
    atomic_json(output/'summary.json',summary)
    atomic_json(output/'telemetry.json',{'seconds':elapsed,'new_positions_generated':generated,
                'new_positions_per_minute':generated*60/elapsed,'completed_draws':draws,
                'draw_rate_of_completed':draws/accepted if accepted else None,'completion_rate':accepted/games,
                'crashes_observed':0,'unfinished_trajectories_directory':'pending'})
    return summary
