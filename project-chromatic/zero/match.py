"""Paired UCI matches with honest incomplete results and conservative promotion."""
import json
import math
import os
from pathlib import Path
import random
import chess
import chess.pgn
from .checkpoint import digest
from .data import atomic_json
from .encoding import outcome
from .native import Native


def opening_identity(game,index):
    opening=game.get('opening')
    if opening is None:return 'fixture-pair:'+str(index//2)
    return json.dumps({'fen':opening['fen'],'moves':opening.get('moves',[])},sort_keys=True)


def statistics(games,confidence=.95,threshold=.55,anytime=False):
    complete=bool(games) and all(game['score'] is not None for game in games)
    pairs=len(games)//2
    if not complete or len(games)%2:
        return {'score':None,'lower_bound':None,'upper_bound':None,'elo':None,'elo_interval':None,
                'decision':'insufficient_evidence','promote':False,'complete':False}
    score=sum(game['score'] for game in games)/len(games)
    clusters={}
    for index in range(0,len(games),2):
        clusters.setdefault(opening_identity(games[index],index),[]).append((games[index]['score']+games[index+1]['score'])/2)
    # Repeating a deterministic opening must not manufacture independent evidence.
    squared_weights=sum((len(values)/pairs)**2 for values in clusters.values())
    alpha=1-confidence
    if anytime:alpha/=pairs*(pairs+1) # union bound over all stopping times
    margin=math.sqrt(math.log(2/alpha)*squared_weights/2)
    lower=max(0.,score-margin);upper=min(1.,score+margin)
    def elo(value):return 400*math.log10(value/(1-value)) if 0<value<1 else None
    decision='promote' if lower>=threshold else 'demonstrated_inferiority' if upper<.5 else 'insufficient_evidence'
    return {'score':score,'lower_bound':lower,'upper_bound':upper,'elo':elo(score),
            'elo_interval':{'lower':elo(lower),'upper':elo(upper),'unbounded_lower':lower==0,'unbounded_upper':upper==1},
            'decision':decision,'promote':decision=='promote','complete':True,
            'unique_opening_clusters':len(clusters),'effective_independent_units':1/squared_weights}


def sprt_state(games,alpha=.05,beta=.05,p0=.5,p1=.65):
    # Exact Bernoulli sign test on non-tied opening pairs, not a fractional-draw Elo SPRT.
    wins=losses=0;seen=set()
    for i in range(0,len(games)-1,2):
        scores=[game['score'] for game in games[i:i+2]]
        if None in scores:continue
        identity=opening_identity(games[i],i)
        if identity in seen:continue
        seen.add(identity)
        score=sum(scores)/2
        wins+=score>.5;losses+=score<.5
    llr=wins*math.log(p1/p0)+losses*math.log((1-p1)/(1-p0))
    lower=math.log(beta/(1-alpha));upper=math.log((1-beta)/alpha)
    return {'test':'non_tied_opening_pair_sign','p0':p0,'p1':p1,'pair_wins':wins,'pair_losses':losses,
            'log_likelihood_ratio':llr,'lower':lower,'upper':upper,'unique_opening_clusters':len(seen),
            'status':'accept_h1' if llr>=upper else 'accept_h0' if llr<=lower else 'continue'}


def write_pgn(path,games,candidate_name,incumbent_name):
    text=[]
    for record in games:
        game=chess.pgn.Game();game.setup(chess.Board(record['root_fen']));node=game
        for move in record['moves']:node=node.add_variation(chess.Move.from_uci(move))
        game.headers.update(Event='CHROMATIC bounded arena',Round=str(record['pair']+1),
                            White=candidate_name if record['candidate_white'] else incumbent_name,
                            Black=incumbent_name if record['candidate_white'] else candidate_name,
                            Result=record['result'],Termination=record['termination'])
        text.append(str(game))
    path=Path(path);temporary=Path(str(path)+'.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        stream.write('\n\n'.join(text)+'\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def arena(executable,candidate,incumbent,output,pairs=8,simulations=128,max_plies=512,seed=1,
          threshold=.55,confidence=.95,opening_fens=None,candidate_executable=None,incumbent_executable=None,
          milliseconds=None,opening_suite=None,candidate_options=None,incumbent_options=None,sprt=False,affinity_mask=1):
    if not 1<=pairs<=10000 or not 1<=max_plies<=4096 or not .5<=threshold<=1 or not 0<confidence<1:
        raise ValueError('invalid bounded arena settings')
    if milliseconds is not None and not 1<=milliseconds<=60000:raise ValueError('invalid time budget')
    candidate_executable=candidate_executable or executable;incumbent_executable=incumbent_executable or executable
    if opening_suite:
        openings=json.loads(Path(opening_suite).read_text(encoding='utf-8'))['openings']
    elif opening_fens:openings=[{'fen':fen,'moves':[]} for fen in opening_fens]
    else:
        rng=random.Random(seed);openings=[]
        for _ in range(pairs):
            board=chess.Board()
            for _ in range(6):board.push(rng.choice(list(board.legal_moves)))
            openings.append({'fen':board.root().fen(),'moves':[m.uci() for m in board.move_stack]})
    if not openings:raise ValueError('empty opening suite')
    settings={'format':2,'candidate_engine':digest(candidate_executable),'incumbent_engine':digest(incumbent_executable),
              'candidate':digest(candidate) if candidate else None,'incumbent':digest(incumbent) if incumbent else None,
              'pairs':pairs,'nodes':simulations if milliseconds is None else None,'milliseconds':milliseconds,
              'max_plies':max_plies,'seed':seed,'openings':openings,'candidate_options':candidate_options or {},
              'incumbent_options':incumbent_options or {},'confidence':confidence,'threshold':threshold,'sprt':sprt,'affinity_mask':affinity_mask}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);progress=output.with_suffix('.progress.json')
    if progress.exists():
        previous=json.loads(progress.read_text(encoding='utf-8'))
        if previous['settings']!=settings:raise ValueError('arena output belongs to different settings')
        games=previous['games']
    else:games=[]
    with Native(candidate_executable,candidate) as challenger,Native(incumbent_executable,incumbent) as champion:
        for engine in (challenger,champion):
            if affinity_mask:engine.pin_cpu(affinity_mask)
            if 'Threads' in engine.options:engine.configure({'Threads':1})
        if candidate_options:challenger.configure(candidate_options)
        if incumbent_options:champion.configure(incumbent_options)
        for pair in range(pairs):
            for candidate_white in (True,False):
                index=pair*2+int(not candidate_white)
                if index<len(games):continue
                opening=openings[pair%len(openings)];board=chess.Board(opening['fen'])
                for move in opening.get('moves',[]):board.push_uci(move)
                if not board.is_valid() or outcome(board):raise ValueError('invalid or terminal arena opening')
                times=[];visits=[];failure=None
                for _ in range(max_plies):
                    if outcome(board):break
                    engine=challenger if board.turn==candidate_white else champion
                    try:
                        result=engine.play(board,milliseconds=milliseconds) if milliseconds is not None else engine.play(board,nodes=simulations)
                        move=chess.Move.from_uci(result['bestmove'])
                        if move not in board.legal_moves:raise RuntimeError('engine returned illegal move')
                    except Exception as exc:failure=str(exc);break
                    times.append(result['elapsed_seconds']);visits.append(result['info'].get('nodes'));board.push(move)
                result=outcome(board)
                score=None if result is None else .5 if result.winner is None else float(result.winner==candidate_white)
                games.append({'pair':pair,'candidate_white':candidate_white,'score':score,
                              'result':result.result() if result else '*','root_fen':board.root().fen(en_passant='fen'),
                              'opening':opening,'moves':[m.uci() for m in board.move_stack],
                              'termination':result.termination.name if result else 'ENGINE_ERROR' if failure else 'TRUNCATED',
                              'error':failure,'move_seconds':times,'visits':visits})
                atomic_json(progress,{'settings':settings,'games':games})
                write_pgn(output.with_suffix('.pgn'),games,challenger.name,champion.name)
            if sprt and sprt_state(games)['status']!='continue':break
    stats=statistics(games,confidence,threshold,anytime=sprt)
    if any(game['error'] for game in games):stats.update(decision='insufficient_evidence',promote=False)
    report={**settings,**stats,'pairs':len(games)//2,'completed_games':sum(g['score'] is not None for g in games),
            'confidence':confidence,'threshold':threshold,'simulations':simulations,'games':games,
            'wdl':{'wins':sum(g['score']==1 for g in games),'draws':sum(g['score']==.5 for g in games),
                   'losses':sum(g['score']==0 for g in games),'unfinished':sum(g['score'] is None for g in games)},
            'engine_errors':sum(g['error'] is not None for g in games),
            'sprt_state':sprt_state(games) if sprt else None,
            'statistical_scope':'paired opening distribution; this small run is not a certified engine rating'}
    atomic_json(output,report);return report
