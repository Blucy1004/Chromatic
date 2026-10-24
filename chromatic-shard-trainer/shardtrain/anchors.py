"""Weak, explicitly partial opening imitation from the official CC0 TSVs."""
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import chess
import chess.pgn
from .quality import action,identity_for,family_for,similarity_bucket_for

def _catalog(path):
    with path.open(encoding='utf-8') as stream:yield from csv.DictReader(stream,delimiter='\t')

def opening_anchors(directory,limit,seed):
    directory=Path(directory);meta=json.loads((directory/'official-openings.source.json').read_text(encoding='utf-8'))
    if meta.get('license')!='CC0-1.0':raise ValueError('opening anchors require official CC0 provenance')
    positions={}
    for letter in 'abcde':
        path=directory/f'official-openings-{letter}.tsv'
        for row in _catalog(path):
            game=chess.pgn.read_game(io.StringIO(row['pgn']))
            if game is None or game.errors:continue
            board=game.board();history=[]
            for ply,move in enumerate(game.mainline_moves()):
                if ply>12:break
                if move not in board.legal_moves:break
                identity=identity_for(board)
                if identity not in positions:positions[identity]={'board':board.copy(stack=True),'history':list(history),'counts':Counter(),'ply':ply}
                positions[identity]['counts'][action(move)]+=1
                board.push(move);history.append(move.uci())
    rows=[]
    for identity,item in positions.items():
        b=item['board'];counts=item['counts'];total=sum(counts.values())
        rows.append({'schema':1,'source':'lichess-openings-cc0','root_fen':chess.STARTING_FEN,'moves':item['history'],
          'fen':b.fen(en_passant='fen'),'identity':identity,'family':family_for(b),'similarity_bucket':similarity_bucket_for(b),
          'phase':'opening','phase_basis':'verified_opening_history','history_known':True,'ply':item['ply'],
          'policy':[[a,n/total] for a,n in sorted(counts.items())],'wdl':[0.,1.,0.],
          'policy_weight':.15,'value_weight':0.,'value_source':'unavailable_opening_value_not_fabricated',
          'policy_kind':'opening_catalog_partial_imitation_not_visit_distribution','policy_support_complete':False,
          'provenance':{'license':'CC0-1.0','url':'https://github.com/lichess-org/chess-openings','version':meta.get('commit') or meta.get('version'),'catalog_count':total}})
    # Round-robin true ply strata; root is always the first entry.
    pools={p:sorted([r for r in rows if r['ply']==p],key=lambda r:hashlib.sha256(f"{seed}:{r['identity']}".encode()).hexdigest()) for p in range(13)}
    selected=[]
    while len(selected)<limit and any(pools.values()):
        for p in range(13):
            if pools[p] and len(selected)<limit:selected.append(pools[p].pop(0))
    root=next(r for r in rows if not r['moves'])
    if not any(r['identity']==root['identity'] for r in selected):selected=[root]+selected[:max(0,limit-1)]
    return selected
