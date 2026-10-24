"""Conservative Lichess Evaluation Database labels for CHROMATIC Policy/WDL."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import chess
from .calibration import Calibration, normalize_white_cp_to_stm
from .profiling import profiled

INPUTS, ACTIONS, SCHEMA = 1280, 4672, 1


class QualityError(ValueError):
    def __init__(self,reason,detail=''):
        self.reason=reason
        super().__init__(reason+(': '+str(detail) if detail else ''))


@dataclass(frozen=True)
class QualityConfig:
    min_depth:int=12
    min_knodes:int=10
    max_pv_plies:int=256
    max_pvs:int=16
    policy_temperature_cp:float=100.
    validate_full_pv:bool=True


def action(move):
    if move.promotion in (chess.KNIGHT,chess.BISHOP,chess.ROOK):
        delta=chess.square_file(move.to_square)-chess.square_file(move.from_square)
        return 4096+move.from_square*9+(delta+1)*3+move.promotion-2
    return move.from_square*64+move.to_square


@profiled('fen_validation')
def board_from_fen(fen):
    if not isinstance(fen,str) or len(fen)>256:raise QualityError('invalid_fen')
    fields=fen.split()
    if len(fields)==4:fields+=['0','1']
    if len(fields)!=6:raise QualityError('invalid_fen_fields')
    try:board=chess.Board(' '.join(fields),chess960=False)
    except (ValueError,IndexError) as exc:raise QualityError('invalid_fen',exc) from exc
    if not board.is_valid():raise QualityError('illegal_position',board.status())
    # Non-capturable EP targets have no legal consequence and otherwise allow
    # duplicates with different encodings of the same legal position.
    if board.ep_square is not None and not board.has_legal_en_passant():board.ep_square=None
    return board


def identity_for(board):
    fields=board.fen(en_passant='legal').split()[:4]
    return hashlib.sha256(' '.join(fields).encode()).hexdigest()


def family_for(board):
    # Group opposite turns/castling/clock variants of the same placement. This is
    # deliberately broader than exact identity to prevent obvious split leakage.
    return hashlib.sha256(board.board_fen().encode()).hexdigest()


def similarity_bucket_for(board):
    material=','.join(str(len(board.pieces(pt,color))) for color in (True,False) for pt in range(1,7))
    pawns=f'{int(board.pawns & board.occupied_co[chess.WHITE]):016x}:{int(board.pawns & board.occupied_co[chess.BLACK]):016x}'
    return hashlib.sha256((material+':'+pawns).encode()).hexdigest()


def phase_for(board):
    # The official dump has four FEN fields, so fullmove=1 is a reconstruction
    # placeholder, never evidence of opening ply. This is a material proxy only.
    units=sum(len(board.pieces(pt,c))*weight for c in (True,False) for pt,weight in
              ((chess.KNIGHT,1),(chess.BISHOP,1),(chess.ROOK,2),(chess.QUEEN,4)))
    men=len(board.piece_map())
    if men>=28 and units>=20:return 'opening'
    if men<=12 or units<=8:return 'endgame'
    return 'middlegame'


def _integer(value,name,minimum=0):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(float(value)) or int(value)!=value or value<minimum:
        raise QualityError('invalid_'+name)
    return int(value)


@profiled('pv_validation')
def _pv(board,pv,config):
    if not isinstance(pv,dict) or ('cp' in pv)==('mate' in pv):raise QualityError('invalid_pv_score')
    if 'cp' in pv:
        cp=pv['cp']
        if isinstance(cp,bool) or not isinstance(cp,(int,float)) or not math.isfinite(float(cp)) or abs(cp)>100000:
            raise QualityError('invalid_cp')
        cp=float(cp);mate=None
        utility=normalize_white_cp_to_stm(cp,board.turn)
    else:
        mate=_integer(abs(pv['mate']),'mate',1) if isinstance(pv['mate'],(int,float)) and not isinstance(pv['mate'],bool) else None
        if mate is None:raise QualityError('invalid_mate')
        mate=mate if pv['mate']>0 else -mate;cp=None
        stm=mate if board.turn else -mate
        utility=(1000000.-min(abs(stm),1000)*100) if stm>0 else (-1000000.+min(abs(stm),1000)*100)
    line=pv.get('line')
    if not isinstance(line,str) or not line.strip() or len(line)>8192:raise QualityError('missing_pv')
    tokens=line.split()
    if len(tokens)>config.max_pv_plies:raise QualityError('pv_too_long')
    current=board.copy(stack=False);first=None;converted=[]
    for token in tokens if config.validate_full_pv else tokens[:1]:
        try:
            move=current.parse_uci(token)
        except (ValueError,chess.IllegalMoveError,chess.InvalidMoveError) as exc:raise QualityError('illegal_pv',token) from exc
        if not move or move not in current.legal_moves:raise QualityError('illegal_pv',token)
        if first is None:first=move
        converted.append(move.uci());current.push(move)
    return {'move':first,'action':action(first),'utility':utility,'cp_white':cp,'mate_white':mate,
            'line':converted,'plies_checked':len(converted)}


@profiled('label_conversion')
def parse_record(raw,calibration=None,config=None,source=None):
    config=config or QualityConfig()
    if config.min_depth<1 or config.min_knodes<1 or config.max_pvs<1 or config.max_pv_plies<1 or not math.isfinite(config.policy_temperature_cp) or config.policy_temperature_cp<=0:
        raise ValueError('invalid quality configuration')
    if not isinstance(raw,dict):raise QualityError('invalid_record')
    board=board_from_fen(raw.get('fen'))
    if board.is_game_over(claim_draw=False) or board.halfmove_clock>=100:raise QualityError('terminal_position')
    evaluations=raw.get('evals')
    if not isinstance(evaluations,list) or not evaluations or len(evaluations)>256:raise QualityError('missing_evaluations')
    candidates=[];reasons=[]
    for entry in evaluations:
        try:
            if not isinstance(entry,dict):raise QualityError('invalid_evaluation')
            depth=_integer(entry.get('depth'),'depth',1);knodes=_integer(entry.get('knodes'),'knodes',1)
            if depth<config.min_depth:raise QualityError('low_depth')
            if knodes<config.min_knodes:raise QualityError('low_nodes')
            pvs=entry.get('pvs')
            if not isinstance(pvs,list) or not pvs or len(pvs)>config.max_pvs:raise QualityError('invalid_pv_count')
            parsed=[_pv(board,pv,config) for pv in pvs]
            if len({p['action'] for p in parsed})!=len(parsed):raise QualityError('duplicate_pv_first_move')
            candidates.append((depth,knodes,len(parsed),parsed))
        except QualityError as exc:reasons.append(exc.reason)
    if not candidates:raise QualityError(reasons[0] if reasons else 'no_usable_evaluation')
    depth,knodes,_,pvs=max(candidates,key=lambda c:(c[0],c[1],c[2]))
    pvs=sorted(pvs,key=lambda p:p['utility'],reverse=True)
    maximum=pvs[0]['utility'];scores=[math.exp(max(-80.,(pv['utility']-maximum)/config.policy_temperature_cp)) for pv in pvs]
    total=sum(scores);policy=[[pv['action'],weight/total] for pv,weight in zip(pvs,scores)]
    legal_count=board.legal_moves.count()
    depth_quality=min(1.,depth/32.)
    nodes_quality=max(.25,min(1.,math.log1p(knodes)/math.log1p(1000000.)))
    quality=depth_quality*nodes_quality*.9 # unknown history/repetition is explicit uncertainty
    partial_weight=.25 if len(pvs)==1 else min(.75,.35+.4*len(pvs)/legal_count)
    policy_weight=quality*partial_weight
    best=pvs[0];wdl=[0.,1.,0.];value_weight=0.;value_source='unavailable_no_accepted_calibration'
    calibration_sha=None
    if best['mate_white'] is not None:
        positive=(best['mate_white']>0)==board.turn
        wdl=[1.,0.,0.] if positive else [0.,0.,1.]
        value_weight=.25*quality;value_source='mate_sign_white_to_stm_weak_teacher'
    elif calibration is not None:
        calibration=calibration if isinstance(calibration,Calibration) else Calibration(calibration)
        calibration_sha=calibration.sha256
        if calibration.usable:
            wdl=calibration.predict(best['cp_white'],board.turn)
            value_weight=quality*calibration.confidence;value_source='independent_game_outcome_calibration'
    fields=raw['fen'].split()
    return {'schema':SCHEMA,'source':'lichess-eval-cc0','root_fen':board.fen(en_passant='legal'),'moves':[],
            'fen':board.fen(en_passant='legal'),'identity':identity_for(board),'family':family_for(board),
            'similarity_bucket':similarity_bucket_for(board),'phase':phase_for(board),'phase_basis':'material_proxy_not_game_ply',
            'history_known':False,'history_unknown_fields':['halfmove_clock','fullmove_number','repetition'] if len(fields)==4 else ['repetition'],
            'policy':policy,'wdl':wdl,'policy_weight':policy_weight,'value_weight':value_weight,
            'value_source':value_source,'calibration_sha256':calibration_sha,'policy_kind':'partial_pv_imitation',
            'observed_policy_moves':len(pvs),'legal_moves':legal_count,'policy_support_complete':len(pvs)==legal_count,
            'source_eval':{'depth':depth,'knodes':knodes,'nodes':knodes*1000,'cp_white':best['cp_white'],
                           'mate_white':best['mate_white'],'pvs':len(pvs),'quality_weight':quality,
                           'pv_moves_uci':[p['move'].uci() for p in pvs],'pv_plies_checked':[p['plies_checked'] for p in pvs],
                           'rejected_evaluations':reasons,'score_pov':'white','target_pov':'side-to-move'},
            'provenance':dict(source or {})}
