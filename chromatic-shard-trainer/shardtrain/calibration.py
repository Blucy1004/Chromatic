"""Empirical CP -> WDL calibration using an independent, grouped outcome corpus.

The eval dump has no game outcomes. Nothing in this module treats CP/100 as a
probability or silently supplies an uncalibrated logistic target. Calibration is
fit only from separate completed CC0 games with annotated CP and Result tags.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Iterable

FORMAT = 1
METHOD = 'symmetric-ordinal-logistic-human-outcomes-v1'
RESULT_CLASS = {'1-0': 0, '1/2-1/2': 1, '0-1': 2}
OFFICIAL_LICENSE = 'CC0-1.0'


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def artifact_digest(artifact):
    return hashlib.sha256(canonical_json({k: v for k, v in artifact.items() if k != 'sha256'}).encode()).hexdigest()


def normalize_white_cp_to_stm(cp_white, turn):
    value = float(cp_white)
    if not math.isfinite(value):
        raise ValueError('nonfinite CP')
    return value if bool(turn) else -value


def _sigmoid(value):
    value = max(-40., min(40., float(value)))
    return 1. / (1. + math.exp(-value))


def white_distribution(cp_white, scale_cp, draw_margin_cp):
    """Ordinal outcome probabilities; parameters must have been independently fit."""
    if not (math.isfinite(scale_cp) and scale_cp > 0 and math.isfinite(draw_margin_cp) and draw_margin_cp >= 0):
        raise ValueError('invalid calibration parameters')
    cp = max(-8000., min(8000., float(cp_white)))
    if not math.isfinite(float(cp_white)):
        raise ValueError('nonfinite CP')
    win = _sigmoid((cp-draw_margin_cp)/scale_cp)
    loss = _sigmoid((-cp-draw_margin_cp)/scale_cp)
    draw = max(0., 1.-win-loss)
    total = win+draw+loss
    return [win/total, draw/total, loss/total]


@dataclass(frozen=True)
class Calibration:
    artifact: dict

    def __post_init__(self):
        a = self.artifact
        if a.get('format') != FORMAT or a.get('method') != METHOD:
            raise ValueError('unsupported calibration artifact')
        if a.get('sha256') != artifact_digest(a):
            raise ValueError('calibration artifact checksum mismatch')
        p = a.get('parameters', {})
        white_distribution(0., float(p.get('scale_cp', float('nan'))), float(p.get('draw_margin_cp', float('nan'))))
        groups = a.get('groups', {})
        named = [set(groups.get(k, [])) for k in ('fit', 'validation', 'test')]
        if any(named[i] & named[j] for i in range(3) for j in range(i+1, 3)):
            raise ValueError('calibration game groups overlap')
        if a.get('status') == 'accepted':
            minimum = int(a.get('minimum_holdout_games', 10))
            if minimum < 1 or len(named[1]) < minimum or len(named[2]) < minimum:
                raise ValueError('accepted calibration lacks independent holdout games')
            if a.get('provenance_license') != OFFICIAL_LICENSE:
                raise ValueError('accepted calibration is not marked official CC0')

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))

    @property
    def usable(self):
        return self.artifact.get('status') == 'accepted'

    @property
    def confidence(self):
        if not self.usable:
            return 0.
        games = len(self.artifact['groups']['test'])
        # A small test corpus is useful for plumbing, not a strong calibration
        # certificate. Keep transfer from human games to analysis positions weak.
        return .75 if games >= 1000 else .5 if games >= 200 else .25

    @property
    def excluded_families(self):
        return frozenset(self.artifact.get('excluded_families', []))

    @property
    def sha256(self):
        return self.artifact['sha256']

    def predict(self, cp_white, turn):
        if not self.usable:
            raise ValueError('calibration did not pass independent validation')
        p = self.artifact['parameters']
        values = white_distribution(cp_white, p['scale_cp'], p['draw_margin_cp'])
        return values if bool(turn) else values[::-1]

    def calibrated_distribution(self, cp_white, turn=True, mate=None):
        if mate is not None:
            mate = int(mate)
            if mate == 0:
                raise ValueError('mate zero has no recoverable sign in nonterminal dump positions')
            win_white = mate > 0
            return [1., 0., 0.] if win_white == bool(turn) else [0., 0., 1.]
        return self.predict(cp_white, turn)


def save_calibration(path, artifact):
    artifact = dict(artifact)
    artifact['sha256'] = artifact_digest(artifact)
    Calibration(artifact)
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path)+'.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(artifact, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary, path)
    return artifact


def _split_group(game_id, seed):
    value = int(hashlib.sha256(f'calibration:{seed}:{game_id}'.encode()).hexdigest()[:16], 16) % 100
    return 'fit' if value < 60 else 'validation' if value < 80 else 'test'


def calibration_metrics(rows, parameters, *, prior=None):
    """Game-equal NLL, three-class Brier and top-class reliability ECE."""
    if not rows:
        return {'rows': 0, 'games': 0, 'nll': None, 'brier': None, 'ece': None}
    game_counts = Counter(str(r['game_id']) for r in rows)
    sums = {'nll': 0., 'brier': 0.}
    bins = [[0., 0., 0.] for _ in range(10)]
    total_weight = 0.
    for r in rows:
        weight = 1. / game_counts[str(r['game_id'])]
        values = list(prior) if prior is not None else white_distribution(r['cp_white'], parameters['scale_cp'], parameters['draw_margin_cp'])
        label = RESULT_CLASS[r['result']]
        sums['nll'] -= weight*math.log(max(1e-12, values[label]))
        sums['brier'] += weight*sum((p-float(i==label))**2 for i,p in enumerate(values))
        confidence = max(values);choice = values.index(confidence);bucket = min(9, int(confidence*10))
        bins[bucket][0] += weight;bins[bucket][1] += weight*confidence;bins[bucket][2] += weight*float(choice==label)
        total_weight += weight
    return {'rows': len(rows), 'games': len(game_counts),
            'nll': sums['nll']/total_weight, 'brier': sums['brier']/total_weight,
            'ece': sum(abs(p-y) for n,p,y in bins if n)/total_weight,
            'weighting': 'each completed game has total weight one'}


def fit_calibration(rows: Iterable[dict], seed=2026, min_holdout_games=10, steps=300):
    """Fit on separate games, select by validation NLL, report untouched test.

    Result labels come from completed games, not from CP. All source-position
    families are exported for exclusion from neural-network training.
    """
    import torch
    if not 1 <= int(steps) <= 10000 or int(min_holdout_games) < 1:
        raise ValueError('invalid calibration budget')
    records=[];seen=set();groups=defaultdict(set)
    for row in rows:
        if row.get('result') not in RESULT_CLASS:
            raise ValueError('calibration requires completed WDL game results')
        cp = float(row['cp_white'])
        if not math.isfinite(cp) or not str(row.get('game_id', '')):
            raise ValueError('invalid calibration row')
        source=row.get('source', {})
        if source.get('license') != OFFICIAL_LICENSE:
            raise ValueError('calibration rows require official CC0 provenance')
        identity=(str(row['game_id']), str(row.get('fen', '')), int(row.get('ply', 0)))
        if identity in seen:continue
        seen.add(identity);record=dict(row, cp_white=cp)
        part=_split_group(str(row['game_id']), seed);record['_part']=part
        records.append(record);groups[part].add(str(row['game_id']))
    if not records:raise ValueError('no independent calibration rows')
    parts={part:[r for r in records if r['_part']==part] for part in ('fit','validation','test')}
    training=parts['fit']
    if not training:raise ValueError('no calibration fit games after deterministic split')
    game_counts=Counter(str(r['game_id']) for r in training)
    weights=torch.tensor([1./game_counts[str(r['game_id'])] for r in training], dtype=torch.float64)
    x=torch.tensor([max(-8000., min(8000., r['cp_white']))/400. for r in training], dtype=torch.float64)
    y=torch.tensor([RESULT_CLASS[r['result']] for r in training], dtype=torch.long)
    theta=torch.nn.Parameter(torch.tensor([0.5413248546,-2.252168461], dtype=torch.float64))
    optimizer=torch.optim.Adam([theta], lr=.05)
    prior_counts=[sum(1./game_counts[str(r['game_id'])] for r in training if RESULT_CLASS[r['result']]==i)+.5 for i in range(3)]
    prior=[v/sum(prior_counts) for v in prior_counts]
    best=None;best_validation=float('inf');history=[]
    for step in range(int(steps)):
        scale=torch.nn.functional.softplus(theta[0])+1e-3
        margin=torch.nn.functional.softplus(theta[1])
        win=torch.sigmoid((x-margin)/scale);loss=torch.sigmoid((-x-margin)/scale)
        probs=torch.stack((win,(1.-win-loss).clamp_min(1e-12),loss), dim=1).clamp_min(1e-12)
        objective=-(weights*torch.log(probs[torch.arange(len(y)),y])).sum()/weights.sum()
        if not bool(torch.isfinite(objective)):raise RuntimeError('nonfinite calibration loss')
        optimizer.zero_grad(set_to_none=True);objective.backward()
        torch.nn.utils.clip_grad_norm_([theta],1.,error_if_nonfinite=True);optimizer.step()
        if not bool(torch.isfinite(theta).all()):raise RuntimeError('nonfinite calibration parameters')
        if step==0 or (step+1)%10==0 or step+1==int(steps):
            params={'scale_cp':float((torch.nn.functional.softplus(theta[0])+1e-3).detach())*400.,
                    'draw_margin_cp':float(torch.nn.functional.softplus(theta[1]).detach())*400.}
            metric=calibration_metrics(parts['validation'],params)
            selection=metric['nll'] if metric['nll'] is not None else float(objective.detach())
            history.append({'step':step+1,'fit_nll':float(objective.detach()),'validation_nll':metric['nll']})
            if selection < best_validation:
                best_validation=selection;best=params
    reports={part:calibration_metrics(part_rows,best) for part,part_rows in parts.items()}
    baseline={part:calibration_metrics(part_rows,best,prior=prior) for part,part_rows in parts.items()}
    reasons=[]
    for part in ('validation','test'):
        if len(groups[part])<int(min_holdout_games):reasons.append(f'insufficient_{part}_games')
    if len({r['result'] for r in training})<3:reasons.append('fit_lacks_all_three_outcomes')
    if reports['validation']['nll'] is None or reports['validation']['nll'] > baseline['validation']['nll']+1e-9:
        reasons.append('validation_nll_not_better_than_constant_prior')
    artifact={'format':FORMAT,'method':METHOD,'status':'accepted' if not reasons else 'unaccepted',
              'reasons':reasons,'parameters':best,'seed':int(seed),'steps':int(steps),
              'minimum_holdout_games':int(min_holdout_games),'provenance_license':OFFICIAL_LICENSE,
              'source_documents':sorted({canonical_json(r['source']) for r in records}),
              'corpus_sha256':hashlib.sha256(canonical_json([{k:r.get(k) for k in ('game_id','cp_white','result','fen','family','ply')} for r in records]).encode()).hexdigest(),
              'groups':{part:sorted(groups[part]) for part in ('fit','validation','test')},
              'excluded_families':sorted({str(r['family']) for r in records if r.get('family')}),
              'reports':reports,'constant_prior':prior,'constant_prior_reports':baseline,'history':history,
              'requires_training_exclusion':True,
              'interpretation':'empirical completed-game outcomes conditional on annotated CP; not perfect-play WDL or an Elo certificate',
              'domain_limits':['mixed Stockfish versions and human playing strengths','history and halfmove counters absent from eval dump',
                               'test games are independent of fit/validation groups; all listed families must be excluded from NN training']}
    artifact['sha256']=artifact_digest(artifact)
    Calibration(artifact)
    return artifact


def iter_pgn_calibration_rows(stream, provenance, max_games=5000, max_rows=20000,
                              min_ply=13, max_rows_per_game=8):
    """Bounded completed-game extraction; CP comments are always read as White POV.

    The caller supplies an already bounded streaming text source. Per-game
    examples are spread over the available evaluated plies, rather than allowing
    long games to dominate. Game IDs are used as indivisible split groups.
    """
    import chess
    import chess.pgn
    from .quality import family_for
    if provenance.get('license')!=OFFICIAL_LICENSE:
        raise ValueError('PGN calibration requires official CC0 provenance')
    if min(max_games,max_rows,max_rows_per_game)<1:raise ValueError('invalid extraction limits')
    emitted=0
    for game_index in range(max_games):
        game=chess.pgn.read_game(stream)
        if game is None:return
        if game.errors or game.headers.get('Result') not in RESULT_CLASS or game.headers.get('Variant','Standard') not in ('Standard','Chess'):
            continue
        board=game.board()
        if board.chess960 or not board.is_valid():continue
        candidates=[]
        for ply,node in enumerate(game.mainline(),1):
            if node.move not in board.legal_moves:
                candidates=[];break
            board.push(node.move)
            if ply<min_ply:continue
            score=node.eval()
            cp=score.white().score() if score is not None else None
            if cp is None or not math.isfinite(float(cp)):continue
            # Result tags describe what happened, not prospective draw claims.
            # Avoid can_claim_threefold_repetition's trial of every legal move
            # at every PGN ply; automatic game-over checks suffice here.
            if board.is_game_over(claim_draw=False):continue
            candidates.append({'cp_white':float(cp),'result':game.headers['Result'],
                               'fen':board.fen(en_passant='legal'),'family':family_for(board),'ply':ply})
        if not candidates:continue
        site=game.headers.get('Site','')
        match=re.search(r'https://lichess\.org/([A-Za-z0-9]{8})(?:\b|/)',site)
        game_id=match.group(1) if match else hashlib.sha256((canonical_json(dict(game.headers))+' '.join(m.uci() for m in game.mainline_moves())).encode()).hexdigest()
        count=min(max_rows_per_game,len(candidates))
        selected=sorted({round(i*(len(candidates)-1)/max(1,count-1)) for i in range(count)})
        for index in selected:
            row=dict(candidates[index],game_id=game_id,source=dict(provenance))
            yield row;emitted+=1
            if emitted>=max_rows:return
