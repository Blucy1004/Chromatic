import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import chess

from shardtrain.calibration import (Calibration, artifact_digest, calibration_metrics,
    fit_calibration, iter_pgn_calibration_rows, normalize_white_cp_to_stm,
    save_calibration, white_distribution)
from shardtrain.quality import (ACTIONS, QualityConfig, QualityError, action,
    board_from_fen, family_for, identity_for, parse_record, phase_for)


PROVENANCE={'url':'https://database.lichess.org/standard/unit-fixture.pgn.zst',
            'license':'CC0-1.0','unit_test_fixture':True}

def row(fen=None,pvs=None,depth=24,knodes=100):
    return {'fen':fen or ' '.join(chess.STARTING_FEN.split()[:4]),
            'evals':[{'depth':depth,'knodes':knodes,'pvs':pvs or [{'cp':20,'line':'e2e4 e7e5'}]}]}


class QualityTests(unittest.TestCase):
    def test_uncalibrated_cp_does_not_invent_draw_target(self):
        result=parse_record(row())
        self.assertEqual(result['value_weight'],0.)
        self.assertEqual(result['value_source'],'unavailable_no_accepted_calibration')
        self.assertEqual(result['policy_kind'],'partial_pv_imitation')
        self.assertFalse(result['policy_support_complete'])
        self.assertEqual(result['observed_policy_moves'],1)
        self.assertGreater(result['policy_weight'],0.)
        self.assertLess(result['policy_weight'],.25)
        self.assertFalse(result['history_known'])
        self.assertIn('fullmove_number',result['history_unknown_fields'])
        self.assertEqual(result['source_eval']['nodes'],100000)

    def test_white_cp_is_sorted_from_black_view(self):
        b=chess.Board();b.turn=False
        r=parse_record(row(' '.join(b.fen().split()[:4]),pvs=[
            {'cp':100,'line':'e7e5'},{'cp':-50,'line':'d7d5'}]))
        self.assertEqual(r['policy'][0][0],action(chess.Move.from_uci('d7d5')))
        self.assertGreater(r['policy'][0][1],r['policy'][1][1])
        self.assertAlmostEqual(sum(p for _,p in r['policy']),1.)
        self.assertEqual(normalize_white_cp_to_stm(100,False),-100.)

    def test_castling_king_to_rook_is_normalized(self):
        fen='r3k2r/8/8/8/8/8/8/R3K2R w KQkq -'
        r=parse_record(row(fen,pvs=[{'cp':30,'line':'e1h1 e8a8'}]))
        self.assertEqual(r['policy'][0][0],action(chess.Move.from_uci('e1g1')))
        self.assertEqual(r['source_eval']['pv_moves_uci'],['e1g1'])
        self.assertEqual(r['source_eval']['pv_plies_checked'],[2])

    def test_underpromotion_action_matches_existing_encoder(self):
        fen='4k3/P7/8/8/8/8/8/4K3 w - -'
        r=parse_record(row(fen,pvs=[{'cp':0,'line':'a7a8n'},{'cp':50,'line':'a7a8q'}]))
        ids={x[0] for x in r['policy']}
        self.assertIn(4531,ids);self.assertIn(action(chess.Move.from_uci('a7a8q')),ids)
        self.assertTrue(all(0<=i<ACTIONS for i in ids))

    def test_mate_is_separate_and_white_to_stm(self):
        b=chess.Board();b.turn=False
        r=parse_record(row(' '.join(b.fen().split()[:4]),pvs=[{'mate':-3,'line':'e7e5'}]))
        self.assertEqual(r['wdl'],[1.,0.,0.])
        self.assertGreater(r['value_weight'],0.)
        self.assertIn('mate_sign',r['value_source'])
        with self.assertRaises(QualityError):parse_record(row(pvs=[{'mate':0,'line':'e2e4'}]))

    def test_invalids_and_full_pv_are_rejected(self):
        for raw in [row(depth=2),row(knodes=1),row(pvs=[{'cp':float('nan'),'line':'e2e4'}]),
                    row(pvs=[{'cp':20,'line':'e2e4 e7e4'}]),
                    row(pvs=[{'cp':20,'line':'e2e4'},{'cp':10,'line':'e2e4'}]),
                    row('8/8/8/8/8/8/8/8 w - -')]:
            with self.subTest(raw=raw),self.assertRaises(QualityError):parse_record(raw)
        raw=row(pvs=[{'cp':20,'line':'e2e4 e7e4'}])
        raw['evals'].append({'depth':18,'knodes':100,'pvs':[{'cp':10,'line':'d2d4 d7d5'}]})
        r=parse_record(raw)
        self.assertEqual(r['source_eval']['depth'],18)
        self.assertIn('illegal_pv',r['source_eval']['rejected_evaluations'])

    def test_no_ply_from_four_field_fen_and_broad_family(self):
        b=board_from_fen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -')
        self.assertEqual(phase_for(b),'opening')
        c=b.copy();c.turn=False;c.castling_rights=0
        self.assertEqual(family_for(b),family_for(c))
        self.assertNotEqual(identity_for(b),identity_for(c))
        r=parse_record(row())
        self.assertEqual(r['phase_basis'],'material_proxy_not_game_ply')

    def test_non_capturable_ep_has_canonical_identity(self):
        b=chess.Board();b.push_uci('e2e4')
        a=board_from_fen(b.fen(en_passant='fen'))
        c=board_from_fen(b.fen(en_passant='legal'))
        self.assertEqual(identity_for(a),identity_for(c))
        self.assertIsNone(a.ep_square)


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Synthetic only: validates code mechanics, never reported as real data.
        cls.rows=[]
        for i in range(120):
            result=['1-0','1/2-1/2','0-1'][i%3]
            cp=[500,0,-500][i%3]
            for j in range(2):
                cls.rows.append({'game_id':f'unit-{i:04d}','cp_white':cp+j,'result':result,
                                 'fen':f'unit-{i}-{j}','family':f'family-{i}','ply':13+j,'source':PROVENANCE})
        cls.artifact=fit_calibration(cls.rows,seed=2026,min_holdout_games=5,steps=60)

    def test_grouped_fit_has_independent_test_and_is_usable(self):
        a=self.artifact;c=Calibration(a)
        self.assertTrue(c.usable,a['reasons'])
        self.assertTrue(set(a['groups']['fit']).isdisjoint(a['groups']['validation']))
        self.assertTrue(set(a['groups']['test']).isdisjoint(a['groups']['validation']))
        self.assertEqual(len(c.excluded_families),120)
        self.assertLess(a['reports']['test']['nll'],a['constant_prior_reports']['test']['nll'])
        w=c.predict(200,True);b=c.predict(200,False)
        self.assertAlmostEqual(sum(w),1.)
        self.assertEqual(w,b[::-1]);self.assertGreater(w[0],w[2])
        self.assertLessEqual(c.confidence,.25)

    def test_accepted_calibration_can_supply_weighted_stm_value(self):
        a=Calibration(self.artifact);b=chess.Board();b.turn=False
        r=parse_record(row(' '.join(b.fen().split()[:4]),pvs=[{'cp':200,'line':'e7e5'}]),calibration=a)
        self.assertGreater(r['value_weight'],0.)
        self.assertLess(r['wdl'][0],r['wdl'][2])
        self.assertEqual(r['calibration_sha256'],a.sha256)

    def test_atomic_roundtrip_corruption_and_overlap(self):
        temporary_root=Path(__file__).resolve().parents[3]/'work'/'label-test-temp'
        temporary_root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(temporary_root.resolve()))
            path=Path(directory)/'calibration.json'
            save_calibration(path,self.artifact)
            self.assertTrue(Calibration.load(path).usable)
            value=json.loads(path.read_text(encoding='utf-8'));value['parameters']['scale_cp']*=2
            path.write_text(json.dumps(value),encoding='utf-8')
            with self.assertRaises(ValueError):Calibration.load(path)
        value=json.loads(json.dumps(self.artifact));value['groups']['test'].append(value['groups']['fit'][0]);value['sha256']=artifact_digest(value)
        with self.assertRaises(ValueError):Calibration(value)

    def test_insufficient_holdout_never_activates_value(self):
        a=fit_calibration(self.rows[:20],min_holdout_games=100,steps=10)
        c=Calibration(a);self.assertFalse(c.usable)
        r=parse_record(row(),calibration=c)
        self.assertEqual(r['value_weight'],0.)
        with self.assertRaises(ValueError):c.predict(10,True)

    def test_probability_range_symmetry_and_metric(self):
        for cp in [-100000,-500,0,500,100000]:
            p=white_distribution(cp,200,80);opposite=white_distribution(-cp,200,80)
            self.assertTrue(all(math.isfinite(x) and 0<=x<=1 for x in p))
            self.assertAlmostEqual(sum(p),1.)
            self.assertAlmostEqual(p[0],opposite[2])
        m=calibration_metrics(self.rows,self.artifact['parameters'])
        self.assertTrue(0<=m['ece']<=1 and 0<=m['brier']<=2)

    def test_pgn_extract_reads_white_pov_and_completed_game_groups(self):
        pgn='''[Event "Unit test"]
[Site "https://lichess.org/AbCd1234"]
[Result "1-0"]

1. e4 { [%eval 1.23] } e5 { [%eval -0.50] } 2. Nf3 { [%eval 0.25] } Nc6 1-0
'''
        rows=list(iter_pgn_calibration_rows(io.StringIO(pgn),PROVENANCE,min_ply=1,max_rows_per_game=8))
        self.assertEqual([r['cp_white'] for r in rows],[123.,-50.,25.])
        self.assertTrue(all(r['game_id']=='AbCd1234' and r['result']=='1-0' for r in rows))
        self.assertEqual(chess.Board(rows[0]['fen']).turn,False)
        with self.assertRaises(ValueError):list(iter_pgn_calibration_rows(io.StringIO(pgn),{'license':'unknown'}))


if __name__=='__main__':unittest.main()
