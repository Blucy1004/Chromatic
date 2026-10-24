import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shardtrain.evaluate import (PolicyWDL,ModelConfig,torch,start_consistency,
                                 outcome_metrics,promotion_assessment,_actual_class,evaluate_run,checkpoint,
                                 common_holdout_report)
from shardtrain.storage import Ledger,ResourceGuard,canonical,digest
from shardtrain.durable import save_bundle
import chess


def report_fixture():
    metrics={'validation':{'rows':1000,'policy_loss':1.,'wdl_loss':1.},
             'test':{'rows':1000,'policy_loss':1.,'wdl_loss':1.},
             'outcome_calibration':{'brier':.5,'ece':.2},'tactics':{'engine_errors':0,'accuracy':.5}}
    candidate=copy.deepcopy(metrics)
    candidate['validation']['policy_loss']=.9;candidate['outcome_calibration']['brier']=.4
    return {'models':{'champion':metrics,'candidate':candidate},'calibration':{'accepted':True},
            'source_integrity_verified':True,'leakage':{'passed':True},'candidate_head_stable':True,
            'match':{'completed_games':64,'complete':True,'engine_errors':0,'lower_bound':.51}}


class EvaluationTests(unittest.TestCase):
    def model(self,bias):
        model=PolicyWDL(ModelConfig(width=2,blocks=0))
        with torch.no_grad():
            for p in model.parameters():p.zero_()
            model.wdl.bias.copy_(torch.tensor(bias,dtype=torch.float32))
        return model

    def test_start_all_twenty_children_sign_and_probability(self):
        measured=start_consistency(self.model([8.,-8.,-8.]))
        self.assertEqual(measured['legal_moves'],20)
        self.assertAlmostEqual(sum(x['policy'] for x in measured['children']),1.,places=6)
        self.assertTrue(measured['all_child_backups_negative'])
        self.assertGreater(measured['absolute_bellman_residual'],1.99)
        draw=start_consistency(self.model([-8.,8.,-8.]))
        self.assertEqual(draw['value'],0.);self.assertEqual(draw['absolute_bellman_residual'],0.)

    def test_actual_outcome_viewpoint_is_side_to_move(self):
        self.assertEqual(_actual_class('1-0',True),0)
        self.assertEqual(_actual_class('1-0',False),2)
        self.assertEqual(_actual_class('1/2-1/2',False),1)

    def test_game_equal_outcome_metrics(self):
        model=self.model([0.,0.,0.])
        root='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
        data=[{'fen':root,'game_id':'first','result':'1-0'} for _ in range(4)]
        data.append({'fen':root,'game_id':'second','result':'0-1'})
        result=outcome_metrics(model,data)
        self.assertEqual(result['games'],2);self.assertEqual(result['rows'],5)
        self.assertAlmostEqual(result['brier'],2/3,places=6)
        self.assertAlmostEqual(result['ece'],1/6,places=6)

    def test_small_or_incomplete_match_never_promotes(self):
        data=report_fixture();data['match'].update(completed_games=3,complete=False,lower_bound=None)
        result=promotion_assessment(data)
        self.assertFalse(result['eligible']);self.assertFalse(result['performed'])
        self.assertIn('match_incomplete_or_engine_error',result['reasons'])

    def test_positive_eligibility_still_does_not_overwrite(self):
        result=promotion_assessment(report_fixture())
        self.assertTrue(result['eligible']);self.assertFalse(result['performed']);self.assertFalse(result['automatic_overwrite'])

    def test_any_independent_gate_failure_blocks(self):
        for modification in ['source','leak','head','calibration','worse','nonfinite','tactical','holdout']:
            with self.subTest(modification=modification):
                report=report_fixture()
                if modification=='source':report['source_integrity_verified']=False
                elif modification=='leak':report['leakage']['passed']=False
                elif modification=='head':report['candidate_head_stable']=False
                elif modification=='calibration':report['calibration']['accepted']=False
                elif modification=='worse':report['models']['candidate']['test']['wdl_loss']=2.
                elif modification=='nonfinite':report['models']['candidate']['validation']['policy_loss']=float('nan')
                elif modification=='tactical':report['models']['candidate']['tactics']['engine_errors']=1
                else:report['models']['candidate']['validation']['rows']=99
                self.assertFalse(promotion_assessment(report)['eligible'])

    def test_manifest_integration_is_read_only_and_keeps_candidate_only(self):
        base=Path(__file__).resolve().parents[3]/'work/evaluation-test-temp';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temporary:
            root=Path(temporary).resolve();self.assertTrue(root.is_relative_to(base.resolve()))
            data=root/'D';checkpoints=root/'C';fixtures=root/'F';fixtures.mkdir()
            ledger=Ledger(data/'ledger.sqlite3')
            try:
                model=self.model([0.,0.,0.]);generator=torch.Generator().manual_seed(17)
                settings={'source_provenance':{'original_frame_integrity':'verified_complete_zstd_frames'}}
                state={'config':{'width':2,'blocks':0},'model':model.state_dict(),'optimizer':None,
                       'step':0,'generator':generator.get_state(),'dataset':[],'settings':settings,
                       'scheduler':None,'progress':{'completed':[],'active':None,'epoch':0,'offset':0,'permutation':[],'positions_processed':0}}
                ledger.put('settings',settings)
                board=chess.Board()
                for number,split in enumerate(['validation','test']):
                    if number:board.push_uci('e2e4')
                    move=next(iter(board.legal_moves));identity='fixture-'+split
                    from shardtrain.evaluate import action
                    row={'root_fen':board.fen(),'moves':[],'fen':board.fen(),'policy':[[action(move),1.]],
                         'wdl':[0.,1.,0.],'policy_weight':1.,'value_weight':1.}
                    ledger.db.execute('INSERT INTO records(id,family,bucket,phase,split,payload) VALUES (?,?,?,?,?,?)',
                                      (identity,identity,identity,'opening',split,canonical(row)))
                _,head=save_bundle(ledger,checkpoints,model,state,ResourceGuard([root],1.,4.),{'kind':'fixture'})
                champion=root/'champion.pt';champion_zero=root/'champion.zero';engine=root/'fake-engine.bin'
                checkpoint.save(champion,state);model.export(champion_zero);engine.write_bytes(b'no engine invoked in mocked integration')
                (fixtures/'official-puzzles.csv.source.json').write_text(json.dumps({'license':'CC0-1.0','excluded_training_families':['reserved']}),encoding='utf-8')
                calibration=SimpleNamespace(artifact={'groups':{'test':[]},'reports':{}},excluded_families=set(),usable=True,sha256='fixture')
                tactics={'positions':20,'accuracy':.5,'engine_errors':0,'correct':10,'results':[]}
                match={'completed_games':0,'complete':False,'engine_errors':0,'lower_bound':None,'games':[]}
                before=(digest(champion),digest(champion_zero),ledger.get('current_manifest'));rng=torch.get_rng_state().clone()
                with patch('shardtrain.evaluate.Calibration.load',return_value=calibration),\
                     patch('shardtrain.evaluate._independent_outcomes',return_value=([],{'license':'CC0-1.0'})),\
                     patch('shardtrain.evaluate._tactics',return_value=tactics),\
                     patch('shardtrain.evaluate.arena',return_value=match):
                    report=evaluate_run(data,checkpoints,engine,champion,champion_zero,fixtures)
                self.assertEqual(before,(digest(champion),digest(champion_zero),ledger.get('current_manifest')))
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                self.assertTrue(report['preservation']['unchanged']);self.assertFalse(report['promotion']['eligible'])
                self.assertFalse(report['promotion']['performed']);self.assertEqual(report['models']['candidate']['test']['rows'],1)
                self.assertTrue((checkpoints/'evaluation-report.json').is_file())
                common=common_holdout_report(data,{'candidate':(data,checkpoints)},champion,root/'common.json')
                self.assertTrue(common['champion_unchanged']);self.assertTrue((root/'common.csv').is_file())
                self.assertEqual(common['models']['champion']['test']['identity_sha256'],
                                 common['models']['candidate']['test']['identity_sha256'])
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                ledger.db.execute("INSERT INTO seen_families(family,split) VALUES ('fixture-validation','train')")
                with self.assertRaisesRegex(ValueError,'leaked'):
                    common_holdout_report(data,{'candidate':(data,checkpoints)},champion,root/'leaked.json')
                self.assertFalse((root/'leaked.json').exists())
            finally:ledger.close()


if __name__=='__main__':unittest.main()
