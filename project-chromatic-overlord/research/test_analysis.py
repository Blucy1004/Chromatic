"""Integrity/measurement contract tests with synthetic journals, no chess engine."""
import gzip,hashlib,json,tempfile,unittest
from pathlib import Path
import analyze
import match_analysis
import chess.pgn

class IngestTests(unittest.TestCase):
    def fixture(self,root,mode='nodes',budget=2):
        z={'bestmove':'e2e4','search_ns':100000,'simulations':2,'neural_evals':3,'verification_evals':0,
           'attempted_simulations':2,'aborted_simulations':0,'max_depth':1,'prune_count':0,'nodes_evicted':0,
           'edges_evicted':0,'arena_nodes':3,'memory_bytes':1000,'peak_memory_bytes':1000,
           'verification_nodes':0,'verified_candidates':0,'verification_changed':False,'mate_proven':False,
           'proof_nodes':0,'importance':1,'discarded_neural_evaluations':0,'visits':[{'uci':'e2e4','n':2}]}
        q={'leaf_depth':[0,2]+[0]*511,'depth_samples':2,'node_creations':2,'edge_expansions':3,
           'mean_depth':1,'median_depth':1,'call_ns':100000,'algorithm_ns':90000,'pv':['e2e4'],
           'mcts_ns':80000,'verification_ns':0,'nn_mcts_ns':50000,'adaptive_calls':0}
        z['research']=q
        row={'fen':analyze.chess.STARTING_FEN,'id':'synthetic','expected':'e2e4','game':'source-game','rating':1500,
             'hit':True,'wall_ns':200000,'cpu_seconds':.001,'rss_after_bytes':2000,'result':z,
             'stage':'screen','model':'zero128','mask':0,'mode':mode,'budget':budget,'repeat':0,'cell':'cell'}
        model='2'*64;engine='1'*64;loaded={'loaded':True,'sha256':model}
        meta={'complete':True,'ids':['synthetic'],'model':'zero128','mask':0,'mode':mode,'budget':budget,
              'options':{},'model_sha256':model,'engine_sha256':engine,'loaded_model':loaded,'loaded_model_after':loaded}
        manifest={'os':'Linux','engine_sha256':engine,'models':{'zero128':{'sha256':model}}}
        (root/'manifest.json').write_text(json.dumps(manifest));(root/'screen').mkdir()
        self.write(root,row,meta);return row,meta
    def write(self,root,row,meta):
        raw=gzip.compress((json.dumps(row)+'\n').encode());meta['raw_gzip_sha256']=hashlib.sha256(raw).hexdigest()
        (root/'screen/cell.jsonl.gz').write_bytes(raw);(root/'screen/cell.meta.json').write_text(json.dumps(meta))
    def test_measurements_and_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root);out=[];r=analyze.ingest(root,out.append,'synthetic')
            self.assertEqual(len(out),1);s=r['cells'][0]['summary']
            self.assertEqual(s['hits'],1);self.assertEqual(s['mean_nn_total'],3)
            self.assertEqual(s['pooled_depth_mean'],1);self.assertEqual(s['mean_cleanup_ms'],.01)
            self.assertEqual(out[0]['nn_total'],out[0]['nn_mcts']+out[0]['nn_verification'])
            with (root/'screen/cell.jsonl.gz').open('ab') as f:f.write(b'changed')
            with self.assertRaises(AssertionError):analyze.ingest(root,lambda _:None,'synthetic')
    def test_bad_count_or_cap_rejected(self):
        for mode,budget,change in [('nodes',2,'nodes'),('nn',2,'cap'),('nodes',3,'visits')]:
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);row,meta=self.fixture(root,mode,budget)
                if change=='nodes':row['result']['arena_nodes']=1
                self.write(root,row,meta)
                with self.assertRaises(AssertionError):analyze.ingest(root,lambda _:None,'synthetic')
    def test_closing_identity_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);row,meta=self.fixture(root);meta['loaded_model_after']={'sha256':'3'*64};self.write(root,row,meta)
            with self.assertRaises(AssertionError):analyze.ingest(root,lambda _:None,'synthetic')
    def test_incomplete_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);row,meta=self.fixture(root);meta['complete']=False;self.write(root,row,meta)
            r=analyze.ingest(root,lambda _:self.fail('incomplete included'),'synthetic')
            self.assertEqual(len(r['cells']),0);self.assertEqual(len(r['incomplete_cells']),1)
class MatchTests(unittest.TestCase):
    def test_terminal_unfinished_and_pair_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'matches').mkdir();model='2'*64;engine='1'*64
            (root/'manifest.json').write_text(json.dumps({'os':'Linux','engine_sha256':engine,'models':{'zero128':{'sha256':model}}}))
            moves='e2e4 e7e5 d1h5 b8c6 f1c4 g8f6 h5f7'.split();board=analyze.chess.Board()
            for m in moves:board.push_uci(m)
            game=chess.pgn.Game.from_board(board);game.headers['Result']='1-0';records=[]
            for color in [True,False]:
                records.append({'model':'zero128','mask':15,'model_sha256':model,'engine_sha256':engine,
                    'error':None,'moves':moves,'pgn':str(game),'result':'1-0','score':float(color),
                    'candidate_white':color,'trace':[{'move':m} for m in moves],'plies':len(moves),'pair':0,
                    'threads':1,'movetime_ms':100,'termination':'CHECKMATE'})
            game=chess.pgn.Game();game.add_variation(analyze.chess.Move.from_uci('e2e4'))
            records.append({**records[0],'moves':['e2e4'],'pgn':str(game),'result':'*','score':None,'trace':[{'move':'e2e4'}],'plies':1,'pair':1,'termination':'ply_limit_unfinished'})
            (root/'matches/zero128-m15.json').write_text(json.dumps(records))
            summary,flat=match_analysis.analyze(root);s=summary[0]
            self.assertEqual((s['finished'],s['unfinished'],s['complete_opening_pairs']),(2,1,1))
            self.assertEqual(s['paired_score'],.5);self.assertEqual(s['descriptive_elo'],0)
            self.assertEqual(s['pair_hoeffding_ci95'],[0,1]);self.assertFalse(s['confirmed_elo_gain'])
            records[0]['score']=0;(root/'matches/zero128-m15.json').write_text(json.dumps(records))
            with self.assertRaises(AssertionError):match_analysis.analyze(root)
    def test_global_holdout_families(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'selection-plan.json').write_text(json.dumps({'best_serial':{'zero128':4,'lichess100':4},'primary_pair':[0,2]}))
            cells=[]
            for model in ['zero128','lichess100']:
                for mask in [0,1,4,5]:
                    values=[0,0,1,1] if mask in [0,1] else [1,1,1,1]
                    cells.append({'summary':{'stage':'holdout','model':model,'mask':mask},
                        'cases':[{'id':str(i),'game':str(i),'hit':v} for i,v in enumerate(values)]})
            runs=[{'root':str(root),'scope':os,'manifest':{'os':os},'cells':cells} for os in ['Linux','Windows']]
            selected=analyze.selected_holdout(runs,200);self.assertEqual(len(selected),4)
            for r in selected:
                self.assertEqual(r['family_size'],4);self.assertEqual((r['gains'],r['losses']),(2,0));self.assertEqual(r['delta_pp'],50)
                self.assertEqual(r['p_holm_family'],1)
            primary=analyze.holdout_confirmation(runs,200);self.assertEqual(len(primary),4)
            for r in primary:
                self.assertEqual(r['synergy_pp'],0);self.assertEqual(r['simultaneous_family_size'],4)
                self.assertEqual(r['component_test_family_size'],8);self.assertFalse(r['positive_synergy_confirmed'])
if __name__=='__main__':unittest.main()
