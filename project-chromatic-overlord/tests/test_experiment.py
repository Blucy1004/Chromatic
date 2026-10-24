import importlib.util,sys,unittest
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'tools/experiment.py'
spec=importlib.util.spec_from_file_location('overlord_experiment',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class Results(unittest.TestCase):
 def test_unfinished_never_becomes_a_draw(self):
  r=module.stats([{'pair':0,'score':1},{'pair':0,'score':None}])
  self.assertEqual((r['finished'],r['unfinished'],r['draws']),(1,1,0))
  self.assertEqual(r['complete_opening_pairs'],0);self.assertEqual(r['paired_score_interval_95_conservative'],[0,1])
  self.assertFalse(r['promote']);self.assertIsNone(r['elo_claim'])
 def test_perfect_tiny_sample_retains_uncertainty(self):
  r=module.stats([{'pair':i//2,'score':1} for i in range(8)])
  self.assertLess(r['paired_score_interval_95_conservative'][0],.5)
  self.assertFalse(r['promote'])
 def test_color_pair_is_the_unit(self):
  r=module.stats([{'pair':0,'score':1},{'pair':0,'score':0}])
  self.assertEqual(r['score'],.5);self.assertEqual(r['complete_opening_pairs'],1)
if __name__=='__main__':unittest.main()
