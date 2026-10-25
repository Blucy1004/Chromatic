import importlib.util,math,unittest
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('factorial_statistics',Path(__file__).with_name('factorial_stats.py'))
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
class StatisticsTests(unittest.TestCase):
    def test_mcnemar_exact(self):
        self.assertEqual(s.mcnemar(0,0),1);self.assertEqual(s.mcnemar(5,0),.0625)
        self.assertEqual(s.mcnemar(3,3),1);self.assertEqual(s.mcnemar(8,1),.0390625)
    def test_holm_monotone(self):
        np.testing.assert_allclose(s.holm([.01,.04,.03]),[.03,.06,.06])
    def test_additive_and_synergy(self):
        weights=np.array([.1,.2,-.3,.4,.05]);cells={m:sum(weights[i] for i in range(5) if m>>i&1) for m in range(32)}
        np.testing.assert_allclose(s.shapley(cells),weights)
        self.assertAlmostEqual(float(s.mobius(cells,7)),0)
        for m in cells:
            if m&3==3:cells[m]+=.12
        np.testing.assert_allclose(s.shapley(cells),weights+[.06,.06,0,0,0])
        self.assertAlmostEqual(float(s.mobius(cells,3)),.12)
        self.assertAlmostEqual(sum(s.shapley(cells)),cells[31]-cells[0])
        np.testing.assert_allclose(s.shapley(cells,range(4),16),weights[:4]+[.06,.06,0,0])
    def test_cluster_weighting_and_paired(self):
        r=s.cluster_bootstrap([1,1,0],['a','a','b'],repetitions=1000)
        self.assertEqual(r['clusters'],2);self.assertAlmostEqual(r['mean'][0],2/3)
        r=s.paired([1,0,1],[0,1,0],repetitions=1000)
        self.assertEqual((r['gains'],r['losses']),(2,1));self.assertAlmostEqual(r['difference'],1/3)
    def test_constant_uncertainty(self):
        r=s.cluster_bootstrap(np.ones((5,3)),repetitions=1000)
        np.testing.assert_allclose(r['simultaneous_ci95'],np.ones((3,2)))
if __name__=='__main__':unittest.main()
