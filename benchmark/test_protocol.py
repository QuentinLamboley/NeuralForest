"""Protocol invariants; run before a full benchmark."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from design import SCENARIOS, ALL_SCENARIOS, create_data
from run import metric_table, objective

class BenchmarkProtocolTests(unittest.TestCase):
    def test_scenario_count_and_pinned_labels(self):
        self.assertEqual(len(ALL_SCENARIOS),19)
        self.assertEqual(sum(map(len,SCENARIOS.values())),19)
        self.assertEqual(ALL_SCENARIOS["identical_100"].proximity,100)

    def test_independent_generated_train_test(self):
        spec=ALL_SCENARIOS["intermediate_45"]
        train,test,metadata=create_data(spec,91)
        self.assertEqual(len(train),spec.n)
        self.assertNotEqual(train.iloc[0].to_dict(),test.iloc[0].to_dict())
        self.assertEqual(metadata["source_commit"],
                         "48c2b28ca7874cbef1f7dde15de962c6e56c5027")

    def test_stratified_counts_and_stress_missingness(self):
        source,test,_=create_data(ALL_SCENARIOS["missing_15pct"],89)
        self.assertGreater(source.isna().sum().sum(),0)
        self.assertGreater(test.isna().sum().sum(),0)
        self.assertEqual(list(source),list(test))
        self.assertEqual(source["Vampirisme"].nunique(),2)

    def test_presence_background_retains_external_true_labels(self):
        source,test,meta=create_data(ALL_SCENARIOS["presence_bg_1to5"],101)
        self.assertIn("unlabeled background",meta["label_semantics"])
        self.assertEqual(source["Vampirisme"].nunique(),2)
        self.assertEqual(test["Vampirisme"].nunique(),2)

    def test_metric_rank_directions(self):
        y=np.array([0,0,1,1])
        p=np.array([.05,.1,.9,.95])
        scores=metric_table(y,p)
        self.assertAlmostEqual(scores["roc_auc"],1.0)
        self.assertAlmostEqual(scores["pr_auc"],1.0)
        self.assertTrue(np.isfinite(scores["logloss"]))
        self.assertAlmostEqual(objective(scores,"auc"),1.0)

if __name__=="__main__":
    unittest.main()
