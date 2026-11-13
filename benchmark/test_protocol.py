"""Protocol invariants; run before a full benchmark."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from design import SCENARIOS, ALL_SCENARIOS, create_data
from run import metric_table, objective, model_candidates
from balanced_prior import BalancedPriorCorrectedRF
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

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

    def test_classical_and_undersampled_forest_baselines_exist(self):
        expected={
            "RandomForest","RF_Bootstrap50","RF_Subsample50",
            "RF_Subsample80","RF_BalancedUndersampled",
            "RF_BalancedUndersampledPrior"}
        self.assertTrue(expected.issubset(model_candidates(11)))
        self.assertEqual(len(model_candidates(11)),23)

    def test_no_replacement_and_class_balance_are_genuine(self):
        rng=np.random.default_rng(12)
        X=rng.normal(size=(200,4))
        y=np.concatenate((np.zeros(180,dtype=int),np.ones(20,dtype=int)))
        model=BaggingClassifier(
            estimator=DecisionTreeClassifier(max_features="sqrt"),
            n_estimators=5,max_samples=0.5,bootstrap=False,random_state=11)
        model.fit(X,y)
        for selected in model.estimators_samples_:
            self.assertEqual(len(selected),100)
            self.assertEqual(len(np.unique(selected)),100)
        brf=BalancedRandomForestClassifier(
            n_estimators=5,bootstrap=False,replacement=False,
            sampling_strategy="all",random_state=11)
        brf.fit(X,y)
        for sampler in brf.samplers_:
            labels=y[sampler.sample_indices_]
            self.assertEqual(np.bincount(labels,minlength=2).tolist(),[20,20])

    def test_prior_adjustment_uses_training_prevalence(self):
        rng=np.random.default_rng(13)
        X=rng.normal(size=(600,3))
        y=(rng.random(600)<0.10).astype(int)
        model=BalancedPriorCorrectedRF(
            n_estimators=25,random_state=13)
        model.fit(X,y)
        prob=model.predict_proba(X[:20])
        raw=model.model_.predict_proba(X[:20])
        self.assertTrue(np.isfinite(prob).all())
        self.assertTrue(np.allclose(prob.sum(axis=1),1))
        self.assertLess(np.mean(prob[:,1]),np.mean(raw[:,1]))
        mask=(raw[:,1]>0)&(raw[:,1]<1)
        self.assertTrue(np.all(prob[mask,1]<raw[mask,1]))

if __name__=="__main__":
    unittest.main()
