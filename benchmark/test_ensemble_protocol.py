"""Unit and protocol checks for scored ensembles and all test metrics."""
import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.special import expit,logit
from sklearn.ensemble import VotingClassifier,StackingClassifier
from sklearn.pipeline import Pipeline

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run import (model_candidates,metric_table,boyce_rank,
                 estimator_with_preprocessing,METRICS,SELECTION_SEEDS)
from multimetric import paired_ci,METRICS as REPORT_METRICS

ENSEMBLES={
    "Voting_RF_ExtraTrees","Voting_RF_CatBoost","Voting_Boosting",
    "Voting_Heterogeneous","Stacking_RF_XGB","Stacking_Boosting"}

class EnsembleProtocolTests(unittest.TestCase):
    def test_all_six_ensemble_families_exist(self):
        candidates=model_candidates(803)
        self.assertEqual(len(candidates),23)
        self.assertTrue(ENSEMBLES.issubset(candidates))
        self.assertEqual(len(SELECTION_SEEDS),10)
        self.assertTrue(set(REPORT_METRICS).issubset(set(METRICS)))
        for name in ENSEMBLES:
            estimator=candidates[name][0]()
            if name.startswith("Stacking_"):
                self.assertIsInstance(estimator,StackingClassifier)
                self.assertEqual(estimator.cv.n_splits,3)
                self.assertTrue(all(isinstance(learner,Pipeline)
                         for _,learner in estimator.estimators))
                self.assertIs(estimator_with_preprocessing(estimator),estimator)
            else:
                self.assertIsInstance(estimator,VotingClassifier)
                self.assertEqual(estimator.voting,"soft")

    def test_rank_boyce_is_monotone_calibration_invariant(self):
        rng=np.random.default_rng(806)
        p=np.clip(rng.uniform(.03,.97,size=320),.03,.97)
        y=rng.binomial(1,p)
        corrected=expit(1.8*logit(p)-0.9)
        self.assertAlmostEqual(boyce_rank(y,p),boyce_rank(y,corrected),
                               places=12)
        self.assertTrue(np.isfinite(metric_table(y,p)["ece"]))

    def test_top_decile_retrieval_and_all_metrics_finite(self):
        y=np.array([1]*12+[0]*108)
        p=np.linspace(.99,.01,120)
        values=metric_table(y,p)
        self.assertEqual(values["precision_at_10pct"],1.)
        self.assertEqual(values["recall_at_10pct"],1.)
        self.assertTrue(set(REPORT_METRICS).issubset(values))
        self.assertTrue(all(np.isfinite(values[k]) for k in REPORT_METRICS
                            if k not in ("boyce","boyce_rank")))
        # If all presences concentrate in fewer than three score bins,
        # Boyce is undefined by design. Do not invent a finite statistic.

    def test_paired_uncertainty_has_favorable_direction(self):
        mu,lo,hi,n=paired_ci([.01,.015,.02,.008,.025,.017])
        self.assertEqual(n,6)
        self.assertLess(lo,mu)
        self.assertLess(mu,hi)
        self.assertGreater(mu,0.)
        mu,lo,hi,n=paired_ci([np.nan])
        self.assertEqual(n,0)
        self.assertTrue(np.isnan(mu))

if __name__=="__main__":
    unittest.main()
