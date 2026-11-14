"""No-label leakage and spatial geometry for effort-corrected SDM."""
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"benchmark"))
from po_effort_design import (
 LANDSCAPES,EFFORTS,SEEDS,OUTER_FOLDS,
 PREDICTORS,cases,surface,validate_case)
from po_effort_run import R_MODELS,PY_MODELS,ALL_MODELS

class IndependentEffortChallenge(unittest.TestCase):
 def test_prespecified_synthetic_habitat_and_effort_regimes(self):
  self.assertEqual(LANDSCAPES,("ecotone","interaction"))
  self.assertEqual(EFFORTS,("uniform","biased"))
  self.assertEqual(SEEDS,(11711,11807,11909))
  self.assertEqual(OUTER_FOLDS,(1,2,3))
  self.assertEqual(len(ALL_MODELS),13)
  self.assertIn("IPP_gradient_true_random",R_MODELS)
  self.assertIn("CatBoost_effort_weighted",PY_MODELS)

 def test_all_train_inner_test_blocks_are_disjoint(self):
  for scenario in LANDSCAPES:
   for detection in EFFORTS:
    for seed in SEEDS:
     seen=set()
     for case in cases(scenario,detection,seed):
      self.assertTrue(validate_case(case))
      self.assertNotIn(case.outer_fold,seen)
      seen.add(case.outer_fold)
      self.assertEqual(tuple(case.p_train.columns),PREDICTORS)
      self.assertFalse(case.training_cells & case.validation_cells)
      self.assertFalse(case.training_cells & case.outer_cells)
      self.assertFalse(case.validation_cells & case.outer_cells)
     self.assertEqual(seen,set(OUTER_FOLDS))

 def test_observation_effort_independent_from_species_oracle(self):
  X,truth,q,proxy,blocks=surface("ecotone")
  self.assertEqual(len(truth),len(q))
  self.assertTrue(np.all(truth>0))
  self.assertGreater(float(np.std(q)),.03)
  self.assertGreater(float(np.std(proxy)),.03)
  biased=next(cases("ecotone","biased",11711))
  plain=next(cases("ecotone","uniform",11711))
  self.assertGreater(float(np.std(biased.q_b_train)),.03)
  self.assertGreater(float(np.std(biased.proxy_b_train)),.02)
  self.assertTrue(np.all(plain.q_b_train==1))
  self.assertTrue(np.all(plain.proxy_b_train==1))
  for frame in (biased.p_train,biased.b_train,
                biased.p_val,biased.b_val):
   self.assertNotIn("oracle_true_habitat",frame.columns)
   self.assertNotIn("detection_effort",frame.columns)
   self.assertNotIn("occupancy",frame.columns)
  self.assertFalse(biased.p_train.equals(plain.p_train))

 def test_no_oracle_wiring_in_native_fit_process(self):
  worker=(ROOT/"benchmark"/"po_effort_models.R").read_text()
  self.assertNotIn("true_b_test",worker)
  self.assertNotIn("oracle_spearman",worker)
  self.assertNotIn("oracle_ipp",worker)
  self.assertNotIn("n_survey_true_absent",worker)
  self.assertIn("nf_po_effort_forest",worker)

if __name__=="__main__":
 unittest.main()
