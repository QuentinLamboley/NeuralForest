"""Design-level no-leakage tests for virtual-species presence-only CV."""
import sys
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"benchmark"))
from po_spatial_design import (
    SCENARIOS,BACKGROUND_POLICIES,EFFORT_CONDITIONS,
    PREDICTORS,BLOCKS,landscape,spatial_cases,validate_case)

class SpatialPresenceOnlyProtocol(unittest.TestCase):
    def test_all_spatial_groups_are_disjoint(self):
        for scenario in SCENARIOS:
          for effort in EFFORT_CONDITIONS:
            for background in BACKGROUND_POLICIES:
              seen=set()
              for case in spatial_cases(scenario,effort,background,8101):
                self.assertTrue(validate_case(case))
                self.assertFalse(case.all_train_cells & case.all_test_cells)
                self.assertGreaterEqual(len(case.presence_test),10)
                self.assertEqual(len(case.background_test),980)
                self.assertEqual(case.fold in seen,False)
                seen.add(case.fold)
              self.assertEqual(seen,set(range(BLOCKS)))

    def test_training_records_do_not_contain_simulator_truth(self):
        case=next(spatial_cases("nonlinear_interaction",
                   "effort_biased","effort_matched",8231))
        self.assertEqual(tuple(case.presence_train.columns),PREDICTORS)
        self.assertEqual(tuple(case.background_train.columns),PREDICTORS)
        for key in ("oracle_true_intensity","true_absence",
                    "detection_effort","occupied","truth"):
          self.assertNotIn(key,case.presence_train.columns)
          self.assertNotIn(key,case.background_train.columns)
        self.assertGreater(case.background_overlap_with_presence,0)

    def test_detection_effort_and_background_policies_change_sampling(self):
        c1=next(spatial_cases("linear","effort_biased",
                             "uniform",8101))
        c2=next(spatial_cases("linear","effort_biased",
                             "effort_matched",8101))
        # Presences do not change when background policy changes.
        self.assertTrue(c1.presence_train.equals(c2.presence_train))
        self.assertFalse(c1.background_train.equals(c2.background_train))
        features,suit,eff,blocks=landscape("linear")
        self.assertEqual(len(suit),len(features))
        self.assertTrue(np.isfinite(suit).all() and (suit>0).all())
        self.assertGreater(np.std(eff),0)

    def test_invalid_designs_fail(self):
        with self.assertRaises(ValueError):
          list(spatial_cases("linear","wrong","uniform",1))
        with self.assertRaises(ValueError):
          list(spatial_cases("linear","unbiased","wrong",1))
        with self.assertRaises(ValueError):
          list(spatial_cases("unknown","unbiased","uniform",1))

if __name__=="__main__":
    unittest.main()
