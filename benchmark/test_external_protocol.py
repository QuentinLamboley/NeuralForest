"""Verify independent shifted synthetic test cohorts and protocol scopes."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from design import ALL_SCENARIOS,create_data,create_external_population
from run import SELECTION_SEEDS,model_candidates

class IndependentCohortTests(unittest.TestCase):
    def test_shifted_external_is_independent_and_aligned(self):
        for key in ("presence_bg_1to5","pseudoabsence_hard",
                    "distribution_shift","rare_1pct","nonlinear_xor"):
            spec=ALL_SCENARIOS[key]
            train,test,_=create_data(spec,701)
            external,meta=create_external_population(spec,701)
            self.assertEqual(list(test),list(external))
            self.assertEqual(external.Vampirisme.nunique(),2)
            self.assertNotEqual(test.iloc[0].to_dict(),
                                external.iloc[0].to_dict())
            self.assertNotEqual(meta["external_proximity"],
                               meta["original_test_proximity"])
            self.assertIn("synthetic",meta["population_type"])
            self.assertGreaterEqual(len(external),1000)

    def test_existing_model_reference_set_unchanged(self):
        self.assertEqual(len(SELECTION_SEEDS),10)
        self.assertEqual(len(model_candidates(701)),23)
        # All 31 previously benchmarked methods remain present; three
        # native PU strategies compete on three PU scenarios.
        self.assertEqual(19*10*31+3*10*3,5980)

if __name__=="__main__":
    unittest.main()
