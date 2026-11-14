"""Independent survey score and no-label-to-fitting regression checks."""
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"benchmark"))
from po_nceas_multi_run import REGIONS,METHODS,external_scores
from po_nceas_multi_summarize import CONTRASTS,PRIMARY

class IndependentSurveyProtocol(unittest.TestCase):
    def test_prespecified_six_regions_and_full_competitors(self):
        self.assertEqual(set(REGIONS),{"AWT","CAN","NSW","NZ","SA","SWI"})
        self.assertEqual(len(METHODS),20)
        for model in ("NeuralForest_POGrad_hidden",
                      "NeuralForest_POGrad_random",
                      "NeuralForest_POGrad_raw",
                      "NeuralForest_POGrad_teacher",
                      "NeuralForest_PO_hidden","maxnet_lqph",
                      "CatBoost","CatBoost_weighted","XGBoost",
                      "XGBoost_weighted"):
            self.assertIn(model,METHODS)
        self.assertIn("NeuralForest_POGrad_random",CONTRASTS)

    def test_true_survey_auc_and_ap(self):
        result=external_scores([0]*15+[1]*15,
                     list(range(1,16))+list(range(16,31)))
        self.assertTrue(result["evaluable"])
        self.assertAlmostEqual(result["survey_auc"],1)
        self.assertAlmostEqual(result["survey_ap"],1)
        same=external_scores([0]*20+[1]*4,range(24))
        self.assertFalse(same["evaluable"])
        self.assertTrue(np.isnan(same["survey_auc"]))
        with self.assertRaises(ValueError):
            external_scores([0,1],[1])
        with self.assertRaises(ValueError):
            external_scores([0,2],[1,2])

    def test_model_process_cannot_open_survey_outcome_labels(self):
        train=(ROOT/"benchmark"/"po_nceas_multi_models.R").read_text()
        self.assertNotIn("survey_outcomes_AUDIT_ONLY.csv",train)
        self.assertNotIn("disPa(",train)
        self.assertNotIn("survey_auc",train)
        source=(ROOT/"benchmark"/"po_nceas_multi_run.py").read_text()
        marker=source.index("# THIS FILE IS NOT ACCESSED UNTIL")
        self.assertNotIn("survey_outcomes_AUDIT_ONLY.csv",source[:marker])
        self.assertIn("survey_outcomes_AUDIT_ONLY.csv",source[marker:])

if __name__=="__main__":
    unittest.main()
