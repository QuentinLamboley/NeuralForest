"""Contract tests for test-only PU attribution report."""
import unittest
import numpy as np
import pandas as pd
from audit.pu_attribution import audit_frame

MODELS=("NeuralForest_PUNeural","NeuralForest_PUStable",
        "NeuralForest_PUStableTeacher","NeuralForest_PUSafe")

def miniature(violate=False):
    rows=[]
    for seed in (911,947):
        for name in MODELS:
            offset=(0.012 if name=="NeuralForest_PUStable" else 0.0)
            if violate and seed==947 and name=="NeuralForest_PUSafe":
                offset=0.02
            metrics={"roc_auc":.72+offset,"pr_auc":.65+offset,
                     "boyce":.5+offset,"boyce_rank":.5+offset,
                     "brier":.15-offset/3,"logloss":.4-offset/3}
            data={"dataset":"Example","regime":"SCAR","seed":seed,
                  "method":name,"total_seconds":2.1}
            data.update(metrics)
            data.update({"external_"+k:v-.01 for k,v in metrics.items()})
            rows.append(data)
    return pd.DataFrame(rows)

class AttributionProtocol(unittest.TestCase):
    def test_matched_control_and_rank_guard(self):
        pairs,reference=audit_frame(miniature(),"real")
        self.assertFalse(pairs.empty)
        self.assertFalse(reference.empty)
        comparison=pairs[(pairs.comparison=="incremental_tree_effect")&
                         (pairs.metric=="roc_auc")&(pairs.cohort=="test")]
        self.assertEqual(len(comparison),1)
        self.assertAlmostEqual(comparison.iloc[0].mean_gain,.012)
        self.assertEqual(int(comparison.iloc[0].gains),2)

    def test_rank_inversion_is_release_failure(self):
        with self.assertRaisesRegex(AssertionError,"rank preservation"):
            audit_frame(miniature(violate=True),"real")

    def test_duplicate_model_seed_is_invalid(self):
        frame=miniature()
        bad=pd.concat([frame,frame.iloc[[0]]],ignore_index=True)
        with self.assertRaisesRegex(ValueError,"Duplicated"):
            audit_frame(bad,"real")

    def test_undefined_boyce_is_reported_without_masking_auc(self):
        frame=miniature()
        frame["boyce"]=np.nan
        frame["external_boyce"]=np.nan
        pairs,_=audit_frame(frame,"real")
        boyce=pairs[(pairs.metric=="boyce")&
                    (pairs.cohort=="test")]
        self.assertTrue((boyce.n==0).all())
        self.assertTrue((boyce.n_attempted==2).all())

if __name__=="__main__":
    unittest.main()
