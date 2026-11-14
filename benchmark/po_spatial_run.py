"""Spatial presence-only neural forest vs MaxEnt, RF, GAM, GLM and boosting.

Publication-grade design PRINCIPLES, but this first simulation suite is
only an engineering benchmark, not biological external validation:
- Presence vs environmental availability (NOT absence)
- Spatial-block held-out prediction and strict train/test grid disjointness
- Uniform and target-group/effort-matched background sampling
- True latent suitability and detection-effort surfaces evaluated ONLY
  by the simulator after all predictions are frozen
- Matched seed/fold/background, paired metrics, fit/inference wall time
- Both continuous Boyce and background-AUC labelled correctly
- No testing outcome, true latent suit/effort maps used in training
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from po_spatial_design import (
 SCENARIOS,EFFORT_CONDITIONS,BACKGROUND_POLICIES,
 BLOCKS,SEEDS,spatial_cases,validate_case)

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"benchmark"/"po_spatial_out"
EXPECTED_R=("NeuralForest_PO_neural","NeuralForest_PO_raw",
    "NeuralForest_PO_random","NeuralForest_PO_hidden",
    "maxnet_lq","maxnet_lqph","ranger_RF",
    "GLM_logistic","GAM_smooth")
SKLEARN_METHODS=("ExtraTrees","RandomForest",
                 "BalancedSubsample_RF","XGBoost","CatBoost")
ENSEMBLE="Ensemble_RF_XGBoost_CatBoost"
EPS=1e-12

def metrics(case,p,b):
    p=np.asarray(p,float);b=np.asarray(b,float)
    if len(p)!=len(case.presence_test) or len(b)!=len(case.background_test):
        raise ValueError("Spatial prediction length mismatch")
    if np.any(~np.isfinite(p)) or np.any(~np.isfinite(b)) or (
        np.any(p<=0) or np.any(b<=0)):
        raise ValueError("Nonfinite or nonpositive relative intensity")
    auc=float(roc_auc_score(np.r_[np.ones(len(p)),np.zeros(len(b))],
                            np.r_[p,b]))
    # Simple finite-bin Boyce index. Avoid presenting tiny-N NA as zero.
    both=np.r_[p,b]
    rng=both.max()-both.min()
    boyce=float("nan")
    if rng>EPS:
        bins=np.linspace(float(both.min()),float(both.max()),13)
        hp=np.histogram(p,bins)[0]
        hb=np.histogram(b,bins)[0]
        valid=(hp>0)&(hb>0)
        if np.count_nonzero(valid)>=3:
            freq=(hp/hp.sum())/(np.maximum(hb/hb.sum(),1e-12))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rr=spearmanr((bins[:-1]+bins[1:])[valid],
                             freq[valid]).statistic
            if np.isfinite(rr):boyce=float(rr)
    truth=case.oracle_true_intensity
    oracle_corr=float(spearmanr(b,truth).statistic)
    effort_corr=float(spearmanr(b,case.oracle_effort).statistic)
    # Oracle continuous IPP log skill versus a spatially uniform null.
    # Can be computed only in a synthetic experiment with KNOWN truth.
    logp=np.log(b)
    z=np.max(logp)+np.log(np.exp(logp-np.max(logp)).mean())
    oracle_log_skill=float(np.dot(truth/truth.sum(),logp-z))
    return dict(auc_pb=auc,boyce=boyce,
           oracle_spearman=oracle_corr,
           oracle_log_skill=oracle_log_skill,
           effort_spearman=effort_corr,
           mean_presence_score=float(p.mean()),
           mean_available_score=float(b.mean()))

def case_models(case,seed):
    validate_case(case)
    records=[]
    names=set()
    def put(name,p,b,fitsec,predsec,implementation):
        if name in names:raise ValueError("Duplicate estimator")
        names.add(name)
        m=metrics(case,p,b)
        records.append(dict(
            scenario=case.scenario,effort_condition=case.effort_condition,
            background_policy=case.background_policy,
            seed=case.seed,fold=case.fold,
            model=name,implementation=implementation,
            n_train_presence=len(case.presence_train),
            n_train_background=len(case.background_train),
            n_test_presence=len(case.presence_test),
            n_test_background=len(case.background_test),
            original_presence_background_overlap=case.background_overlap_with_presence,
            train_seconds=float(fitsec),predict_seconds=float(predsec),
            total_seconds=float(fitsec+predsec),**m))
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        data=(case.presence_train,case.background_train,
              case.presence_test,case.background_test)
        for f,dataframe in zip(("p.csv","bg.csv","test.csv","grid.csv"),data):
            dataframe.to_csv(root/f,index=False,float_format="%.16g")
        subprocess.run(["Rscript",str(ROOT/"benchmark"/"po_spatial_models.R"),
            *[str(root/f) for f in ("p.csv","bg.csv","test.csv","grid.csv")],
            str(root/"out.json"),str(seed)],cwd=ROOT,check=True)
        models=json.loads((root/"out.json").read_text(encoding="utf-8"))["models"]
        if {m["name"] for m in models}!=set(EXPECTED_R):
            raise RuntimeError("MaxEnt/native ecological comparisons missing")
        for m in models:
            put(m["name"],m["presence_prediction"],m["background_prediction"],
                m["training_seconds"],m["prediction_seconds"],"R-native-or-CRAN")
    import sklearn
    X=pd.concat([case.presence_train,case.background_train],
                ignore_index=True)
    y=np.r_[np.ones(len(case.presence_train),dtype=int),
            np.zeros(len(case.background_train),dtype=int)]
    P=case.presence_test;B=case.background_test
    fits=[
      ("RandomForest",RandomForestClassifier(
         n_estimators=200,max_depth=7,min_samples_leaf=4,
         max_features="sqrt",random_state=seed,n_jobs=1)),
      ("BalancedSubsample_RF",RandomForestClassifier(
         n_estimators=200,max_depth=7,min_samples_leaf=4,
         class_weight="balanced_subsample",
         max_features="sqrt",random_state=seed,n_jobs=1)),
      ("ExtraTrees",ExtraTreesClassifier(
         n_estimators=200,max_depth=7,min_samples_leaf=4,
         max_features="sqrt",random_state=seed,n_jobs=1)),
      ("XGBoost",XGBClassifier(n_estimators=180,max_depth=4,
         learning_rate=.06,subsample=.85,colsample_bytree=.85,
         tree_method="hist",eval_metric="logloss",
         random_state=seed,n_jobs=1)),
      ("CatBoost",CatBoostClassifier(iterations=180,depth=5,
         learning_rate=.06,verbose=False,thread_count=1,
         random_seed=seed))]
    memo={}
    for name,model in fits:
        start=time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X,y)
        fitsec=time.perf_counter()-start
        start=time.perf_counter()
        pr=model.predict_proba(P)[:,1]
        br=model.predict_proba(B)[:,1]
        predsec=time.perf_counter()-start
        memo[name]=(pr,br,fitsec,predsec)
        # P/B odds are a relative intensity proxy, not occupancy.
        pp=np.clip(pr,1e-8,1-1e-8)
        bb=np.clip(br,1e-8,1-1e-8)
        put(name,pp/(1-pp),bb/(1-bb),fitsec,predsec,
            "sklearn-or-xgboost-or-catboost")
    # Fixed equal-weight committee; training cost is the sum of three
    # member fits, rather than pretending a free new algorithm exists.
    selected=("RandomForest","XGBoost","CatBoost")
    pr=np.mean([memo[name][0] for name in selected],axis=0)
    br=np.mean([memo[name][1] for name in selected],axis=0)
    pp=np.clip(pr,1e-8,1-1e-8);bb=np.clip(br,1e-8,1-1e-8)
    put(ENSEMBLE,pp/(1-pp),bb/(1-bb),
        sum(memo[name][2] for name in selected),
        sum(memo[name][3] for name in selected),
        "fixed-shared-component-ensemble")
    expected=set(EXPECTED_R)|set(SKLEARN_METHODS)|{ENSEMBLE}
    if names!=expected:raise RuntimeError("Benchmark methods incomplete")
    print(f"{case.scenario}/{case.effort_condition}/"
          f"{case.background_policy}/seed{case.seed}/fold{case.fold}: "
          f"{len(records)} models, "
          f"PO hidden oracle={next(r['oracle_spearman'] for r in records if r['model']=='NeuralForest_PO_hidden'):.3f}, "
          f"maxnet lqph oracle={next(r['oracle_spearman'] for r in records if r['model']=='maxnet_lqph'):.3f}",flush=True)
    return records

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--scenario",choices=SCENARIOS,required=True)
    parser.add_argument("--seeds",default="8101")
    parser.add_argument("--folds",default="0,2,4")
    args=parser.parse_args()
    seeds=tuple(map(int,args.seeds.split(",")))
    folds=set(map(int,args.folds.split(",")))
    if not folds.issubset(set(range(BLOCKS))) or len(folds)==0:
        raise ValueError("Invalid held-out spatial fold set")
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in seeds:
      for effort in EFFORT_CONDITIONS:
        for background in BACKGROUND_POLICIES:
          for case in spatial_cases(args.scenario,effort,background,seed):
            if case.fold in folds:
                rows.extend(case_models(case,seed+case.fold*101))
    df=pd.DataFrame(rows)
    expected=len(seeds)*len(EFFORT_CONDITIONS)*len(BACKGROUND_POLICIES)*len(folds)*15
    if len(df)!=expected or df[[
         "scenario","effort_condition","background_policy",
         "seed","fold","model"]].duplicated().any():
        raise RuntimeError(f"Missing prediction scores: {len(df)}/{expected}")
    (OUT/f"{args.scenario}.csv").write_text(df.to_csv(index=False),
                                            encoding="utf-8")
    meta=dict(scenario=args.scenario,seeds=seeds,folds=sorted(folds),
        records=len(df),independent_model_fits=len(df)-len(seeds)*
        len(EFFORT_CONDITIONS)*len(BACKGROUND_POLICIES)*len(folds),
        expected_methods=15,
        background_is_availability=True,
        true_absences_supplied_to_models=False,
        spatial_holdouts_independent=True,
        oracle_suitability_only_used_for_postfit_evaluation=True,
        pilot_not_confirmatory=True,github_sha=os.getenv("GITHUB_SHA","local"))
    (OUT/f"{args.scenario}.json").write_text(json.dumps(meta,indent=2)+"\n",
                                              encoding="utf-8")
    print(f"Complete: {len(df)} per-case observations, {args.scenario}")

if __name__=="__main__":
    main()
