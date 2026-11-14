"""Exploratory nested spatial PO IPP-gradient forest benchmark.

Test objective-aligned native regression forests against an EXACT
matched neural IPP teacher, MaxEnt/maxnet LQPH, GAM, XGBoost, CatBoost.
Virtual species: true ecological intensity is known to the simulator
but neither training nor *inner* validation ever sees it.

Nested spatial design:
 - Outer block: fully untouched held-out region (both evaluation presence
   sites and environmental availability)
 - Inner block: a different geographic block within outer training only,
   used solely to choose the number of native IPP gradient stages
 - Following inner selection, the final model is refit on all outer
   training blocks; no outer-test labels/locations enter fitting.
 - Background locations may be occupied, NEVER true absences.
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
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from po_spatial_design import (
 SCENARIOS,EFFORT_CONDITIONS,BACKGROUND_POLICIES,
 BLOCKS,spatial_cases,validate_case)
from po_spatial_run import metrics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"benchmark"/"po_gradient_out"
NATIVE=(
  "NeuralForest_POGrad_raw","NeuralForest_POGrad_random",
  "NeuralForest_POGrad_hidden","NeuralForest_POGrad_teacher",
  "maxnet_lqph","GAM_smooth")
BASELINES=("RandomForest","XGBoost","CatBoost")
EXPECTED=set(NATIVE)|set(BASELINES)

def partition_inner(case):
    # Deterministic disjoint *geographic* inner block, with sample-size
    # checks. Read only spatial block membership and observation counts.
    available=sorted(set(case.presence_train_folds) &
                     set(case.background_train_folds))
    if case.fold in available:
        raise AssertionError("Outer test block appears in inner training")
    choices=[]
    for b in available:
        pv=case.presence_train_folds==b
        bv=case.background_train_folds==b
        if pv.sum()>=15 and bv.sum()>=20 and (
            (~pv).sum()>=50) and ((~bv).sum()>=80):
            choices.append((b,pv,bv))
    if not choices:
        raise RuntimeError("No viable spatially disjoint inner fold")
    b,pv,bv=choices[0]
    data=dict(
        ptrain=case.presence_train.loc[~pv].reset_index(drop=True),
        bgtrain=case.background_train.loc[~bv].reset_index(drop=True),
        pval=case.presence_train.loc[pv].reset_index(drop=True),
        bgval=case.background_train.loc[bv].reset_index(drop=True))
    return b,data

def case_models(case,seed):
    validate_case(case)
    block,partition=partition_inner(case)
    names=set()
    records=[]
    def save(name,p,b,fit,ptime,stages,kind):
        if name in names:raise AssertionError(f"Duplicate: {name}")
        names.add(name)
        score=metrics(case,p,b)
        records.append(dict(
            scenario=case.scenario,effort_condition=case.effort_condition,
            background_policy=case.background_policy,
            seed=case.seed,outer_block=case.fold,inner_block=block,
            model=name,implementation=kind,
            selected_stages=int(stages) if stages is not None else None,
            n_outer_train_presences=len(case.presence_train),
            n_outer_train_background=len(case.background_train),
            n_inner_fit_presences=len(partition["ptrain"]),
            n_inner_fit_background=len(partition["bgtrain"]),
            n_inner_validation_presences=len(partition["pval"]),
            n_inner_validation_background=len(partition["bgval"]),
            train_seconds=float(fit),predict_seconds=float(ptime),
            total_seconds=float(fit+ptime),**score))
    with tempfile.TemporaryDirectory() as tempdir:
        root=Path(tempdir)
        inputs=(("ptrain.csv",partition["ptrain"]),
                ("bgtrain.csv",partition["bgtrain"]),
                ("pval.csv",partition["pval"]),("bgval.csv",partition["bgval"]),
                ("ptest.csv",case.presence_test),
                ("grid.csv",case.background_test))
        for filename,df in inputs:
            df.to_csv(root/filename,index=False,float_format="%.16g")
        output=root/"output.json"
        subprocess.run([
           "Rscript",str(ROOT/"benchmark"/"po_gradient_spatial_models.R"),
           *(str(root/file) for file,_ in inputs),
           str(output),str(seed)],cwd=ROOT,check=True)
        fitted=json.loads(output.read_text(encoding="utf-8"))["models"]
        if {m["name"] for m in fitted}!=set(NATIVE):
            raise RuntimeError("Incomplete point-process/model reference suite")
        for r in fitted:
            if r["name"].startswith("NeuralForest_POGrad_") and (
                not r["used_spatial_inner_validation"]):
                raise RuntimeError("Native stage selection used nonspatial holdout")
            save(r["name"],r["presence_prediction"],
                 r["background_prediction"],r["training_seconds"],
                 r["prediction_seconds"],r.get("selected_stages"),
                 "native_Cpp17_R_or_maxnet_GAM")
    X=pd.concat([case.presence_train,case.background_train],
                ignore_index=True)
    # P versus background is a sampling indicator, NOT true absence.
    y=np.r_[np.ones(len(case.presence_train),dtype=int),
            np.zeros(len(case.background_train),dtype=int)]
    P=case.presence_test
    B=case.background_test
    baselines=[
        ("RandomForest",RandomForestClassifier(
          n_estimators=200,max_depth=7,min_samples_leaf=4,
          max_features="sqrt",n_jobs=1,random_state=seed)),
        ("XGBoost",XGBClassifier(n_estimators=180,max_depth=4,
          learning_rate=.06,subsample=.85,colsample_bytree=.85,
          tree_method="hist",eval_metric="logloss",
          random_state=seed,n_jobs=1)),
        ("CatBoost",CatBoostClassifier(iterations=180,depth=5,
          learning_rate=.06,verbose=False,thread_count=1,
          random_seed=seed))]
    for name,model in baselines:
        t0=time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X,y)
        fitsec=time.perf_counter()-t0
        t0=time.perf_counter()
        pr=np.clip(model.predict_proba(P)[:,1],1e-8,1-1e-8)
        br=np.clip(model.predict_proba(B)[:,1],1e-8,1-1e-8)
        ptime=time.perf_counter()-t0
        save(name,pr/(1-pr),br/(1-br),fitsec,ptime,
             None,"presence_background_binary_reference")
    if names!=EXPECTED:
        raise RuntimeError("Missing spatial IPP-gradient estimator output")
    by={d["model"]:d for d in records}
    print(f"{case.scenario} effort={case.effort_condition} "
          f"background={case.background_policy} "
          f"seed={case.seed} outer={case.fold} inner={block}: "
          f"learned={by['NeuralForest_POGrad_hidden']['oracle_spearman']:.3f},"
          f"teacher={by['NeuralForest_POGrad_teacher']['oracle_spearman']:.3f},"
          f"maxnet={by['maxnet_lqph']['oracle_spearman']:.3f},"
          f"catboost={by['CatBoost']['oracle_spearman']:.3f},"
          f"stages={by['NeuralForest_POGrad_hidden']['selected_stages']}",
          flush=True)
    return records

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scenario",required=True,choices=SCENARIOS)
    ap.add_argument("--seeds",default="9203,9337")
    ap.add_argument("--folds",default="0,2,4")
    args=ap.parse_args()
    folds=set(int(x) for x in args.folds.split(","))
    seeds=tuple(int(x) for x in args.seeds.split(","))
    if not folds or not folds.issubset(set(range(BLOCKS))):
        raise ValueError("Invalid spatial fold specification")
    if not seeds:raise ValueError("No replication seeds")
    OUT.mkdir(parents=True,exist_ok=True)
    records=[]
    for seed in seeds:
        for effort in EFFORT_CONDITIONS:
            for policy in BACKGROUND_POLICIES:
                for case in spatial_cases(args.scenario,effort,policy,seed):
                    if case.fold in folds:
                        records.extend(case_models(
                            case,seed+case.fold*101))
    expected=len(seeds)*len(EFFORT_CONDITIONS)*len(
           BACKGROUND_POLICIES)*len(folds)*len(EXPECTED)
    if len(records)!=expected:raise RuntimeError(
      f"Incomplete gradient spatial benchmark: {len(records)}/{expected}")
    df=pd.DataFrame.from_records(records)
    if df[["scenario","effort_condition","background_policy",
           "seed","outer_block","model"]].duplicated().any():
        raise RuntimeError("Duplicate outer-fold prediction")
    df.to_csv(OUT/f"{args.scenario}.csv",index=False)
    manifest=dict(scenario=args.scenario,seeds=seeds,
       folds=sorted(folds),methods=sorted(EXPECTED),records=len(df),
       independent_fits=len(df)-len(seeds)*len(EFFORT_CONDITIONS)*
       len(BACKGROUND_POLICIES)*len(folds),
       inner_holdout_spatial=True,
       refit_fixed_stages_on_full_outer_training=True,
       background_not_verified_absence=True,
       oracle_only_used_after_all_predictions=True,
       confirmatory_external_ecology=False,
       head_sha=os.environ.get("GITHUB_SHA","local"))
    (OUT/f"{args.scenario}.json").write_text(
        json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(f"Completed {len(df)} point-process gradient predictions",flush=True)

if __name__=="__main__":
    main()
