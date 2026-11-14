"""Independent virtual-species observation-effort IPP benchmark.

Native teacher and native IPP-gradient corrections are matched
on training species presence/available-background and one training-
only inner geographic block. The scorer knows true latent habitat
intensity ONLY AFTER all predictors have been frozen.

Primary metric: Spearman correlation of predicted ECOLOGICAL habitat
relative intensity with known virtual species habitat intensity on
the disjoint held-out geographic region.
Secondary: simulator expected ecological IPP log score, observed
presence-vs-background AUC (confounded by effort), training/inference
cost and spatial transfer. Strong effort-aware CatBoost weights
provide a non-neural comparison to matched bias correction.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import subprocess
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

from po_effort_design import (
 LANDSCAPES,EFFORTS,SEEDS,OUTER_FOLDS,cases,validate_case)

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"benchmark"/"po_effort_out"
R_MODELS={
 "IPP_neural_naive","IPP_neural_true","IPP_neural_proxy",
 "IPP_gradient_naive_hidden","IPP_gradient_true_hidden",
 "IPP_gradient_true_raw","IPP_gradient_true_random",
 "IPP_gradient_proxy_hidden","GLM_naive","GLM_true",
 "MaxEnt_lqph"}
PY_MODELS={"CatBoost_naive","CatBoost_effort_weighted"}
ALL_MODELS=R_MODELS|PY_MODELS
assert len(ALL_MODELS)==13

def assess(c,pp,bb):
    pp=np.asarray(pp,dtype=float).ravel()
    bb=np.asarray(bb,dtype=float).ravel()
    if len(pp)!=len(c.p_test) or len(bb)!=len(c.b_test):
        raise ValueError("Incorrect virtual species outer cohort predictions")
    if np.any(~np.isfinite(pp)) or np.any(~np.isfinite(bb)) or (
        np.any(pp<=0) or np.any(bb<=0)):
        raise ValueError("Invalid positive relative point-process scores")
    truth=c.true_b_test
    rho=float(spearmanr(bb,truth).statistic)
    if not np.isfinite(rho):
        raise ValueError("Undefined oracle ecological correlation")
    # Expected ecological IPP log-score improvement vs constant
    # f(s)=0 on the outer accessible region.
    z=np.log(bb)
    risk=float(np.average(z,weights=truth)-
               (logsumexp(z)-np.log(len(z))))
    observed_auc=float(roc_auc_score(
        np.r_[np.ones(len(pp)),np.zeros(len(bb))],
        np.r_[pp,bb]))
    q=c.effort_b_test
    effort_rho=(float(spearmanr(bb,q).statistic)
      if np.ptp(q)>1e-9 else float("nan"))
    return dict(oracle_spearman=rho,oracle_ipp_logskill=risk,
       observation_biased_auc_pb=observed_auc,
       effort_spearman=effort_rho)

def run_case(c,seed):
    validate_case(c)
    names=set()
    rows=[]
    def push(name,pp,bb,fitsec,predsec,stages=None,assumption=None):
        if name in names:
            raise RuntimeError("Duplicate effort-challenge method")
        names.add(name)
        rows.append(dict(landscape=c.landscape,
          effort_condition=c.effort_condition,seed=c.seed,
          outer_fold=c.outer_fold,inner_fold=c.inner_fold,
          model=name,effort_assumption=assumption,
          selected_stages=stages,
          n_training_presence=len(c.p_train),
          n_training_background=len(c.b_train),
          n_inner_presence=len(c.p_val),
          n_inner_background=len(c.b_val),
          n_outer_presence=len(c.p_test),
          n_outer_grid=len(c.b_test),
          train_seconds=float(fitsec),
          predict_seconds=float(predsec),
          **assess(c,pp,bb)))
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)
        inputs=[
            ("p_train.csv",c.p_train),
            ("b_train.csv",c.b_train),
            ("p_inner.csv",c.p_val),
            ("b_inner.csv",c.b_val),
            ("p_test.csv",c.p_test),
            ("grid_test.csv",c.b_test)]
        effort=[
            ("effort_b_train.csv",c.q_b_train),
            ("effort_b_inner.csv",c.q_b_val),
            ("proxy_b_train.csv",c.proxy_b_train),
            ("proxy_b_inner.csv",c.proxy_b_val),
            ("effort_p_train.csv",c.q_p_train),
            ("effort_p_inner.csv",c.q_p_val)]
        for file,df in inputs:
            df.to_csv(path/file,index=False,float_format="%.16g")
        for file,data in effort:
            pd.DataFrame({"effort":data}).to_csv(
              path/file,index=False,float_format="%.16g")
        outfile=path/"pred.json"
        subprocess.run(["Rscript",str(ROOT/"benchmark"/"po_effort_models.R"),
            *(str(path/f) for f,_ in inputs),
            *(str(path/f) for f,_ in effort),
            str(outfile),str(seed)],check=True,cwd=ROOT)
        models=json.loads(outfile.read_text(encoding="utf-8"))["models"]
        if {m["model"] for m in models}!=R_MODELS:
            raise RuntimeError("Missing native/IP P ecologic baseline")
        for m in models:
            stages=m.get("selected_stages")
            if isinstance(stages,str) and stages in ("NA","NaN","null"):
                stages=None
            push(m["model"],m["presence"],m["background"],
                m["train_seconds"],m["predict_seconds"],
                stages,m.get("effort_assumption"))
    # A fixed non-neural C++ booster with/without an independently
    # measured positive/available background recording-intensity
    # correction. No tuning on TRUE oracle habitat or outer test.
    px=pd.concat([c.p_train,c.p_val],ignore_index=True)
    bx=pd.concat([c.b_train,c.b_val],ignore_index=True)
    X=pd.concat([px,bx],ignore_index=True)
    y=np.r_[np.ones(len(px),dtype=int),
            np.zeros(len(bx),dtype=int)]
    q=np.r_[c.q_b_train,c.q_b_val]
    q=q/max(1e-12,float(q.mean()))
    def catfit(name,sample_weight):
        model=CatBoostClassifier(iterations=180,depth=5,
              learning_rate=.06,verbose=False,thread_count=1,
              random_seed=seed)
        begin=time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X,y,sample_weight=sample_weight)
        seconds=time.perf_counter()-begin
        begin=time.perf_counter()
        pp=np.clip(model.predict_proba(c.p_test)[:,1],
                   1e-8,1-1e-8)
        bb=np.clip(model.predict_proba(c.b_test)[:,1],
                   1e-8,1-1e-8)
        psec=time.perf_counter()-begin
        push(name,pp/(1-pp),bb/(1-bb),seconds,psec,
             None,"independent_known_effort" if sample_weight is
             not None else "none")
    catfit("CatBoost_naive",None)
    catfit("CatBoost_effort_weighted",
           np.r_[np.ones(len(px)),q])
    if names!=ALL_MODELS:
        raise RuntimeError("Virtual effort suite incomplete")
    by={r["model"]:r for r in rows}
    print(f"{c.landscape}/{c.effort_condition}/s{c.seed}/outer{c.outer_fold}: "
      f"rho teacher naive {by['IPP_neural_naive']['oracle_spearman']:+.3f} "
      f"known {by['IPP_neural_true']['oracle_spearman']:+.3f}; "
      f"grad naive {by['IPP_gradient_naive_hidden']['oracle_spearman']:+.3f} "
      f"known {by['IPP_gradient_true_hidden']['oracle_spearman']:+.3f}, "
      f"CatBoost naive {by['CatBoost_naive']['oracle_spearman']:+.3f} "
      f"known {by['CatBoost_effort_weighted']['oracle_spearman']:+.3f}",
      flush=True)
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--landscape",required=True,choices=LANDSCAPES)
    ap.add_argument("--seeds",default="11711,11807,11909")
    ap.add_argument("--folds",default="1,2,3")
    args=ap.parse_args()
    seeds=tuple(map(int,args.seeds.split(",")))
    folds=set(map(int,args.folds.split(",")))
    if not folds.issubset(set(OUTER_FOLDS)):
        raise ValueError("Unexpected preregistered outer folds")
    OUT.mkdir(parents=True,exist_ok=True)
    results=[]
    for seed in seeds:
        for effort in EFFORTS:
            for c in cases(args.landscape,effort,seed):
                if c.outer_fold in folds:
                    results.extend(run_case(c,seed+c.outer_fold*201))
    df=pd.DataFrame.from_records(results)
    expected=len(seeds)*len(EFFORTS)*len(folds)*len(ALL_MODELS)
    if len(df)!=expected or df[
       ["landscape","effort_condition","seed","outer_fold","model"]
       ].duplicated().any():
        raise RuntimeError(f"Incomplete fixed effort challenge: "
                           f"{len(df)} != {expected}")
    df.to_csv(OUT/f"{args.landscape}.csv",index=False)
    manifest=dict(
      landscape=args.landscape,seeds=seeds,outer_folds=sorted(folds),
      model_names=sorted(ALL_MODELS),rows=len(df),
      independently_known_detection_effort=True,
      imperfect_target_group_effort_proxy_is_independent_of_focal_labels=True,
      latent_ecological_habitat_never_passed_to_training=True,
      disjoint_inner_and_outer_geographic_blocks=True,
      accessible_background_is_not_a_true_absence=True,
      observed_presence_background_auc_is_effort_confounding_not_occupancy=True,
      no_external_empirical_evidence=True,
      no_outcome_based_tuning=True,
      github_sha=os.getenv("GITHUB_SHA","local"))
    (OUT/f"{args.landscape}.json").write_text(
      json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(f"Completed {len(df)} effort-aware test predictions",flush=True)

if __name__=="__main__":main()
