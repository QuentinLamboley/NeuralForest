"""Prespecified, test-only PU residual-tree pilot.

One public-data diabetes progression source and one digit-eight task,
both previously inspected, are used for *engineering diagnostics only*.
No externally independent center is claimed. New random seeds are used
but neither task is an untouched confirmatory scientific hypothesis.

Each PUResidual prediction is paired with its EXACT trained neural
teacher; time spent fitting this shared teacher is not counted twice.
"""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes,load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"audit"/"pu_residual_pilot_results"
SEEDS=(2717,2791)
SOURCES=("Diabetes_Progression180","Digits_Eight")
REGIMES=("SCAR","SAR")

def source(name):
    if name=="Diabetes_Progression180":
        r=load_diabetes()
        X=r.data
        y=(r.target>=180).astype(int)
    elif name=="Digits_Eight":
        r=load_digits()
        X=r.data
        y=(r.target==8).astype(int)
    else: raise ValueError(name)
    return pd.DataFrame(X,columns=[f"x{i:02d}" for i in range(X.shape[1])]),y

def test_metrics(y,p):
    p=np.clip(np.asarray(p,dtype=float),1e-9,1-1e-9)
    y=np.asarray(y,dtype=int)
    bins=np.linspace(p.min(),p.max(),21)
    a=np.histogram(p,bins=bins)[0].astype(float)
    pos=np.histogram(p[y==1],bins=bins)[0].astype(float)
    valid=(a>0)&(pos>0)
    boyce=(float(spearmanr(((bins[1:]+bins[:-1])/2)[valid],
            ((pos/pos.sum())/(a/a.sum()))[valid]).statistic)
             if valid.sum()>=3 else float("nan"))
    return dict(auc=float(roc_auc_score(y,p)),
       ap=float(average_precision_score(y,p)),
       brier=float(brier_score_loss(y,p)),
       logloss=float(log_loss(y,p,labels=[0,1])),
       boyce=boyce)

def evaluate_case(name,regime,seed):
    X,y=source(name)
    tr,other=train_test_split(np.arange(len(y)),train_size=.6,
           stratify=y,random_state=seed)
    te,ext=train_test_split(other,test_size=.5,
           stratify=y[other],random_state=seed+17)
    xtr=X.iloc[tr].reset_index(drop=True)
    xte=X.iloc[te].reset_index(drop=True)
    xext=X.iloc[ext].reset_index(drop=True)
    yt=y[tr]
    rng=np.random.default_rng(seed+71)
    positives=np.flatnonzero(yt==1)
    n_known=max(20,int(round(.7*len(positives))))
    if n_known>=len(positives): raise RuntimeError("Not enough PU positives")
    weights=None
    if regime=="SAR":
        pos=xtr.iloc[positives].to_numpy(float)
        column=int(np.argmax(pos.var(axis=0)))
        z=pos[:,column]
        z=(z-np.median(z))/max(z.std(),1e-8)
        weights=np.exp(np.clip(2*z,-6,6))
        weights/=weights.sum()
    seen=rng.choice(positives,size=n_known,replace=False,p=weights)
    pu=np.zeros(len(xtr),dtype=int)
    pu[seen]=1
    records=[]
    def add(model,p1,p2,trainsec,predsec,selected="",nrounds=0):
        m=test_metrics(y[te],p1)
        ex=test_metrics(y[ext],p2)
        records.append(dict(source=name,regime=regime,seed=seed,
          model=model,train_seconds=float(trainsec),
          predict_seconds=float(predsec),
          total_seconds=float(trainsec+predsec),
          selected_candidate=selected,selected_rounds=nrounds,
          shared_fitted_teacher=(model=="NeuralForest_PUResidualTeacher"),
          **m,**{"external_"+k:v for k,v in ex.items()}))
    with tempfile.TemporaryDirectory() as tmp:
        w=Path(tmp)
        xtr.assign(Vampirisme=pu).to_csv(w/"train.csv",index=False)
        xte.assign(Vampirisme=y[te]).to_csv(w/"test.csv",index=False)
        xext.assign(Vampirisme=y[ext]).to_csv(w/"ext.csv",index=False)
        subprocess.run(["Rscript",str(ROOT/"benchmark"/"pu_residual_pilot_r.R"),
           str(w/"train.csv"),str(w/"test.csv"),str(w/"models.json"),
           "auc",str(seed),str(w/"ext.csv")],check=True,cwd=ROOT)
        objects=json.loads((w/"models.json").read_text(encoding="utf-8"))["models"]
        for r in objects:
            add(r["name"],r["probability"],r["external_probability"],
                r["train_seconds"],r["prediction_seconds"],
                r["selected_candidate"],int(r["selected_rounds"]))
    # These are supervised positive-vs-unlabeled CONTROL methods, not nnPU.
    baselines=[
      ("RandomForest",RandomForestClassifier(n_estimators=130,
          min_samples_leaf=3,max_features="sqrt",random_state=seed,n_jobs=1)),
      ("RF_Subsample50",BaggingClassifier(
          estimator=DecisionTreeClassifier(min_samples_leaf=3,
                  max_features="sqrt"),
          n_estimators=130,max_samples=.5,bootstrap=False,
          random_state=seed,n_jobs=1))]
    for label,model in baselines:
        t0=time.perf_counter()
        model.fit(xtr,pu)
        sec=time.perf_counter()-t0
        t0=time.perf_counter()
        p1=model.predict_proba(xte)[:,1]
        p2=model.predict_proba(xext)[:,1]
        predsec=time.perf_counter()-t0
        add(label,p1,p2,sec,predsec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from xgboost import XGBClassifier
        from catboost import CatBoostClassifier
        methods=[
          ("XGBoost",XGBClassifier(n_estimators=130,max_depth=3,
             learning_rate=.07,subsample=.85,colsample_bytree=.9,
             tree_method="hist",eval_metric="logloss",
             random_state=seed,n_jobs=1)),
          ("CatBoost",CatBoostClassifier(iterations=130,depth=4,
             learning_rate=.07,verbose=False,thread_count=1,
             random_seed=seed))]
        for label,model in methods:
            t0=time.perf_counter()
            model.fit(xtr,pu)
            sec=time.perf_counter()-t0
            t0=time.perf_counter()
            p1=model.predict_proba(xte)[:,1]
            p2=model.predict_proba(xext)[:,1]
            add(label,p1,p2,sec,time.perf_counter()-t0)
    if len(records)!=6:raise RuntimeError("Incomplete PU pilot comparison")
    print(f"{name}/{regime}/{seed}: "
        +", ".join(f"{r['model']} auc={r['auc']:.3f}"
                    for r in records),flush=True)
    return records

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name in SOURCES:
      for regime in REGIMES:
        for seed in SEEDS:
          rows.extend(evaluate_case(name,regime,seed))
    df=pd.DataFrame(rows)
    if len(df)!=len(SOURCES)*len(REGIMES)*len(SEEDS)*6:
        raise RuntimeError("Incomplete pilot")
    df.to_csv(OUT/"all_runs.csv",index=False)
    comparisons=[]
    for (source,regime,seed),g in df.groupby(["source","regime","seed"]):
        student=g[g.model=="NeuralForest_PUResidual"].iloc[0]
        teacher=g[g.model=="NeuralForest_PUResidualTeacher"].iloc[0]
        for metric in ("auc","ap","brier","logloss","boyce",
                       "external_auc","external_ap","external_brier",
                       "external_logloss","external_boyce"):
            sign=-1 if metric.endswith(("brier","logloss")) else 1
            comparisons.append(dict(source=source,regime=regime,seed=seed,
               metric=metric,positive_favors_residual=sign*(
                   student[metric]-teacher[metric]),
               selected=student.selected_candidate,
               rounds=student.selected_rounds,
               student_train_seconds=student.train_seconds,
               teacher_train_seconds=teacher.train_seconds))
    diffs=pd.DataFrame(comparisons)
    diffs.to_csv(OUT/"paired_effect.csv",index=False)
    lines=["# NNPU residual-tree pilot, only exploratory","",
       "The two public data tasks have been inspected in earlier work;",
       "fresh random seeds do NOT make them externally independent.",
       "SCAR or feature-dependent SAR masks true training positives.",
       "No test label was used to train, tune or select a model.",
       "The matched teacher shares the exact residual-model training.",
       "Its training time should not be double-counted.",
       "",
       "| Source | Regime | Residual AUC | Teacher AUC | Delta | Selected / runs |",
       "| --- | --- | ---: | ---: | ---: | --- |"]
    for (source,regime),group in df.groupby(["source","regime"]):
        r=group[group.model=="NeuralForest_PUResidual"]
        t=group[group.model=="NeuralForest_PUResidualTeacher"]
        ra=r.auc.mean();ta=t.auc.mean()
        chosen=", ".join(r.selected_candidate.astype(str))
        lines.append(f"| {source} | {regime} | {ra:.4f} | {ta:.4f} | "
                     f"{ra-ta:+.4f} | {chosen} |")
    lines+=["","## Baseline means and runtime","",
       "| Source | Regime | Method | AUC | AP | Shifted AUC | Total seconds |",
       "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for (source,regime,model),g in df.groupby(["source","regime","model"]):
        lines.append(f"| {source} | {regime} | {model} | {g.auc.mean():.4f} | "
                     f"{g.ap.mean():.4f} | {g.external_auc.mean():.4f} | "
                     f"{g.total_seconds.mean():.3f} |")
    lines+=["","## Caveats","",
      "The C++ gradient approximates nonnegative PU log-loss, assuming",
      "representative positives/background and a predeclared prior grid.",
      "SAR/SNAR and selected pseudo-absence populations violate this.",
      "Even when the PU-risk selection gate accepts trees, external",
      "discrimination and calibration may worsen. The pilot has only",
      "two seeds per scenario and is NOT an efficacy demonstration.",
      "Gradient-boosted trees are not necessarily superior to neural-only",
      "or conventional RF/boosting; the purpose is to find model failures.",
      "Full data and exact timing: all_runs.csv; paired_effect.csv.",
      ""]
    (OUT/"RESULTS.md").write_text("\n".join(lines),encoding="utf8")
    manifest=dict(sources=list(SOURCES),seeds=list(SEEDS),
          regimes=list(REGIMES),rows=len(df),trained_fits=len(df)-8,
          residual_models_are_shared_fit=True,
          independent_external_sites=False,
          previous_test_families_inspected=True,
          run_sha=os.getenv("GITHUB_SHA","local"))
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf8")

if __name__=="__main__":
    main()
