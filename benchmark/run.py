"""Reproducible multi-scenario benchmark against independent test samples."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              VotingClassifier, StackingClassifier,
                              BaggingClassifier,
                              HistGradientBoostingClassifier,
                              GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, average_precision_score,
    roc_auc_score, log_loss, brier_score_loss, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC

from design import SCENARIOS, create_data, create_external_population
from balanced_prior import BalancedPriorCorrectedRF

ROOT=Path(__file__).resolve().parents[1]
SELECTION_SEEDS=(17,43,71,97,131,197,251,347,401,503)
METRICS=("roc_auc","pr_auc","accuracy","balanced_accuracy","precision",
         "recall","specificity","f1","mcc","brier","logloss","boyce",
         "boyce_rank","ece","precision_at_10pct","recall_at_10pct")
LOWER={"brier","logloss"}

def boyce_index(y,p,bins=20):
    y=np.asarray(y,dtype=int)
    p=np.asarray(p,dtype=float)
    if np.sum(y==1)<5 or len(np.unique(p))<3:
        return np.nan
    breaks=np.linspace(p.min(),p.max(),bins+1)
    a=np.histogram(p,bins=breaks)[0].astype(float)
    h=np.histogram(p[y==1],bins=breaks)[0].astype(float)
    ok=(a>0)&(h>0)
    if ok.sum()<3: return np.nan
    rate=(h/h.sum())/np.maximum(a/a.sum(),1e-12)
    return float(spearmanr((breaks[:-1]+breaks[1:])[ok]/2,rate[ok]).statistic)

def boyce_rank(y,p,bins=20):
    """Diagnostic Boyce on rank-transformed scores, monotone-invariant."""
    p=np.asarray(p,dtype=float)
    if len(p)<10: return np.nan
    ranks=(rankdata(p,method="average")-0.5)/len(p)
    return boyce_index(y,ranks,bins=bins)

def expected_calibration_error(y,p,bins=10):
    """Equal-width expected calibration error; descriptive, not a proper score."""
    y=np.asarray(y,dtype=int)
    p=np.asarray(p,dtype=float)
    edges=np.linspace(0.,1.,bins+1)
    index=np.clip(np.searchsorted(edges,p,side="right")-1,0,bins-1)
    error=0.
    for b in range(bins):
        mask=(index==b)
        if mask.any():
            error+=mask.mean()*abs(y[mask].mean()-p[mask].mean())
    return float(error)

def metric_table(y,p):
    y=np.asarray(y,dtype=int)
    p=np.clip(np.asarray(p,dtype=float),1e-8,1-1e-8)
    top=max(1,int(np.ceil(0.1*len(p))))
    leaders=np.argsort(-p,kind="mergesort")[:top]
    hits=int(y[leaders].sum())
    pred=(p>=0.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {
        "roc_auc":float(roc_auc_score(y,p)) if np.unique(y).size==2 else np.nan,
        "pr_auc":float(average_precision_score(y,p)),
        "accuracy":float(accuracy_score(y,pred)),
        "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
        "precision":float(precision_score(y,pred,zero_division=0)),
        "recall":float(recall_score(y,pred,zero_division=0)),
        "specificity":float(tn/max(1,tn+fp)),
        "f1":float(f1_score(y,pred,zero_division=0)),
        "mcc":float(matthews_corrcoef(y,pred)),
        "brier":float(brier_score_loss(y,p)),
        "logloss":float(log_loss(y,np.column_stack([1-p,p]),labels=[0,1])),
        "boyce":boyce_index(y,p),
        "boyce_rank":boyce_rank(y,p),
        "ece":expected_calibration_error(y,p),
        "precision_at_10pct":float(hits/top),
        "recall_at_10pct":float(hits/max(1,np.sum(y)))
    }

def objective(row, metric):
    k={"auc":"roc_auc","pr_auc":"pr_auc","boyce":"boyce",
       "brier":"brier","logloss":"logloss",
       "f1":"f1","balanced_accuracy":"balanced_accuracy"}[metric]
    v=row.get(k,float("nan"))
    return (float("inf") if k in LOWER else -float("inf")) if not np.isfinite(v) else v

def better(a,b,metric):
    return a<b if metric in LOWER else a>b

def model_candidates(seed):
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    from imblearn.ensemble import BalancedRandomForestClassifier
    return {
      "Dummy":[lambda:DummyClassifier(strategy="prior")],
      "Logistic":[lambda c=c:make_pipeline(StandardScaler(),LogisticRegression(
          C=c,max_iter=700,solver="lbfgs",random_state=seed)) for c in (0.2,2.0)],
      "RandomForest":[lambda leaf=leaf:RandomForestClassifier(
          n_estimators=160,min_samples_leaf=leaf,max_features="sqrt",
          random_state=seed,n_jobs=1) for leaf in (1,4)],
      # Bootstrap with partial-row sampling: a genuine smaller in-bag RF.
      "RF_Bootstrap50":[lambda leaf=leaf:RandomForestClassifier(
          n_estimators=160,max_samples=0.5,bootstrap=True,
          min_samples_leaf=leaf,max_features="sqrt",
          random_state=seed,n_jobs=1) for leaf in (1,4)],
      # Bagged randomized trees sampled without replacement, a controlled
      # classical forest subsampling baseline.
      "RF_Subsample50":[lambda leaf=leaf:BaggingClassifier(
          estimator=DecisionTreeClassifier(
              min_samples_leaf=leaf,max_features="sqrt"),
          n_estimators=160,max_samples=0.5,bootstrap=False,
          random_state=seed,n_jobs=1) for leaf in (1,4)],
      "RF_Subsample80":[lambda leaf=leaf:BaggingClassifier(
          estimator=DecisionTreeClassifier(
              min_samples_leaf=leaf,max_features="sqrt"),
          n_estimators=160,max_samples=0.8,bootstrap=False,
          random_state=seed,n_jobs=1) for leaf in (1,4)],
      # Under-sampling means an equal number of majority/minority examples
      # per tree, with no replacement. This changes probability calibration.
      "RF_BalancedUndersampled":[lambda leaf=leaf:BalancedRandomForestClassifier(
          n_estimators=160,bootstrap=False,replacement=False,
          sampling_strategy="all",min_samples_leaf=leaf,
          max_features="sqrt",random_state=seed,n_jobs=1)
          for leaf in (1,4)],
      "RF_BalancedUndersampledPrior":[lambda leaf=leaf:BalancedPriorCorrectedRF(
          n_estimators=160,min_samples_leaf=leaf,
          max_features="sqrt",random_state=seed,n_jobs=1)
          for leaf in (1,4)],
      "ExtraTrees":[lambda leaf=leaf:ExtraTreesClassifier(
          n_estimators=160,min_samples_leaf=leaf,max_features="sqrt",
          random_state=seed,n_jobs=1) for leaf in (1,4)],
      "GradientBoosting":[lambda depth=depth:GradientBoostingClassifier(
          n_estimators=120,max_depth=depth,learning_rate=0.07,
          random_state=seed) for depth in (2,3)],
      "HistGradientBoosting":[lambda leaf=leaf:HistGradientBoostingClassifier(
          max_iter=140,max_leaf_nodes=leaf,learning_rate=0.075,
          early_stopping=False,random_state=seed) for leaf in (15,31)],
      "AdaBoost":[lambda n=n:AdaBoostClassifier(
          n_estimators=n,learning_rate=0.7,random_state=seed) for n in (80,160)],
      "XGBoost":[lambda depth=depth:XGBClassifier(
          n_estimators=160,max_depth=depth,learning_rate=0.08,
          subsample=0.9,colsample_bytree=0.85,reg_lambda=2,
          tree_method="hist",eval_metric="logloss",
          random_state=seed,n_jobs=1) for depth in (3,5)],
      "LightGBM":[lambda leaves=leaves:LGBMClassifier(
          n_estimators=160,num_leaves=leaves,max_depth=-1,
          learning_rate=0.075,min_child_samples=15,
          verbose=-1,random_state=seed,n_jobs=1) for leaves in (15,31)],
      "CatBoost":[lambda depth=depth:CatBoostClassifier(
          iterations=170,depth=depth,learning_rate=0.07,
          l2_leaf_reg=3,verbose=False,thread_count=1,
          random_seed=seed) for depth in (4,6)],
      "SVM_RBF":[lambda C=C:make_pipeline(StandardScaler(),SVC(
          C=C,kernel="rbf",probability=True,random_state=seed))
          for C in (0.8,3.0)],
      "MLP":[lambda width=width:make_pipeline(StandardScaler(),MLPClassifier(
          hidden_layer_sizes=width,max_iter=180,
          early_stopping=True,n_iter_no_change=12,
          learning_rate_init=0.003,alpha=0.002,
          random_state=seed)) for width in ((32,),(64,32))],
      # The following are *ensembles of separately fitted learners*,
      # not re-labelled individual forests. All selection happens on
      # the inner holdout; stack training uses its own 3-fold OOF predictions.
      "Voting_RF_ExtraTrees":[lambda w=w:VotingClassifier(
          estimators=[
             ("rf",RandomForestClassifier(n_estimators=120,
                min_samples_leaf=2,max_features="sqrt",
                random_state=seed,n_jobs=1)),
             ("extra",ExtraTreesClassifier(n_estimators=120,
                min_samples_leaf=2,max_features="sqrt",
                random_state=seed+1,n_jobs=1))],
          voting="soft",weights=w,n_jobs=1) for w in ((1,1),(1,2))],
      "Voting_RF_CatBoost":[lambda w=w:VotingClassifier(
          estimators=[
             ("rf",RandomForestClassifier(n_estimators=120,
                min_samples_leaf=2,random_state=seed,n_jobs=1)),
             ("cat",CatBoostClassifier(iterations=140,depth=5,
                learning_rate=0.07,verbose=False,thread_count=1,
                random_seed=seed+2))],
          voting="soft",weights=w,n_jobs=1) for w in ((1,1),(1,2))],
      "Voting_Boosting":[lambda w=w:VotingClassifier(
          estimators=[
             ("cat",CatBoostClassifier(iterations=140,depth=5,
                learning_rate=0.07,verbose=False,thread_count=1,
                random_seed=seed+3)),
             ("xgb",XGBClassifier(n_estimators=140,max_depth=4,
                learning_rate=0.08,tree_method="hist",eval_metric="logloss",
                random_state=seed+4,n_jobs=1)),
             ("lgb",LGBMClassifier(n_estimators=140,num_leaves=20,
                learning_rate=0.07,min_child_samples=15,verbose=-1,
                random_state=seed+5,n_jobs=1))],
          voting="soft",weights=w,n_jobs=1)
          for w in ((1,1,1),(2,1,1))],
      "Voting_Heterogeneous":[lambda w=w:VotingClassifier(
          estimators=[
             ("linear",make_pipeline(StandardScaler(),LogisticRegression(
                C=1.0,max_iter=700,random_state=seed))),
             ("extra",ExtraTreesClassifier(n_estimators=120,
                max_features="sqrt",min_samples_leaf=2,
                random_state=seed+6,n_jobs=1)),
             ("cat",CatBoostClassifier(iterations=140,depth=5,
                learning_rate=0.07,verbose=False,thread_count=1,
                random_seed=seed+7))],
          voting="soft",weights=w,n_jobs=1)
          for w in ((1,1,1),(2,1,1))],
      "Stacking_RF_XGB":[lambda:StackingClassifier(
          estimators=[
             ("rf",make_pipeline(SimpleImputer(strategy="median"),
                RandomForestClassifier(n_estimators=120,
                max_features="sqrt",min_samples_leaf=2,
                random_state=seed+8,n_jobs=1))),
             ("xgb",make_pipeline(SimpleImputer(strategy="median"),
                XGBClassifier(n_estimators=130,max_depth=4,
                learning_rate=0.08,tree_method="hist",eval_metric="logloss",
                random_state=seed+9,n_jobs=1)))],
          final_estimator=LogisticRegression(C=1.0,max_iter=700),
          cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=seed+10),
          stack_method="predict_proba",n_jobs=1)],
      "Stacking_Boosting":[lambda:StackingClassifier(
          estimators=[
             ("cat",make_pipeline(SimpleImputer(strategy="median"),
                CatBoostClassifier(iterations=140,depth=5,
                learning_rate=0.07,verbose=False,thread_count=1,
                random_seed=seed+11))),
             ("lgb",make_pipeline(SimpleImputer(strategy="median"),
                LGBMClassifier(n_estimators=140,num_leaves=20,
                learning_rate=0.07,min_child_samples=15,verbose=-1,
                random_state=seed+12)))],
          final_estimator=LogisticRegression(C=1.0,max_iter=700),
          cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=seed+13),
          stack_method="predict_proba",n_jobs=1)],
    }

def estimator_with_preprocessing(estimator):
    # Stacking trains its meta-learner on 3-fold OOF predictions.
    # Each base model in a stack owns its imputer so its CV fold never
    # receives imputation medians fitted from its held-out fold.
    if isinstance(estimator,StackingClassifier):
        return estimator
    return make_pipeline(SimpleImputer(strategy="median"),estimator)

def evaluate_sklearn(candidates,Xtr,ytr,Xte,yte,Xexternal,selection_metric,seed):
    xa,xb,ya,yb=train_test_split(
        Xtr,ytr,test_size=0.2,stratify=ytr,random_state=seed+57)
    output=[]
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      for name,options in candidates.items():
        t0=time.perf_counter()
        best=None; best_s=float("inf") if selection_metric in LOWER else -float("inf")
        best_i=None
        for i,maker in enumerate(options):
            try:
                m=estimator_with_preprocessing(maker())
                m.fit(xa,ya)
                valp=m.predict_proba(xb)[:,list(m.classes_).index(1)]
                score=objective(metric_table(yb,valp),selection_metric)
                if best_i is None or better(score,best_s,selection_metric):
                    best_s=score;best_i=i
            except Exception as exc:
                if best_i is None and i==len(options)-1: raise
        if best_i is None: raise RuntimeError("No valid model parameters")
        selection_time=time.perf_counter()-t0
        t_final=time.perf_counter()
        model=estimator_with_preprocessing(options[best_i]())
        model.fit(Xtr,ytr)
        fit_time=time.perf_counter()-t_final
        train_time=selection_time+fit_time
        t1=time.perf_counter()
        p=model.predict_proba(Xte)[:,list(model.classes_).index(1)]
        pexternal=model.predict_proba(Xexternal)[:,list(model.classes_).index(1)]
        pred_time=time.perf_counter()-t1
        output.append((name,p,float(train_time),float(pred_time),
                       float(best_s),f"trial={best_i+1}/{len(options)}",
                       float(selection_time),float(fit_time),pexternal))
    return output

def evaluate_R(train,test,external,metric,seed,root,script="fit_r.R"):
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)
        train.to_csv(path/"train.csv",index=False)
        test.to_csv(path/"test.csv",index=False)
        external.to_csv(path/"external.csv",index=False)
        subprocess.run(["Rscript",str(root/"benchmark"/script),
                        str(path/"train.csv"),str(path/"test.csv"),
                        str(path/"output.json"),metric,str(seed),
                        str(path/"external.csv")],check=True,
                        cwd=root,stdout=subprocess.DEVNULL)
        obj=json.loads((path/"output.json").read_text(encoding="utf8"))
        return obj["models"]

def run_case(spec,seed,output):
    train,test,metadata=create_data(spec,seed)
    external,external_metadata=create_external_population(spec,seed)
    metadata["external_validation"]=external_metadata
    Xtr=train.drop(columns="Vampirisme")
    Xte=test.drop(columns="Vampirisme")
    Xext=external.drop(columns="Vampirisme")
    if list(Xext)!=list(Xte):
        raise RuntimeError("External cohort predictors are misaligned")
    ytr=train["Vampirisme"].to_numpy(dtype=int)
    yte=test["Vampirisme"].to_numpy(dtype=int)
    yext=external["Vampirisme"].to_numpy(dtype=int)
    dataset_id=f"{spec.key}_{seed}"
    rows=[]
    def add(name,p,trainsec,predsec,inner_score,config,
            selectionsec=np.nan,finalfitsec=np.nan,
            external_probability=None,status="ok",error=""):
        if status=="ok":
            if external_probability is None:
                raise RuntimeError(f"Missing external predictions for {name}")
            metrics=metric_table(yte,p)
            external_metrics=metric_table(yext,external_probability)
        else:
            metrics={m:np.nan for m in METRICS}
            external_metrics={m:np.nan for m in METRICS}
        rows.append(dict(
            scenario=spec.key,seed=seed,model=name,
            selected_metric=spec.target_metric,
            train_n=len(train),test_n=len(test),
            train_positives=int(ytr.sum()),test_positives=int(yte.sum()),
            proximity=spec.proximity,test_proximity=spec.test_proximity,
            prevalence=spec.prevalence,extension=spec.extension,
            target_semantics=metadata["label_semantics"],
            train_seconds=trainsec,predict_seconds=predsec,
            external_n=len(external),external_positives=int(yext.sum()),
            external_proximity=external_metadata["external_proximity"],
            selection_seconds=selectionsec,final_fit_seconds=finalfitsec,
            total_seconds=trainsec+predsec,
            validation_score=inner_score,configuration=config,
            status=status,error=error,**metrics,
            **{"external_"+m:v for m,v in external_metrics.items()}))
    try:
        for name,p,tr,pt,inner,config,select,final,pext in evaluate_sklearn(
                model_candidates(seed),Xtr,ytr,Xte,yte,Xext,spec.target_metric,seed):
            add(name,p,tr,pt,inner,config,select,final,pext)
    except Exception:
        # Try every baseline independently so one dependency/estimator failure
        # cannot mask the other algorithms.
        for name,makers in model_candidates(seed).items():
            try:
                for nm,p,tr,pt,inner,config,select,final,pext in evaluate_sklearn(
                        {name:makers},Xtr,ytr,Xte,yte,Xext,spec.target_metric,seed):
                    add(nm,p,tr,pt,inner,config,select,final,pext)
            except Exception as exc:
                add(name,None,np.nan,np.nan,np.nan,"",
                    status="error",error=f"{type(exc).__name__}: {exc}")
    try:
        for run in evaluate_R(train,test,external,spec.target_metric,seed,ROOT):
            p=np.asarray(run["probability"],dtype=float)
            if len(p)!=len(yte):
                raise ValueError("R output and test sample lengths differ")
            add(run["name"],p,float(run["train_seconds"]),
                float(run["prediction_seconds"]),
                float(run["validation_score"]),run.get("configuration",""),
                float(run.get("selection_seconds",np.nan)),
                float(run.get("final_fit_seconds",np.nan)),
                np.asarray(run["external_probability"],dtype=float))
    except Exception as exc:
        message=f"{type(exc).__name__}: {exc}"
        for model in ("ranger","NeuralForest_CART","NeuralForest_Differentiable",
                      "NeuralForest_Adaptive","NeuralForest_Calibrated",
                      "NeuralForest_SubSample50_Informed",
                      "NeuralForest_UndersampledPlain",
                      "NeuralForest_UndersampledInformed"):
            add(model,None,np.nan,np.nan,np.nan,"",status="error",error=message)
    if spec.extension in ("bg1","bg5","hard"):
        try:
            for run in evaluate_R(train,test,external,
                                  spec.target_metric,seed,ROOT,
                                  script="fit_pu_r.R"):
                p=np.asarray(run["probability"],dtype=float)
                pext=np.asarray(run["external_probability"],dtype=float)
                if len(p)!=len(yte) or len(pext)!=len(yext):
                    raise ValueError("PU outputs differ from test cohort sizes")
                add(run["name"],p,float(run["train_seconds"]),
                    float(run["prediction_seconds"]),
                    float(run["validation_score"]),run["configuration"],
                    float(run["selection_seconds"]),
                    float(run["final_fit_seconds"]),pext)
        except Exception as exc:
            for method in ("NeuralForest_PUNeural",
                           "NeuralForest_PUInformed",
                           "NeuralForest_PUStable",
                           "NeuralForest_PUSafe"):
                add(method,None,np.nan,np.nan,np.nan,"",
                    status="error",error=f"{type(exc).__name__}: {exc}")
    output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(output/f"{dataset_id}.csv",index=False)
    (output/f"{dataset_id}_design.json").write_text(
        json.dumps(metadata,indent=2,ensure_ascii=False),encoding="utf8")
    print(f"{dataset_id}: {sum(r['status']=='ok' for r in rows)}/{len(rows)} fitted")
    for r in rows:
        value=r["roc_auc"]
        print(f"  {r['model']:29s} AUC={value:.3f} AP={r['pr_auc']:.3f} "
              f"fit={r['train_seconds']:.2f}s test={r['predict_seconds']:.3f}s",
              flush=True)
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--group",choices=list(SCENARIOS),required=True)
    ap.add_argument("--out",type=Path,default=Path("benchmark/out"))
    ap.add_argument("--smoke",action="store_true")
    args=ap.parse_args()
    scenarios=SCENARIOS[args.group]
    if args.smoke:scenarios=scenarios[:1]
    seeds=SELECTION_SEEDS[:1] if args.smoke else SELECTION_SEEDS
    for scenario in scenarios:
      for seed in seeds:
        try:
            run_case(scenario,seed,args.out)
        except Exception as exc:
            print(f"FAILED {scenario.key}/{seed}: {exc}",file=sys.stderr)
            traceback.print_exc()
            raise

if __name__=="__main__":
    main()
