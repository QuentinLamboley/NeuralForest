"""Predeclared synthetic PU residual stress challenge.

Three generative mechanisms (nonlinear XOR interaction, ordinary linear
signal, no signal), five seeds each, SCAR/SAR confirmation, and two
independent true-label cohorts. The 'verified audit' is separate from
the independent test and transport samples. It may select between the
already-fitted residual predictor and its exact neural teacher but may
NEVER adjust hyperparameters. Explicitly not real clinical validation.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"audit"/"pu_residual_verified_results"
SEEDS=(3509,3517,3527,3539,3547)
SCENARIOS=("interaction_xor","linear","null")
REGIMES=("SCAR","SAR")
BASELINES=("RandomForest","RF_Subsample50","XGBoost","CatBoost")
PU_MODELS=("NeuralForest_PUResidual","NeuralForest_PUResidualTeacher",
           "NeuralForest_PUResidualVerified")

def cohort(seed,n,scenario,shift=False):
    rng=np.random.default_rng(seed)
    X=rng.normal(size=(n,5))
    if shift:
        X[:,3]=2.5-X[:,3]
        X[:,4]+=1.8
    if scenario=="interaction_xor":
        # Multi-variable XOR causes shallow additive teachers to struggle.
        signal=2.4*np.where((X[:,0]>0)==(X[:,1]>0),1.,-1.)
        logit=signal+.35*X[:,2]-1.1
        proba=expit(logit)
    elif scenario=="linear":
        proba=expit(1.3*X[:,0]+1.0*X[:,1]-1.65)
    elif scenario=="null":
        proba=np.full(n,.25)
    else:raise ValueError(scenario)
    y=rng.binomial(1,proba,size=n)
    data=pd.DataFrame(X,columns=[f"x{i:02d}" for i in range(X.shape[1])])
    return data,np.asarray(y,dtype=int)

def make_pu(x,y,regime,seed):
    pos=np.flatnonzero(y==1)
    n_known=int(round(.7*len(pos)))
    if n_known<30:raise RuntimeError("Not enough confirmed training positives")
    rng=np.random.default_rng(seed)
    p=None
    if regime=="SAR":
        z=x.iloc[pos,3].to_numpy()
        z=(z-np.median(z))/max(z.std(),1e-8)
        weights=np.exp(np.clip(2*z,-6,6))
        p=weights/weights.sum()
    elif regime!="SCAR":raise ValueError(regime)
    visible=rng.choice(pos,size=n_known,replace=False,p=p)
    out=np.zeros(len(y),dtype=int)
    out[visible]=1
    return out

def metric(y,p):
    y=np.asarray(y,dtype=int)
    p=np.clip(np.asarray(p,dtype=float),1e-9,1-1e-9)
    bins=np.linspace(p.min(),p.max(),21)
    boyce=float("nan")
    if np.ptp(p)>1e-12:
        observed=np.histogram(p[y==1],bins=bins)[0].astype(float)
        available=np.histogram(p,bins=bins)[0].astype(float)
        valid=(observed>0)&(available>0)
        if valid.sum()>=3:
            value=spearmanr(
                ((bins[:-1]+bins[1:])/2)[valid],
                ((observed/observed.sum())/
                (available/available.sum()))[valid]).statistic
            if np.isfinite(value):boyce=float(value)
    return dict(auc=float(roc_auc_score(y,p)),
        ap=float(average_precision_score(y,p)),
        brier=float(brier_score_loss(y,p)),
        logloss=float(log_loss(y,p,labels=[0,1])),
        boyce=boyce)

def one(scenario,regime,seed):
    train,truth=cohort(seed+10,820,scenario)
    audit,ya=cohort(seed+50,320,scenario)
    test,yt=cohort(seed+90,420,scenario)
    ext,ye=cohort(seed+130,420,scenario,shift=True)
    pu=make_pu(train,truth,regime,seed+301)
    rows=[]
    def record(name,p,external,trainsec,predsec,selected,rounds,gated):
        scores=metric(yt,p)
        shifted=metric(ye,external)
        rows.append(dict(scenario=scenario,regime=regime,seed=seed,
           model=name,selection=selected,residual_rounds=rounds,
           verified_choice=gated,
           training_seconds=float(trainsec),
           prediction_seconds=float(predsec),
           total_seconds=float(trainsec+predsec),
           **scores,**{"shift_"+m:v for m,v in shifted.items()}))
    with tempfile.TemporaryDirectory() as tmp:
        home=Path(tmp)
        train.assign(Vampirisme=pu).to_csv(home/"train.csv",index=False)
        audit.assign(Vampirisme=ya).to_csv(home/"audit.csv",index=False)
        test.assign(Vampirisme=yt).to_csv(home/"test.csv",index=False)
        ext.assign(Vampirisme=ye).to_csv(home/"shift.csv",index=False)
        subprocess.run(["Rscript",str(ROOT/"benchmark"/"pu_residual_verified_r.R"),
          str(home/"train.csv"),str(home/"audit.csv"),
          str(home/"test.csv"),str(home/"shift.csv"),
          str(home/"result.json"),str(seed)],check=True,cwd=ROOT)
        output=json.loads((home/"result.json").read_text(encoding="utf8"))["models"]
        for entry in output:
            record(entry["name"],entry["probability"],
                entry["external_probability"],entry["train_seconds"],
                entry["prediction_seconds"],entry["selected_candidate"],
                entry["correction_rounds"],entry["verified_selected"])
    baselines=[
        ("RandomForest",RandomForestClassifier(n_estimators=130,
           min_samples_leaf=3,max_features="sqrt",random_state=seed,n_jobs=1)),
        ("RF_Subsample50",BaggingClassifier(
           estimator=DecisionTreeClassifier(min_samples_leaf=3,max_features="sqrt"),
           n_estimators=130,max_samples=.5,bootstrap=False,
           random_state=seed,n_jobs=1))]
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    baselines += [
        ("XGBoost",XGBClassifier(n_estimators=130,max_depth=3,
            learning_rate=.07,subsample=.85,colsample_bytree=.9,
            tree_method="hist",eval_metric="logloss",
            random_state=seed,n_jobs=1)),
        ("CatBoost",CatBoostClassifier(iterations=130,depth=4,
            learning_rate=.07,verbose=False,thread_count=1,
            random_seed=seed))]
    for name,model in baselines:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start=time.perf_counter()
            model.fit(train,pu)
            fitsec=time.perf_counter()-start
            start=time.perf_counter()
            p=model.predict_proba(test)[:,1]
            q=model.predict_proba(ext)[:,1]
            psec=time.perf_counter()-start
        record(name,p,q,fitsec,psec,"classic_pu_control",0,"none")
    if len(rows)!=len(BASELINES)+len(PU_MODELS):
        raise RuntimeError("Incomplete controlled challenge")
    print(f"{scenario}/{regime}/{seed}: "+
       "; ".join(f"{r['model']}={r['auc']:.3f}" for r in rows),flush=True)
    return rows

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for scenario in SCENARIOS:
        for regime in REGIMES:
            for seed in SEEDS:
                rows+=one(scenario,regime,seed)
    df=pd.DataFrame(rows)
    expected=len(SCENARIOS)*len(REGIMES)*len(SEEDS)*(len(BASELINES)+len(PU_MODELS))
    if len(df)!=expected:raise RuntimeError("Missing challenge evaluations")
    df.to_csv(OUT/"all_runs.csv",index=False)
    pair=[]
    for (scenario,regime,seed),g in df.groupby(["scenario","regime","seed"]):
        baseline=g[g.model==PU_MODELS[1]].iloc[0]
        for name in (PU_MODELS[0],PU_MODELS[2]):
            student=g[g.model==name].iloc[0]
            for metric_name in ("auc","ap","brier","logloss","boyce",
                                "shift_auc","shift_ap","shift_brier",
                                "shift_logloss","shift_boyce"):
                sign=-1 if metric_name.endswith(("brier","logloss")) else 1
                pair.append(dict(scenario=scenario,regime=regime,seed=seed,
                    student=name,baseline=PU_MODELS[1],metric=metric_name,
                    gain=sign*(student[metric_name]-baseline[metric_name]),
                    selected_rounds=int(student.residual_rounds),
                    verified_choice=student.verified_choice))
    diffs=pd.DataFrame(pair)
    diffs.to_csv(OUT/"paired_effect.csv",index=False)
    lines=["# Native PU residual-gradient forest: predeclared controlled challenge","",
      "Three synthetic mechanisms: XOR interactions, linear outcome, and",
      "an independent-feature null; SCAR and feature-dependent SAR masks.",
      "Five fixed seeds, 820 PU training examples, 320 verified-label",
      "audit examples and two disjoint 420-example test cohorts.",
      "Verified audit outcomes only gate already fitted models; they do",
      "not train trees or networks, and independent test outcomes do",
      "not select hyperparameters or model families.",
      "",
      "| Mechanism | PU confirmation | Teacher AUC | Residual AUC | Verified-gate AUC | Forest selected | Verified gate selected |",
      "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for (scenario,regime),g in df.groupby(["scenario","regime"]):
        get=lambda m:g[g.model==m]
        teacher=get(PU_MODELS[1])
        residual=get(PU_MODELS[0])
        verified=get(PU_MODELS[2])
        chosen=int((residual.residual_rounds>0).sum())
        selected=int((verified.verified_choice=="candidate").sum())
        lines.append(f"| {scenario} | {regime} | {teacher.auc.mean():.4f} | "
          f"{residual.auc.mean():.4f} | {verified.auc.mean():.4f} | "
          f"{chosen}/{len(SEEDS)} | {selected}/{len(SEEDS)} |")
    lines += ["","## Held-out differential performance and timing","",
       "| Mechanism | Regime | Method | AUC | AP | Brier | Shifted AUC | Total seconds |",
       "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for (scenario,regime,model),g in df.groupby(["scenario","regime","model"]):
        lines.append(f"| {scenario} | {regime} | {model} | "
          f"{g.auc.mean():.4f} | {g.ap.mean():.4f} | {g.brier.mean():.4f} | "
          f"{g.shift_auc.mean():.4f} | {g.total_seconds.mean():.3f} |")
    lines += ["","## Interpretation restrictions","",
       "These are deliberately designed synthetic mechanisms, not an",
       "untouched confirmatory experiment or a field/cohort evaluation.",
       "The verified-label audit uses outcomes unavailable to a typical",
       "presence-only workflow. Results from SCAR cannot justify use",
       "under SAR/SNAR or targeted pseudo-absence sampling.",
       "A selected forest can fail to generalize even when all empirical",
       "PU risk checks pass. Keep the neural fallback and do not select",
       "new hyperparameters using the held-out test observations.",
       "Training times of verified, teacher and residual predictions",
       "SHARE the same initial model fit and must not be added.",
       "Per-seed data and paired effects: all_runs.csv, paired_effect.csv.",
       ""]
    (OUT/"RESULTS.md").write_text("\n".join(lines),encoding="utf8")
    manifest=dict(scenarios=list(SCENARIOS),regimes=list(REGIMES),
        seeds=list(SEEDS),train=820,verified_audit=320,test=420,shift=420,
        total_predictions=len(df),independent_training_fits=
          len(SCENARIOS)*len(REGIMES)*len(SEEDS)*(len(BASELINES)+1),
        verified_labels_train_neural_or_tree=False,
        test_labels_used_for_selection=False,
        is_synthetic=True,external_clinical_validation=False,
        version_sha=os.environ.get("GITHUB_SHA","local"))
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",
                                      encoding="utf8")

if __name__=="__main__":
    main()
