"""Strictly paired neural hidden-feature PU forest ablation.

Three synthetic mechanisms (XOR, linear, null), SCAR/SAR positive-label
confirmation, three predeclared seeds. Train, verified-audit, original
test and shifted test are generated independently. The only differences
between native RAW and HIDDEN candidates are added frozen neural
activations and bounded feature-sampling mass. Both share identical
neural teacher predictions (enforced by R).
Results are exploratory, NOT site-independent clinical validation.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"benchmark"))
from pu_residual_verified import cohort,make_pu,metric
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

OUT=ROOT/"audit"/"pu_hidden_feature_results"
SEEDS=(5701,5717,5741)
SCENARIOS=("interaction_xor","linear","null")
REGIMES=("SCAR","SAR")
NATIVE=("NeuralForest_PUHiddenTeacher",
        "NeuralForest_PUHiddenRawCandidate",
        "NeuralForest_PUHiddenLatentCandidate",
        "NeuralForest_PUHiddenRawVerified",
        "NeuralForest_PUHiddenLatentVerified")
BASELINES=("RandomForest","ExtraTrees","XGBoost","CatBoost")
METRICS=("auc","ap","brier","logloss","boyce")
MAIN_COLS=("scenario","regime","seed","model")

def one(scenario,regime,seed):
    train,y=cohort(seed+10,820,scenario)
    audit,ya=cohort(seed+50,320,scenario)
    test,yt=cohort(seed+90,420,scenario)
    shifted,ye=cohort(seed+130,420,scenario,shift=True)
    visible=make_pu(train,y,regime,seed+301)
    results=[]
    def put(name,p,q,tfit,tpred,selected="none",latent_share=None,
            verified_sec=0):
        with np.errstate(divide="ignore",invalid="ignore"):
            m=metric(yt,p)
            ex=metric(ye,q)
        if not np.isfinite(m["auc"]) or not np.isfinite(ex["auc"]):
            raise RuntimeError(f"Nonfinite AUC: {name}")
        results.append(dict(scenario=scenario,regime=regime,seed=seed,
            model=name,train_seconds=float(tfit),
            predict_seconds=float(tpred),
            total_seconds=float(tfit+tpred),
            selected=str(selected),
            latent_importance_fraction=latent_share,
            verification_seconds=float(verified_sec),
            **m,**{"shift_"+k:v for k,v in ex.items()}))
    with tempfile.TemporaryDirectory() as td:
        temp=Path(td)
        train.assign(Vampirisme=visible).to_csv(temp/"train.csv",index=False)
        audit.assign(Vampirisme=ya).to_csv(temp/"audit.csv",index=False)
        test.assign(Vampirisme=yt).to_csv(temp/"test.csv",index=False)
        shifted.assign(Vampirisme=ye).to_csv(temp/"shift.csv",index=False)
        subprocess.run(["Rscript",str(ROOT/"benchmark"/"pu_hidden_pilot_r.R"),
          str(temp/"train.csv"),str(temp/"audit.csv"),
          str(temp/"test.csv"),str(temp/"shift.csv"),
          str(temp/"out.json"),str(seed)],check=True,cwd=ROOT)
        models=json.loads((temp/"out.json").read_text(encoding="utf-8"))["models"]
        if {r["name"] for r in models}!=set(NATIVE):
            raise RuntimeError("Incomplete frozen-hidden matched ablation")
        for model in models:
            put(model["name"],model["probability"],
                model["external_probability"],
                model["train_seconds"],model["prediction_seconds"],
                model.get("selected","none"),
                model.get("latent_importance_fraction"),
                model.get("verification_seconds",0))
    baselines=[
      ("RandomForest",RandomForestClassifier(
          n_estimators=130,min_samples_leaf=5,max_depth=5,
          max_features="sqrt",n_jobs=1,random_state=seed)),
      ("ExtraTrees",ExtraTreesClassifier(
          n_estimators=130,min_samples_leaf=5,max_depth=5,
          max_features="sqrt",n_jobs=1,random_state=seed)),
      ("XGBoost",XGBClassifier(n_estimators=130,max_depth=4,
          learning_rate=.07,subsample=.85,colsample_bytree=.9,
          tree_method="hist",eval_metric="logloss",
          random_state=seed,n_jobs=1)),
      ("CatBoost",CatBoostClassifier(iterations=130,depth=5,
          learning_rate=.07,verbose=False,thread_count=1,
          random_seed=seed))]
    for name,model in baselines:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0=time.perf_counter()
            model.fit(train,visible)
            fitsec=time.perf_counter()-t0
            t0=time.perf_counter()
            p=model.predict_proba(test)[:,1]
            q=model.predict_proba(shifted)[:,1]
            predsec=time.perf_counter()-t0
        put(name,p,q,fitsec,predsec,"P_U_label_baseline")
    if len(results)!=len(NATIVE)+len(BASELINES):
        raise RuntimeError("Incorrect comparison count")
    print(f"{scenario}/{regime}/{seed}: "+
       "; ".join(f"{r['model']} AUC={r['auc']:.3f}"
                  for r in results),flush=True)
    return results

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for scenario in SCENARIOS:
      for regime in REGIMES:
        for seed in SEEDS:
            rows+=one(scenario,regime,seed)
    df=pd.DataFrame(rows)
    expect=len(SCENARIOS)*len(REGIMES)*len(SEEDS)*(
         len(NATIVE)+len(BASELINES))
    if len(df)!=expect or df[list(MAIN_COLS)].duplicated().any():
        raise RuntimeError("Incomplete or duplicate hidden feature experiment")
    df.to_csv(OUT/"all_runs.csv",index=False)
    paired=[]
    for (scenario,regime,seed),g in df.groupby(["scenario","regime","seed"]):
        model=lambda name:g[g.model==name].iloc[0]
        base=model(NATIVE[0]);raw=model(NATIVE[1]);latent=model(NATIVE[2])
        raw_gate=model(NATIVE[3]);latent_gate=model(NATIVE[4])
        for name,student,ref in (
            ("hidden_vs_raw",latent,raw),
            ("raw_vs_nnpu",raw,base),
            ("hidden_vs_nnpu",latent,base),
            ("hidden_verified_vs_nnpu",latent_gate,base),
            ("raw_verified_vs_nnpu",raw_gate,base)):
            for m in METRICS:
              for prefix in ("","shift_"):
                key=prefix+m
                sign=-1 if m in ("brier","logloss") else 1
                paired.append(dict(scenario=scenario,regime=regime,
                    seed=seed,contrast=name,
                    cohort="shift" if prefix else "test",
                    metric=m,numerator=student.model,
                    denominator=ref.model,
                    favorable_delta=sign*(student[key]-ref[key])))
    pairs=pd.DataFrame(paired)
    pairs.to_csv(OUT/"paired_effect.csv",index=False)
    lines=["# Native nnPU hidden-feature-guided forest, paired ablation",
       "",
       "Predeclared training-only comparisons: original numeric features",
       "versus identical original features plus frozen C++ hidden nnPU",
       "activations. The native neural teachers are tested for EXACT",
       "agreement across raw and augmented fits for every seed.",
       "Sole ablated factors: the feature coordinates and constrained",
       "candidate-feature sampling mass. All other native tree settings",
       "and the per-seed training P/U indicators are held fixed.",
       "",
       "The candidate scores shown below were NOT selected on true labels.",
       "They are exploratory research ablations, not deployable predictors.",
       "Guarded versions use a distinct verified-outcome cohort to",
       "choose candidate vs the matching neural teacher, on two halves.",
       "An entirely separate test and shifted test remain untouched.",
       "The supervised controls classify observed P/U indicators and",
       "are not assumed to estimate true positive status under SAR.",
       "",
       "| Mechanism | PU labels | Teacher AUC | Raw forest AUC | Hidden forest AUC | Hidden minus raw | Verified hidden AUC | Gated selections |",
       "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for (scenario,regime),group in df.groupby(["scenario","regime"]):
        get=lambda m:group[group.model==m]
        teacher=get(NATIVE[0])
        raw=get(NATIVE[1])
        latent=get(NATIVE[2])
        gated=get(NATIVE[4])
        selected=int((gated.selected=="candidate").sum())
        lines.append(f"| {scenario} | {regime} | {teacher.auc.mean():.4f} | "
           f"{raw.auc.mean():.4f} | {latent.auc.mean():.4f} | "
           f"{latent.auc.mean()-raw.auc.mean():+.4f} | "
           f"{gated.auc.mean():.4f} | {selected}/{len(SEEDS)} |")
    lines+=["","## All controls, multi-metric tests and time","",
      "| Mechanism | Regime | Method | AUC | PR-AUC | Brier | Shift AUC | Train s | Predict s | Hidden importance share |",
      "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for (scenario,regime,model),g in df.groupby(["scenario","regime","model"]):
        latent=pd.to_numeric(g.latent_importance_fraction,errors="coerce").mean()
        lines.append(f"| {scenario} | {regime} | {model} | "
          f"{g.auc.mean():.4f} | {g.ap.mean():.4f} | "
          f"{g.brier.mean():.4f} | {g.shift_auc.mean():.4f} | "
          f"{g.train_seconds.mean():.3f} | {g.predict_seconds.mean():.3f} | "
          f"{latent:.3f} |")
    lines+=["","## Scope and limitations","",
       "This is a synthetic research simulation, not a real-world",
       "independent clinical or ecological field validation.",
       "XOR and null settings were inspected in earlier audits, so",
       "this cannot be treated as an untouched confirmation test.",
       "The neural hidden units are learned in-sample from PU labels;",
       "only the subsequent evaluation/test cases are fully held out.",
       "Under SAR/SNAR and selected background, P/U classification",
       "may be a systematically biased surrogate.",
       "The raw and hidden candidates were each trained independently",
       "with the same deterministic teacher seed, not from a shared",
       "fit in memory: their training costs are REAL and separate.",
       "The teacher row is an ablation of the raw model's trained",
       "networks: do not add its recorded training time to raw total.",
       "The experiment must not be used to tune on the test outcomes.",
       "See all_runs.csv and paired_effect.csv for all per-seed metrics.",
       ""]
    (OUT/"RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
    manifest=dict(scenarios=list(SCENARIOS),regimes=list(REGIMES),
       seeds=list(SEEDS),native_versions=list(NATIVE),
       baseline_controls=list(BASELINES),
       expected_records=expect,actual_records=len(df),
       derived_nnpu_teacher_shared_with_raw_forest=True,
       raw_and_hidden_nnpu_identical_seed=True,
       audit_true_outcomes_resolved_separately=True,
       test_labels_used_for_any_training_or_selection=False,
       external_clinical_validation=False,
       commit=os.environ.get("GITHUB_SHA","local"))
    (OUT/"manifest.json").write_text(
       json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print("Frozen hidden-feature paired ablation complete:",len(df),"prediction records")

if __name__=="__main__":
    main()
