"""Independent six-region NCEAS species distribution survey evaluation.

Source: disdat 1.1-0, NCEAS Elith et al. 2020.
- Species selected exclusively by training PO support.
- Native and reference estimators train on PO + accessible BG.
- Geographic inner holdout (when feasible) selects ONLY IPP gradient
  stage count; never the independent surveyed PA labels.
- PA survey outcomes loaded only AFTER all model scores are frozen.
- Unweighted/weighted tree ensembles distinguished.
- Model rankings not used for parameter or species selection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

ROOT=Path(__file__).resolve().parents[1]
REGIONS=("AWT","CAN","NSW","NZ","SA","SWI")
R_METHODS=(
 "NeuralForest_PO_neural","NeuralForest_PO_raw",
 "NeuralForest_PO_random","NeuralForest_PO_hidden",
 "NeuralForest_POGrad_teacher","NeuralForest_POGrad_raw",
 "NeuralForest_POGrad_random","NeuralForest_POGrad_hidden",
 "maxnet_lq","maxnet_lqph","ranger_RF",
 "GLM_logistic","GAM_smooth")
PY_METHODS=("RandomForest","BalancedSubsample_RF",
            "ExtraTrees","XGBoost","XGBoost_weighted",
            "CatBoost","CatBoost_weighted")
METHODS=tuple(sorted(set(R_METHODS)|set(PY_METHODS)))
assert len(METHODS)==20

def new_seed(identifier):
    return int.from_bytes(hashlib.sha256(identifier.encode("utf-8")).digest()[:4],
                          "big")%2_000_000_000

def external_scores(labels,predictions):
    y=np.asarray(labels,dtype=int)
    v=np.asarray(predictions,dtype=float)
    if len(y)!=len(v) or not len(y) or np.any(~np.isfinite(v)):
        raise ValueError("Invalid independently surveyed prediction vector")
    if not set(np.unique(y)).issubset({0,1}):
        raise ValueError("Survey outcomes are not binary")
    p=int(y.sum());n=int(len(y)-p)
    if min(p,n)<5:
        return dict(evaluable=False,survey_auc=float("nan"),
                    survey_ap=float("nan"))
    return dict(evaluable=True,survey_auc=float(roc_auc_score(y,v)),
                survey_ap=float(average_precision_score(y,v)))

def one_species(region,info,tempdir):
    sp=str(info["id"]);path=tempdir/sp
    p=pd.read_csv(path/"presence_train.csv")
    bg=pd.read_csv(path/"background_train.csv")
    survey=pd.read_csv(path/"survey_predictors.csv")
    if not list(p)==list(bg)==list(survey):
        raise RuntimeError("Misaligned training-only PO and survey predictors")
    if any(df.isna().to_numpy().any() for df in (p,bg,survey)):
        raise RuntimeError("Missing pre-encoded ecological predictors")
    seed=new_seed(region+"/"+sp)
    report=path/"models.json"
    subprocess.run(["Rscript",str(ROOT/"benchmark"/"po_nceas_multi_models.R"),
      str(path/"presence_train.csv"),
      str(path/"background_train.csv"),
      str(path/"presence_training_coordinates.csv"),
      str(path/"background_training_coordinates.csv"),
      str(path/"survey_predictors.csv"),
      str(report),str(seed)],cwd=ROOT,check=True)
    parsed=json.loads(report.read_text(encoding="utf-8"))
    native=parsed["models"]
    if {x["name"] for x in native}!=set(R_METHODS):
        raise RuntimeError(f"Incomplete 0.10.1 neural+ecological models {sp}")
    prep=parsed["preprocessing"]
    # Same covariate-space pruning as the R estimator, learned only
    # from complete training P/B rows (never the survey).
    removed=set(prep["removed_train_constant_features"] or [])
    cols=[x for x in p.columns if x not in removed]
    if len(cols)!=int(prep["n_active_columns"]) or len(cols)<2:
        raise RuntimeError("Python/R training feature mismatch")
    X=pd.concat([p,bg],ignore_index=True)[cols]
    S=survey[cols]
    y=np.r_[np.ones(len(p),dtype=int),np.zeros(len(bg),dtype=int)]
    if not np.isfinite(X.to_numpy(float)).all():
        raise RuntimeError("Non-finite NCEAS training predictors")
    predictions={}
    def save(name,score,fitsec,predsec,stages=None,
             inner="none",shared=False,implementation=""):
        if name in predictions:
            raise RuntimeError("Duplicate NCEAS benchmark model: "+name)
        s=np.asarray(score,dtype=float)
        if s.ndim!=1 or len(s)!=len(S) or not np.isfinite(s).all() or (
            s<=0).any():
            raise RuntimeError("Invalid relative NCEAS scores: "+name)
        predictions[name]=dict(scores=s,train_seconds=float(fitsec),
            predict_seconds=float(predsec),stages=stages,
            inner_validation=inner,shares_fit=bool(shared),
            implementation=implementation)
    for m in native:
        stage=m.get("selected_stages")
        if isinstance(stage,str) and stage in ("NA","NaN","null",""):
            stage=None
        save(m["name"],m["prediction"],
             m["train_seconds"],m["predict_seconds"],stage,
             m.get("inner_validation","not_applicable"),
             m.get("shared_fit",False),"R_Cpp17_maxnet_or_GAM")
    w=max(1.0,len(bg)/max(1,len(p)))
    baselines=[
      ("RandomForest",RandomForestClassifier(
         n_estimators=200,max_depth=7,min_samples_leaf=4,
         max_features="sqrt",random_state=seed,n_jobs=1)),
      ("BalancedSubsample_RF",RandomForestClassifier(
         n_estimators=200,max_depth=7,min_samples_leaf=4,
         max_features="sqrt",class_weight="balanced_subsample",
         random_state=seed,n_jobs=1)),
      ("ExtraTrees",ExtraTreesClassifier(
         n_estimators=200,max_depth=7,min_samples_leaf=4,
         max_features="sqrt",random_state=seed,n_jobs=1)),
      ("XGBoost",XGBClassifier(
         n_estimators=180,max_depth=4,learning_rate=.06,
         subsample=.85,colsample_bytree=.85,tree_method="hist",
         eval_metric="logloss",random_state=seed,n_jobs=1)),
      ("XGBoost_weighted",XGBClassifier(
         n_estimators=180,max_depth=4,learning_rate=.06,
         subsample=.85,colsample_bytree=.85,tree_method="hist",
         scale_pos_weight=w,eval_metric="logloss",
         random_state=seed,n_jobs=1)),
      ("CatBoost",CatBoostClassifier(
         iterations=180,depth=5,learning_rate=.06,verbose=False,
         random_seed=seed,thread_count=1)),
      ("CatBoost_weighted",CatBoostClassifier(
         iterations=180,depth=5,learning_rate=.06,verbose=False,
         class_weights=[1.0,w],random_seed=seed,thread_count=1))]
    for name,model in baselines:
        begin=time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X,y)
        fitsec=time.perf_counter()-begin
        begin=time.perf_counter()
        proba=np.clip(model.predict_proba(S)[:,1],1e-8,1-1e-8)
        predsec=time.perf_counter()-begin
        save(name,proba/(1-proba),fitsec,predsec,
             implementation="Python_native_trees_or_boosting")
    if set(predictions)!=set(METHODS):
        raise RuntimeError(f"Incomplete NCEAS model registry for {region}/{sp}")
    # THIS FILE IS NOT ACCESSED UNTIL ALL FITS & SCORES ARE COMPLETE.
    labels=pd.read_csv(path/"survey_outcomes_AUDIT_ONLY.csv")[
                          "survey_outcome"].to_numpy(dtype=int)
    if len(labels)!=len(S):
        raise RuntimeError("Survey alignment lost after fitting")
    rows=[]
    for model,x in predictions.items():
        scored=external_scores(labels,x["scores"])
        rows.append(dict(
          region=region,species=sp,group=info["group"],model=model,
          n_training_presence=len(p),n_background=len(bg),
          n_survey=len(S),n_survey_positive=int(labels.sum()),
          n_survey_nondetection=int((labels==0).sum()),
          n_original_covariates=int(prep["n_original_columns"]),
          n_model_covariates=int(prep["n_active_columns"]),
          inner_validation=str(x["inner_validation"]),
          gradient_stages=x["stages"],
          fit_shared_with_other_prediction=x["shares_fit"],
          train_seconds=x["train_seconds"],predict_seconds=x["predict_seconds"],
          total_seconds=x["train_seconds"]+x["predict_seconds"],
          model_tuned_on_survey_labels=False,
          model_selected_on_survey_labels=False,
          implementation=x["implementation"],**scored))
    lookup={r["model"]:r for r in rows}
    print(f"{region}/{sp}: AUC hidden={lookup['NeuralForest_PO_hidden']['survey_auc']:.4f}, "
      f"gradient={lookup['NeuralForest_POGrad_hidden']['survey_auc']:.4f}, "
      f"maxnet={lookup['maxnet_lqph']['survey_auc']:.4f}, "
      f"catboost={lookup['CatBoost']['survey_auc']:.4f}, "
      f"gradient_inner={prep['inner_validation_type']}",flush=True)
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--region",choices=REGIONS,required=True)
    ap.add_argument("--max-species",type=int,default=4)
    ap.add_argument("--min-presences",type=int,default=70)
    args=ap.parse_args()
    temp=ROOT/"benchmark"/"po_nceas_multi_tmp"/args.region
    out=ROOT/"benchmark"/"po_nceas_multi_out"
    out.mkdir(parents=True,exist_ok=True)
    subprocess.run(["Rscript",str(ROOT/"benchmark"/"po_nceas_multi_export.R"),
        args.region,str(args.max_species),str(args.min_presences),str(temp)],
        check=True,cwd=ROOT)
    manifest=json.loads((temp/"manifest.json").read_text(encoding="utf-8"))
    infos=manifest["selected_species"]
    if isinstance(infos,dict):infos=[infos]
    if not isinstance(infos,list):
        raise RuntimeError("Invalid training-only selected-species manifest")
    results=[]
    for info in infos:
        results+=one_species(args.region,info,temp)
    df=pd.DataFrame(results)
    expected=len(infos)*len(METHODS)
    if expected and (len(df)!=expected or df[
        ["region","species","model"]].duplicated().any()):
        raise RuntimeError("Missing NCEAS cross-model species pairs")
    if expected:
        df.to_csv(out/f"{args.region}.csv",index=False)
    else:
        # Explicitly record no training-supported species: do not
        # fabricate regional performance or silently drop a region.
        columns=("region","species","group","model","evaluable",
                 "survey_auc","survey_ap")
        pd.DataFrame(columns=columns).to_csv(
                         out/f"{args.region}.csv",index=False)
    summary=dict(region=args.region,
      species_examined=len(infos),predeclared_minimum=args.min_presences,
      limit=args.max_species,available_species_above_support=
      manifest["n_available_species_above_minimum"],
      methods=list(METHODS),models_per_species=len(METHODS),
      prediction_records=len(df),
      independent_fits=len(df)-len(infos),
      geographic_inner_validation_mechanism=
         "training-only x/y geographic band; flagged random fallback",
      reference_dataset="Elith et al 2020; CRAN disdat 1.1-0",
      data_source_version=manifest["data_package_version"],
      semantic_background="environmental availability, not absence",
      primary_outcome="independent survey ROC AUC",
      secondary_outcome="independent survey average precision",
      survey_nondetection_not_guaranteed_true_unoccupied=True,
      survey_outcomes_used_for_fitting=False,
      survey_outcomes_used_for_species_selection=False,
      experiment_confirmatory=False,
      github_sha=os.getenv("GITHUB_SHA","local"))
    (out/f"{args.region}.json").write_text(
            json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    evaluable=int(df[df.evaluable==True].species.nunique()) if expected else 0
    print(f"{args.region}: {len(infos)} selected species, {evaluable} "
          f"evaluable independent survey, {len(df)} model predictions",
          flush=True)

if __name__=="__main__":main()
