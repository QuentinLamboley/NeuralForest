"""Public NCEAS disdat presence-only training and independent survey audit.

Species selected *only* by number of observed PO training records, never
by surveyed true outcomes or evaluation AUC. Native PO IPP forests,
MaxEnt/maxnet, GAM/GLM/ranger and Python tree/boosting controls are
trained with presences and environmental availability ONLY.

Surveyed outcomes (genuine independently collected PA) are loaded
AFTER all models fit, solely for ROC-AUC and AP ranking evaluation.
These tests are not used for fitting, prior selection, calibration or
hyperparameter tuning. Same region/species ecosystem may share sites,
so results across species are not independent field experiments.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

ROOT=Path(__file__).resolve().parents[1]
R_METHODS=(
  "NeuralForest_PO_neural","NeuralForest_PO_raw",
  "NeuralForest_PO_random","NeuralForest_PO_hidden",
  "maxnet_lq","maxnet_lqph","ranger_RF","GLM_logistic","GAM_smooth")
PY_METHODS=("RandomForest","BalancedSubsample_RF",
            "ExtraTrees","XGBoost","CatBoost")
METHODS=set(R_METHODS)|set(PY_METHODS)

def score(true,p):
    true=np.asarray(true,dtype=int)
    p=np.asarray(p,dtype=float)
    if len(true)!=len(p) or not np.isfinite(p).all() or (p<=0).any():
        raise ValueError("Invalid heldout species suitability scores")
    npos=int((true==1).sum());nneg=int((true==0).sum())
    if min(npos,nneg)<5:
        return dict(survey_auc=float("nan"),survey_ap=float("nan"),
                    evaluable=False)
    return dict(survey_auc=float(roc_auc_score(true,p)),
                survey_ap=float(average_precision_score(true,p)),
                evaluable=True)

def new_seed(name):
    return int.from_bytes(
       hashlib.sha256(name.encode("utf-8")).digest()[:4],"big")%2000000000

def one(info,region,folder):
    species=str(info["id"])
    path=folder/species
    pr=pd.read_csv(path/"presence_train.csv")
    bg=pd.read_csv(path/"background_train.csv")
    survey=pd.read_csv(path/"survey_predictors.csv")
    if not list(pr)==list(bg)==list(survey):
        raise RuntimeError("Predictor mismatch across PO/background/test")
    # No surveyed true labels have yet been read.
    seed=new_seed(region+"/"+species)
    output=path/"models.json"
    subprocess.run(["Rscript",str(ROOT/"benchmark"/"po_nceas_models.R"),
        str(path/"presence_train.csv"),
        str(path/"background_train.csv"),
        str(path/"survey_predictors.csv"),
        str(output),str(seed)],check=True,cwd=ROOT)
    rmodels=json.loads(output.read_text(encoding="utf-8"))["models"]
    if {m["name"] for m in rmodels}!=set(R_METHODS):
        raise RuntimeError("Incomplete ecological R/CPP model suite")
    trained={}
    def store(name,p,trainsec,predsec):
        if name in trained:raise RuntimeError("Duplicate fitted method")
        trained[name]=(np.asarray(p,dtype=float),
                       float(trainsec),float(predsec))
    for m in rmodels:
        store(m["name"],m["score"],m["train_seconds"],
              m["predict_seconds"])

    XX=pd.concat([pr,bg],ignore_index=True)
    y=np.r_[np.ones(len(pr),dtype=int),
            np.zeros(len(bg),dtype=int)]
    # Train-only categorical design matrix (avoid test-level leakage).
    design=pd.get_dummies(XX,drop_first=False)
    pred_design=pd.get_dummies(survey,drop_first=False).reindex(
        columns=design.columns,fill_value=0)
    if not np.isfinite(design.to_numpy(float)).all() or not np.isfinite(
       pred_design.to_numpy(float)).all():
        raise RuntimeError("Nonfinite NCEAS model predictors")
    baselines=[
      ("RandomForest",RandomForestClassifier(
        n_estimators=200,max_depth=7,min_samples_leaf=4,
        max_features="sqrt",n_jobs=1,random_state=seed)),
      ("BalancedSubsample_RF",RandomForestClassifier(
        n_estimators=200,max_depth=7,min_samples_leaf=4,
        class_weight="balanced_subsample",
        max_features="sqrt",n_jobs=1,random_state=seed)),
      ("ExtraTrees",ExtraTreesClassifier(
        n_estimators=200,max_depth=7,min_samples_leaf=4,
        max_features="sqrt",n_jobs=1,random_state=seed)),
      ("XGBoost",XGBClassifier(
        n_estimators=180,max_depth=4,learning_rate=.06,
        subsample=.85,colsample_bytree=.85,tree_method="hist",
        eval_metric="logloss",
        scale_pos_weight=max(1,len(bg)/len(pr)),
        random_state=seed,n_jobs=1)),
      ("CatBoost",CatBoostClassifier(
        iterations=180,depth=5,learning_rate=.06,
        class_weights=[1.0,max(1,len(bg)/len(pr))],
        thread_count=1,random_seed=seed,verbose=False))]
    for name,model in baselines:
        begin=time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(design,y)
        fittime=time.perf_counter()-begin
        begin=time.perf_counter()
        p=np.clip(model.predict_proba(pred_design)[:,1],
                  1e-8,1-1e-8)
        predict_time=time.perf_counter()-begin
        store(name,p/(1-p),fittime,predict_time)
    if set(trained)!=METHODS:
        raise RuntimeError("Incomplete NCEAS native and comparator models")

    # Completely separate from every fitting call above.
    truth=pd.read_csv(path/"survey_outcomes_DO_NOT_TRAIN.csv")
    labels=truth["survey_outcome"].to_numpy(int)
    if len(labels)!=len(survey) or not set(labels).issubset({0,1}):
        raise RuntimeError("Misaligned independently surveyed outcomes")
    rows=[]
    for name,(p,fitsec,predsec) in trained.items():
        vals=score(labels,p)
        rows.append(dict(region=region,species=species,
         group=info["group"],model=name,
         n_training_presence=len(pr),n_available_background=len(bg),
         n_survey_sites=len(survey),n_survey_true_positive=int(labels.sum()),
         n_survey_true_negative=int((labels==0).sum()),
         selection_rule="training_PO_count_only",
         model_tuned_on_survey_labels=False,
         train_seconds=fitsec,predict_seconds=predsec,
         total_seconds=fitsec+predsec,**vals))
    by={r["model"]:r for r in rows}
    print(f"{region}/{species}: independent survey AUC "
      f"NeuralForest={by['NeuralForest_PO_hidden']['survey_auc']:.3f}, "
      f"Maxnet={by['maxnet_lqph']['survey_auc']:.3f}, "
      f"CatBoost={by['CatBoost']['survey_auc']:.3f}, "
      f"PA samples={len(labels)}",flush=True)
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--region",default="AWT")
    ap.add_argument("--limit",type=int,default=8)
    args=ap.parse_args()
    temp=ROOT/"benchmark"/"po_nceas_tmp"/args.region
    out=ROOT/"benchmark"/"po_nceas_results"
    out.mkdir(parents=True,exist_ok=True)
    subprocess.run(["Rscript",str(ROOT/"benchmark"/"po_nceas_export.R"),
        args.region,str(args.limit),str(temp)],check=True,cwd=ROOT)
    manifest=json.loads((temp/"input_manifest.json").read_text(
        encoding="utf-8"))
    infos=manifest["species"]
    if isinstance(infos,dict):infos=[infos]
    if not infos:raise RuntimeError("No selected training-supported species")
    results=[]
    for info in infos:
        results.extend(one(info,args.region,temp))
    df=pd.DataFrame(results)
    expected=len(infos)*len(METHODS)
    if len(df)!=expected or df[
        ["region","species","model"]].duplicated().any():
        raise RuntimeError("Incomplete NCEAS out-of-source benchmark")
    df.to_csv(out/f"{args.region}_all_methods.csv",index=False)
    eligible=df[df.evaluable]
    n_species=eligible.species.nunique()
    if n_species==0:raise RuntimeError(
        "No evaluable independent survey species")
    means=eligible.groupby(["model"])[
         ["survey_auc","survey_ap","train_seconds",
          "predict_seconds"]].mean().reset_index()
    means.to_csv(out/f"{args.region}_means.csv",index=False)
    result_info=dict(
       region=args.region,selected_training_only_species=[
         info["id"] for info in infos],
       input_reference="disdat 1.1-0 / Elith et al 2020",
       dataset_doi="10.17161/bi.v15i2.13384",
       test_surveys_truly_separate_from_presence_only_training=True,
       geographic_outer_block_independence=False,
       n_models=len(METHODS),n_model_fits=len(df),
       n_evaluable_species=n_species,
       species_filtering_by_test_performance=False,
       test_data_used_for_tuning=False,
       scores_are_uncalibrated_relative_intensity=True,
       preliminary_training_only_species_count_selection=True,
       paper_ready_confirmatory=False)
    (out/f"{args.region}_manifest.json").write_text(
        json.dumps(result_info,indent=2)+"\n",encoding="utf-8")
    lines=[
       f"# NCEAS independently surveyed ecological test: {args.region}","",
       "**Pilot, not a publishable multi-region conclusion.**",
       "Species were chosen using ONLY their training presence count,",
       "not survey outcomes or performance. Published NCEAS disdat",
       "presence-only training and environmental background were fitted",
       "with 14 algorithm variants and assessed against genuinely",
       "independent designed-survey presence/absence outcomes.",
       "Those surveyed outcomes were read only AFTER all methods fit.",
       "",
       "The survey true negatives are genuinely observed non-detections",
       "under the survey protocol, not assigned to background. No",
       "detection histories or full occupancy probability calibration",
       "are inferred from them. This is separate-source validation,",
       "not necessarily geographic transfer validation.",
       "",
       "| Method | Mean independent AUC | Mean independent AP | Train s |",
       "| --- | ---: | ---: | ---: |"]
    for row in means.itertuples():
       lines.append(f"| {row.model} | {row.survey_auc:.4f} | "
           f"{row.survey_ap:.4f} | {row.train_seconds:.3f} |")
    lines+=["","## Species-by-species AUC, including unfavorable outcomes","",
       "| Species | Group | Survey presences | Survey negatives | NN IPP | Hidden forest | MaxEnt LQPH | CatBoost |",
       "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for sp,group in df.groupby("species",sort=True):
      m={r.model:r for r in group.itertuples()}
      anyrow=next(iter(m.values()))
      def val(k):return f"{m[k].survey_auc:.4f}"
      lines.append(f"| {sp} | {anyrow.group} | "
         f"{anyrow.n_survey_true_positive} | "
         f"{anyrow.n_survey_true_negative} | "
         f"{val('NeuralForest_PO_neural')} | "
         f"{val('NeuralForest_PO_hidden')} | "
         f"{val('maxnet_lqph')} | {val('CatBoost')} |")
    lines+=["","## Interpretation limits","",
      "This is a small pilot from a well-known public benchmark that",
      "has been studied many times. A truly confirmatory study",
      "requires prespecified species/regions, matched hyperparameter",
      "tuning inside spatial outer-training blocks, and comparison",
      "against published DeepMaxent and strong PO-specific RF models.",
      "Independent surveys are not a guarantee of no geographic",
      "autocorrelation or perfect detection.",
      "AUC and AP on surveys are genuine evaluation metrics but do not",
      "calibrate the relative intensity scores into occupancy.",
      "The common available background was subsampled to 2500 training",
      "records for runtime; this is not a full 10k-background benchmark.",
      "All per-species unfavorable results, runtime, sample sizes and",
      "undefined AUCs are retained in the machine-readable CSV.",
      "",
      "Raw disdat species predictors/survey labels are NOT redistributed",
      "by this workflow. Use the published package with attribution.",
      ""]
    (out/f"{args.region}_REPORT.md").write_text(
         "\n".join(lines),encoding="utf-8")
    print(f"Independent surveyed real species audit: {n_species} "
          f"evaluable of {len(infos)} selected, {len(df)} method rows",
          flush=True)

if __name__=="__main__":
    main()
