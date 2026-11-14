"""Spatial species-distribution benchmark aggregation.

No tuning, best-model selection, or outcome-driven reruns are carried
out here. All scores were previously frozen on spatial holdout cells.
Paired comparisons use exactly matched case/fold/seed/background.
Intervals on folds are DESCRIPTIVE, not independent confidence bounds:
spatial folds share much of their training data. No claims of
independent site-level replication or universal superiority.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/"benchmark"/"po_spatial_results"
INPUT=ROOT/"benchmark"/"po_spatial_out"
GROUPS=("scenario","effort_condition","background_policy")
PAIRED=("seed","fold")
METRICS=("auc_pb","boyce","oracle_spearman","oracle_log_skill",
         "effort_spearman","train_seconds","predict_seconds")
BENCHMARKS=(
   "NeuralForest_PO_neural","NeuralForest_PO_raw",
   "NeuralForest_PO_random","NeuralForest_PO_hidden",
   "maxnet_lq","maxnet_lqph","ranger_RF","GLM_logistic","GAM_smooth",
   "RandomForest","BalancedSubsample_RF","ExtraTrees",
   "XGBoost","CatBoost","Ensemble_RF_XGBoost_CatBoost")

def pair(df,student,teacher,metric):
    keys=list(GROUPS)+list(PAIRED)
    a=df[df.model==student].set_index(keys)
    b=df[df.model==teacher].set_index(keys)
    if a.index.has_duplicates or b.index.has_duplicates or set(a.index)!=set(b.index):
        raise ValueError("Nonpaired baseline/experimental outputs")
    aa=a.sort_index();bb=b.reindex(aa.index)
    dif=pd.DataFrame({k:aa.index.get_level_values(k) for k in keys})
    dif["student"]=student;dif["teacher"]=teacher;dif["metric"]=metric
    dif["delta"]=pd.to_numeric(aa[metric],errors="coerce").to_numpy() - (
                   pd.to_numeric(bb[metric],errors="coerce").to_numpy())
    # Seconds are a cost; keep positive delta = additional cost.
    dif["valid"]=np.isfinite(dif.delta)
    return dif

def audit_results():
    files=[INPUT/f"{name}.csv" for name in
           ("linear","nonlinear_interaction","multimodal")]
    if any(not x.exists() for x in files):
        raise RuntimeError("All three ecological scenarios must complete")
    RESULTS.mkdir(parents=True,exist_ok=True)
    raw=pd.concat([pd.read_csv(x) for x in files],ignore_index=True)
    keys=list(GROUPS)+list(PAIRED)+["model"]
    if raw[keys].duplicated().any():
        raise ValueError("Duplicate spatial fold model evaluation")
    if set(raw.model)!=set(BENCHMARKS):
        raise ValueError("Baseline suite incomplete")
    basekeys=list(GROUPS)+list(PAIRED)
    all_cases=raw[basekeys].drop_duplicates()
    for method in BENCHMARKS:
        if raw[raw.model==method][basekeys].drop_duplicates().shape[0]!=len(all_cases):
            raise RuntimeError(f"Missing spatial cells for {method}")
    for k in ["auc_pb","oracle_spearman","oracle_log_skill","train_seconds"]:
        if not np.isfinite(raw[k].astype(float)).all():
            raise RuntimeError(f"Nonfinite performance/time {k}")
    if raw["auc_pb"].min()<0 or raw["auc_pb"].max()>1:
        raise RuntimeError("Impossible presence-background AUC")
    raw.to_csv(RESULTS/"all_runs.csv",index=False)
    raw.groupby([*GROUPS,"model"],dropna=False)[list(METRICS)].mean(
        numeric_only=True).reset_index().to_csv(
            RESULTS/"group_means.csv",index=False)
    contrasts=(
      ("NeuralForest_PO_hidden","NeuralForest_PO_neural"),
      ("NeuralForest_PO_hidden","NeuralForest_PO_raw"),
      ("NeuralForest_PO_hidden","NeuralForest_PO_random"),
      ("NeuralForest_PO_hidden","maxnet_lqph"),
      ("NeuralForest_PO_hidden","ranger_RF"),
      ("NeuralForest_PO_hidden","XGBoost"),
      ("NeuralForest_PO_hidden","CatBoost"),
      ("NeuralForest_PO_hidden","Ensemble_RF_XGBoost_CatBoost"),
      ("NeuralForest_PO_random","maxnet_lqph"),
      ("NeuralForest_PO_neural","maxnet_lqph"))
    pairs=pd.concat([pair(raw,s,t,m) for s,t in contrasts for m in METRICS],
                    ignore_index=True)
    pairs.to_csv(RESULTS/"paired_differences.csv",index=False)
    summary=[]
    for keys,gr in pairs.groupby([*GROUPS,"student","teacher","metric"],sort=True):
        valid=gr.loc[gr.valid,"delta"].to_numpy(dtype=float)
        if not len(valid):continue
        summary.append(dict(**dict(zip([*GROUPS,"student","teacher","metric"],keys)),
           n=int(len(valid)),mean_delta=float(valid.mean()),
           sd_folds=float(valid.std(ddof=1)) if len(valid)>1 else np.nan,
           wins=int(sum(valid>1e-9)),losses=int(sum(valid< -1e-9)),
           ties=int(sum(abs(valid)<=1e-9))))
    pd.DataFrame(summary).to_csv(RESULTS/"paired_summary.csv",index=False)
    # Honest journal reporting: default to multi-metric evidence table,
    # not a leaderboard of only the best cases for NeuralForest.
    lines=[
      "# Presence-only NeuralForest: spatial benchmark (EXPLORATORY)","",
      "The observed-positive records are not labels of the whole landscape.",
      "Background locations describe environmental availability, NOT true",
      "negatives. Therefore AUC_pb is presence-background discrimination,",
      "not occupancy accuracy, and proper binary calibration scores",
      "must not be computed against background-as-absence labels.",
      "",
      "Three virtual species with known relative habitat intensity are",
      "sampled under unbiased and geographically biased detection effort.",
      "Both spatially uniform and effort-matched target-group background",
      "selection are studied. Strict geographical block holdouts are used.",
      "Oracle suitability scores are used ONLY AFTER model predictions",
      "have been frozen, never to train the learned representation.",
      "",
      "No algorithm selection or hyperparameter tuning uses test outcomes.",
      "Methods include native C++ IPP neural, raw/learned/random-tanh forests,",
      "MaxEnt/maxnet (LQ and LQPH), GAM, GLM, ranger RF, ordinary and balanced",
      "RF, ExtraTrees, XGBoost, CatBoost and a fixed voting ensemble.",
      "",
      "## All methods and scenarios: spatially held-out means","",
      "| Scenario | Observation effort | Background policy | Model | AUC_pb | Boyce | Oracle suitability rho | Oracle IPP log-skill | Train seconds |",
      "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    groups=[*GROUPS,"model"]
    for keys,g in raw.groupby(groups,sort=True):
        vals=g[list(METRICS)].mean(numeric_only=True)
        lines.append(f"| {' | '.join(map(str,keys))} | "
          f"{vals.auc_pb:.4f} | {vals.boyce:.4f} | "
          f"{vals.oracle_spearman:.4f} | {vals.oracle_log_skill:.4f} | "
          f"{vals.train_seconds:.3f} |")
    lines += ["","## Paired native hidden-feature vs reference methods","",
       "| Species | Effort | Background | Comparator | Oracle rho change | AUC_pb change | Win folds / comparisons |",
       "| --- | --- | --- | --- | ---: | ---: | ---: |"]
    for g,group in pairs[pairs.student=="NeuralForest_PO_hidden"].groupby(
           [*GROUPS,"teacher"],sort=True):
        temp=group[group.metric=="oracle_spearman"]
        au=group[group.metric=="auc_pb"]
        delta=pd.to_numeric(temp.delta,errors="coerce").dropna()
        auc=pd.to_numeric(au.delta,errors="coerce").dropna()
        lines.append(f"| {' | '.join(map(str,g))} | "
          f"{delta.mean():+.4f} | {auc.mean():+.4f} | "
          f"{int((delta>0).sum())}/{len(delta)} |")
    lines += ["","## Critical publication and interpretation limits","",
       "- The simulator has already been inspected while developing the",
       "  estimator. These outcomes are exploratory and cannot prove generality.",
       "- Spatial folds share much of their training areas, so fold-level",
       "  wins and variance are NOT independent biological replications.",
       "- AUC_pb and Boyce depend on the accessible study region and",
       "  how available environment is sampled.",
       "- Observation effort is known in simulation only; target-group",
       "  matching could fail if the target group has different biases.",
       "- Oracle correlation is only possible for virtual species.",
       "- No fully independent real-species datasets or held-out",
       "  external ecological surveys were evaluated here.",
       "- Learned hidden features require random-projection controls",
       "  of identical dimensions and fixed feature-sampling budgets.",
       "- Comparing two MaxEnt feature sets and one predeclared set of",
       "  tuning parameters does not replace equal-budget nested spatial",
       "  tuning across all methods in a publishable confirmatory study.",
       "- Never claim universal superiority, occupancy probability,",
       "  true absence accuracy, or significance from cherry-picked strata.",
       "",
       "See all_runs.csv, group_means.csv, paired_differences.csv",
       "and paired_summary.csv for machine-readable results."]
    (RESULTS/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    info=dict(input_scenarios=[x.name for x in files],
       rows=len(raw),cases=len(all_cases),methods=len(BENCHMARKS),
       verified_species_absence_labels_seen_by_models=False,
       spatial_blocked=True,fully_independent_real_data=False,
       posthoc_simulator_oracle_metrics=True,
       nested_hyperparameter_tuning=False,
       exploratory_not_confirmatory=True,
       source_sha=os.getenv("GITHUB_SHA","local"))
    (RESULTS/"manifest.json").write_text(
        json.dumps(info,indent=2)+"\n",encoding="utf-8")
    print(f"Presence-only SDM audit: {len(raw)} predictions over "
          f"{len(all_cases)} held-out spatial cases",flush=True)

if __name__=="__main__":
    audit_results()
