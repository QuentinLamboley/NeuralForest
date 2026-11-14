"""Unfiltered descriptive nested-spatial presence-only gradient audit.

AUC_pb is NOT discrimination of true presence versus absence.
Virtual-species oracle suitability correlation is only possible in
simulation. The inner-block IPP loss selects additive stages; no outer
test score selects features/hyperparameters/boost depth or method.
Outer folds share training areas and are not independent replications.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/"benchmark"/"po_gradient_out"
OUTPUT=ROOT/"benchmark"/"po_gradient_results"
SCENARIOS=("linear","nonlinear_interaction","multimodal")
GROUPS=("scenario","effort_condition","background_policy")
CASE_KEYS=GROUPS+("seed","outer_block")
METRICS=("oracle_spearman","oracle_log_skill","auc_pb","boyce",
         "effort_spearman","train_seconds","predict_seconds")
MODELS=("NeuralForest_POGrad_raw","NeuralForest_POGrad_random",
        "NeuralForest_POGrad_hidden","NeuralForest_POGrad_teacher",
        "maxnet_lqph","GAM_smooth","RandomForest","XGBoost","CatBoost")

def main():
    files=[INPUT/f"{name}.csv" for name in SCENARIOS]
    if any(not p.exists() for p in files):
        raise RuntimeError("All three virtual species required")
    OUTPUT.mkdir(parents=True,exist_ok=True)
    data=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
    if set(data.model)!=set(MODELS):
        raise RuntimeError("Incomplete registered reference algorithms")
    if data[[*CASE_KEYS,"model"]].duplicated().any():
        raise RuntimeError("Duplicated spatial fold predictions")
    cases=data[list(CASE_KEYS)].drop_duplicates()
    expected=3*2*2*2*3
    if len(cases)!=expected or len(data)!=expected*len(MODELS):
        raise RuntimeError(f"Expected {expected} cases and "
                           f"{expected*len(MODELS)} predictions, found "
                           f"{len(cases)} cases and {len(data)} rows")
    for model in MODELS:
        if data[data.model==model][list(CASE_KEYS)].drop_duplicates().shape[0]!=expected:
            raise RuntimeError(f"Missing cases for {model}")
    if (data["inner_block"]==data["outer_block"]).any():
        raise RuntimeError("Selected inner block leaks outer evaluation")
    native=data[data.model.str.startswith("NeuralForest_POGrad_")]
    if not bool(native.loc[native.model!="NeuralForest_POGrad_teacher",
                  "selected_stages"].dropna().between(0,4).all()):
        raise RuntimeError("Gradient stage count is outside protocol")
    for metric in ("auc_pb","oracle_spearman","oracle_log_skill",
                   "train_seconds","predict_seconds"):
        if not np.isfinite(data[metric].to_numpy(float)).all():
            raise RuntimeError(f"Invalid {metric}")
    data.to_csv(OUTPUT/"all_runs.csv",index=False)
    mean=data.groupby([*GROUPS,"model"],dropna=False)[list(METRICS)].mean(
        numeric_only=True).reset_index()
    mean.to_csv(OUTPUT/"group_means.csv",index=False)
    pairs=[]
    contrasts=(
      ("NeuralForest_POGrad_hidden","NeuralForest_POGrad_teacher"),
      ("NeuralForest_POGrad_hidden","NeuralForest_POGrad_raw"),
      ("NeuralForest_POGrad_hidden","NeuralForest_POGrad_random"),
      ("NeuralForest_POGrad_hidden","maxnet_lqph"),
      ("NeuralForest_POGrad_hidden","GAM_smooth"),
      ("NeuralForest_POGrad_hidden","RandomForest"),
      ("NeuralForest_POGrad_hidden","XGBoost"),
      ("NeuralForest_POGrad_hidden","CatBoost"),
      ("NeuralForest_POGrad_raw","NeuralForest_POGrad_teacher"),
      ("NeuralForest_POGrad_random","NeuralForest_POGrad_teacher"))
    case_keys=list(CASE_KEYS)
    for student,teacher in contrasts:
        a=data[data.model==student].set_index(case_keys).sort_index()
        b=data[data.model==teacher].set_index(case_keys).reindex(a.index)
        if a.index.has_duplicates or b.isna().all(axis=1).any():
            raise RuntimeError("Unpaired spatial case")
        for m in METRICS:
            av=pd.to_numeric(a[m],errors="coerce").to_numpy(float)
            bv=pd.to_numeric(b[m],errors="coerce").to_numpy(float)
            for i,(keys,d) in enumerate(zip(a.index,av-bv)):
                if not isinstance(keys,tuple):keys=(keys,)
                pairs.append(dict(**dict(zip(case_keys,keys)),
                  student=student,teacher=teacher,metric=m,
                  difference=float(d),finite=bool(np.isfinite(d))))
    contrasts_df=pd.DataFrame(pairs)
    contrasts_df.to_csv(OUTPUT/"paired_effect.csv",index=False)
    summary=[]
    for keys,g in contrasts_df.groupby([
        *GROUPS,"student","teacher","metric"],sort=True):
        v=g.difference[np.isfinite(g.difference)].to_numpy(float)
        if not len(v):continue
        summary.append(dict(**dict(zip([
            *GROUPS,"student","teacher","metric"],keys)),
            mean_difference=float(v.mean()),n=len(v),
            wins=int((v>1e-9).sum()),
            ties=int((np.abs(v)<=1e-9).sum()),
            losses=int((v< -1e-9).sum()),
            sd=float(v.std(ddof=1)) if len(v)>1 else None))
    pd.DataFrame(summary).to_csv(OUTPUT/"paired_summary.csv",index=False)
    stage=native.groupby([*GROUPS,"model"])["selected_stages"].agg(
        ["count","mean",lambda x:int((x.fillna(0)>0).sum())]).reset_index()
    stage.columns=[*GROUPS,"model","n","mean_stages","selected_positive"]
    stage.to_csv(OUTPUT/"stage_selection.csv",index=False)
    report=[
       "# Nested-spatial point-process gradient forest benchmark","",
       "**This is an exploratory simulation, NOT an external ecological",
       "validation or a demonstration of universal superiority.**","",
       "All model predictions are paired on 72 geographically held-out",
       "cases: three virtual species × unbiased/biased observation effort",
       "× uniform/effort-matched background × two independent seeds ×",
       "three outer geographic blocks. A *different* geographic block",
       "inside each outer training set selects the number of C++ IPP",
       "gradient forest stages; after selection, fits use all outer",
       "training blocks. The outer block is never used in model selection.",
       "Each of nine model predictions is recorded per case (648 rows).",
       "",
       "Background is available environment and can contain presences.",
       "Neither AUC_pb nor Boyce identifies absolute occupancy. Synthetic",
       "oracle Spearman checks whether relative scores reflect the TRUE",
       "virtual habitat intensity without exposing it to training.",
       "No hyperparameter search used evaluation outcomes.","",
       "## All algorithms and conditions: no favorable subset omitted","",
       "| Species | Effort | Background | Model | Oracle Spearman | Oracle IPP log skill | AUC_pb | Boyce | Train s |",
       "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in mean.itertuples(index=False):
        report.append(
          f"| {row.scenario} | {row.effort_condition} | "
          f"{row.background_policy} | {row.model} | "
          f"{row.oracle_spearman:.4f} | {row.oracle_log_skill:.4f} | "
          f"{row.auc_pb:.4f} | {row.boyce:.4f} | "
          f"{row.train_seconds:.3f} |")
    report+=["","## Gradient forest vs its EXACT neural teacher, random-feature control and strong competitors","",
      "| Species | Effort | Background | Reference | Oracle rho change | AUC_pb change |",
      "| --- | --- | --- | --- | ---: | ---: |"]
    for keys,g in contrasts_df[
      contrasts_df.student=="NeuralForest_POGrad_hidden"
      ].groupby([*GROUPS,"teacher"]):
        oracle=g[g.metric=="oracle_spearman"].difference.mean()
        auc=g[g.metric=="auc_pb"].difference.mean()
        report.append(f"| {' | '.join(map(str,keys))} | "
                      f"{oracle:+.4f} | {auc:+.4f} |")
    report+=["","## Guardrails and unresolved questions","",
       "- The native model fits an IPP-negative-gradient regression forest.",
       "  It does NOT fit false binary biological absences.",
       "- Spatial outer blocks and a separate spatial inner block mitigate",
       "  spatially optimistic selection but do NOT create 72 independent",
       "  real-world replicate species. Inner-fold selection is a single",
       "  fixed 1-of-4 training-block rule, not equal-budget nested tuning.",
       "- MaxEnt LQPH, GAM, and CatBoost are substantive competitors;",
       "  superiority should be examined across ALL relevant metrics.",
       "- Variation in sampling/detection effort is a separate causal",
       "  source of bias; an effort-matched background is not a guarantee",
       "  of unbiased occurrence observation.",
       "- Oracle correlation and oracle IPP skill are known only for",
       "  virtual species. Independent empirical field validation",
       "  still requires new true ecological survey evidence.",
       "- Existing Ixodes hard pseudoabsences sampled outside an occupied",
       "  FAMD convex hull are NOT an unbiased accessible-background",
       "  sample. They are a separate sensitivity design.",
       "- No true occurrence prevalence, disease risk, abundance or",
       "  calibrated occupancy can be inferred from these scores.",
       "- Fold-level dispersion is exploratory because fold training",
       "  sets overlap and the simulator families were previously",
       "  examined during method development.",
       "",
       "See all_runs.csv, group_means.csv, stage_selection.csv,",
       "paired_effect.csv and paired_summary.csv for exhaustive results."]
    (OUTPUT/"REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    manifest=dict(
       n_scenarios=3,n_spatial_cases=len(cases),
       n_methods=len(MODELS),n_predictions=len(data),
       n_training_fits=len(data)-len(cases),
       note="Exact teacher prediction shares the raw native fit; R modes are separate independent fits.",
       spatially_disjoint_inner_and_outer=True,
       virtual_species_oracle_never_in_training=True,
       background_can_contain_true_presences=True,
       nested_stage_selection_univariate_without_hyperparameter_tuning=True,
       genuine_real_world_external_validation=False,
       experiment_version_sha=os.environ.get("GITHUB_SHA","local"))
    (OUTPUT/"manifest.json").write_text(
      json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(f"Nested spatial gradient IPP report: {len(data)} records, "
          f"{len(cases)} cases",flush=True)

if __name__=="__main__":
    main()
