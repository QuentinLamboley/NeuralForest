"""Unfiltered paired virtual-species effort-offset result audit.

No test outcome is used for model fitting or hyperparameter selection.
The simulator's true ecological habitat suitability is used ONLY
post-prediction. Spatial folds share training regions and therefore
do not constitute independent ecological field studies. This pilot
cannot be presented as real-species independent validation.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"benchmark"/"po_effort_out"
RESULTS=ROOT/"benchmark"/"po_effort_results"
LANDSCAPES=("ecotone","interaction")
METHODS=(
 "IPP_neural_naive","IPP_neural_true","IPP_neural_proxy",
 "IPP_gradient_naive_hidden","IPP_gradient_true_hidden",
 "IPP_gradient_true_raw","IPP_gradient_true_random",
 "IPP_gradient_proxy_hidden","GLM_naive","GLM_true",
 "MaxEnt_lqph","CatBoost_naive","CatBoost_effort_weighted")
CASES=("landscape","effort_condition","seed","outer_fold")
REGIMES=("landscape","effort_condition")
METRICS=("oracle_spearman","oracle_ipp_logskill",
         "observation_biased_auc_pb","train_seconds","predict_seconds")
CONTRASTS=(
 ("IPP_neural_true","IPP_neural_naive"),
 ("IPP_neural_proxy","IPP_neural_naive"),
 ("IPP_gradient_true_hidden","IPP_gradient_naive_hidden"),
 ("IPP_gradient_proxy_hidden","IPP_gradient_naive_hidden"),
 ("IPP_gradient_true_hidden","IPP_gradient_true_raw"),
 ("IPP_gradient_true_hidden","IPP_gradient_true_random"),
 ("IPP_gradient_true_hidden","IPP_neural_true"),
 ("IPP_gradient_true_hidden","CatBoost_effort_weighted"),
 ("IPP_gradient_true_hidden","MaxEnt_lqph"),
 ("GLM_true","GLM_naive"),
 ("CatBoost_effort_weighted","CatBoost_naive"))

def main():
    files=[SOURCE/f"{s}.csv" for s in LANDSCAPES]
    if any(not p.exists() for p in files):
        raise RuntimeError("Every prespecified habitat mechanism must complete")
    RESULTS.mkdir(parents=True,exist_ok=True)
    all_df=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
    if all_df[[*CASES,"model"]].duplicated().any() or (
      set(all_df.model)!=set(METHODS)):
        raise RuntimeError("Missing/duplicated effort model references")
    n_cases=all_df[list(CASES)].drop_duplicates()
    expected=2*2*3*3
    if len(n_cases)!=expected or len(all_df)!=expected*len(METHODS):
        raise RuntimeError(f"Wrong preregistered effort challenge size "
                           f"{len(n_cases)},{len(all_df)}")
    for name in METHODS:
        if len(all_df[all_df.model==name])!=expected:
            raise RuntimeError("Incomplete effort estimator grid: "+name)
    for metric in ("oracle_spearman","oracle_ipp_logskill",
                   "observation_biased_auc_pb","train_seconds"):
        if not np.isfinite(all_df[metric].to_numpy(float)).all():
            raise RuntimeError(f"Invalid native effort model {metric}")
    all_df.to_csv(RESULTS/"all_runs.csv",index=False)
    means=all_df.groupby([*REGIMES,"model"],sort=True)[
         list(METRICS)].mean(numeric_only=True).reset_index()
    means.to_csv(RESULTS/"group_means.csv",index=False)
    paired=[]
    for student,reference in CONTRASTS:
        a=all_df[all_df.model==student].set_index(list(CASES)).sort_index()
        b=all_df[all_df.model==reference].set_index(list(CASES)).reindex(a.index)
        if a.index.has_duplicates or b.isna().all(axis=1).any():
            raise RuntimeError("Unpaired virtual species effort baseline")
        for metric in METRICS:
            av=a[metric].to_numpy(float);bv=b[metric].to_numpy(float)
            for key,gain in zip(a.index,av-bv):
                paired.append(dict(**dict(zip(CASES,key)),
                     student=student,reference=reference,
                     metric=metric,difference=float(gain)))
    differences=pd.DataFrame(paired)
    differences.to_csv(RESULTS/"paired_effect.csv",index=False)
    summary=[]
    for key,g in differences.groupby(
        [*REGIMES,"student","reference","metric"],sort=True):
        d=g.difference[np.isfinite(g.difference)].to_numpy(float)
        if not len(d):continue
        summary.append(dict(**dict(zip(
             [*REGIMES,"student","reference","metric"],key)),
             n=len(d),mean_delta=float(d.mean()),
             wins=int((d>1e-9).sum()),
             losses=int((d< -1e-9).sum()),
             ties=int((abs(d)<=1e-9).sum()),
             std_across_correlated_folds=float(d.std(ddof=1))
                if len(d)>1 else np.nan))
    summ=pd.DataFrame(summary)
    summ.to_csv(RESULTS/"paired_summary.csv",index=False)
    stages=all_df[all_df.model.str.contains("gradient")].groupby(
        [*REGIMES,"model"]).selected_stages.agg(
           ["count","mean",lambda z:int((z>0).sum())]).reset_index()
    stages.columns=[*REGIMES,"model","n","mean_stages","selected_positive"]
    stages.to_csv(RESULTS/"stage_selections.csv",index=False)
    lines=[
      "# Independent virtual-species recorder effort-offset challenge","",
      "**Development simulation, NOT independent empirical ecological validation.**","",
      "Observed presence process is habitat intensity MULTIPLIED by an",
      "independently generated recorder-effort process. Training background",
      "samples describe environmental availability, NOT true absence.",
      "True ecological intensity is known only to the simulator and used",
      "only for final post-fit scoring. Test landscapes and observer effort",
      "never enter estimator training or inner stage selection.",
      "",
      "Hypotheses were set before running this new simulation suite:",
      "an explicit known-effort IPP offset should reduce habitat/effort",
      "confounding under biased observation, while nearly matching",
      "the naive formulation under constant effort. A deliberately",
      "imperfect independently generated effort proxy measures fragility.",
      "An effort-weighted CatBoost control evaluates whether any gain",
      "is simply generic class-background sampling correction.",
      "",
      "Three independent random seeds × three geographically separate",
      "outer folds × two virtual habitats × two observation-effort",
      "regimes = 36 paired conditions, 13 models = 468 prediction",
      "records. A different training-only geographical block selects",
      "the number of native forest correction stages. Spatial-fold",
      "scores share much of their training data and are correlated.",
      "",
      "## Ecological oracle Spearman correlation (higher better)","",
      "| Habitat | Recorder bias | Native neural naive | Native neural known q | Gradient naive | Gradient known q | Gradient proxy q | MaxEnt naive | CatBoost effort-weighted |",
      "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for (land,regime),g in means.groupby(list(REGIMES),sort=True):
        m=g.set_index("model")
        def get(k):return float(m.loc[k,"oracle_spearman"])
        lines.append(
           f"| {land} | {regime} | "
           f"{get('IPP_neural_naive'):.4f} | "
           f"{get('IPP_neural_true'):.4f} | "
           f"{get('IPP_gradient_naive_hidden'):.4f} | "
           f"{get('IPP_gradient_true_hidden'):.4f} | "
           f"{get('IPP_gradient_proxy_hidden'):.4f} | "
           f"{get('MaxEnt_lqph'):.4f} | "
           f"{get('CatBoost_effort_weighted'):.4f} |")
    lines+=["","## Paired oracle suitability effects, all cases","",
      "| Habitat | Recorder bias | Estimator contrast | Mean oracle rho gain | Wins/cases |",
      "| --- | --- | --- | ---: | ---: |"]
    for r in summ[summ.metric=="oracle_spearman"].itertuples():
        lines.append(f"| {r.landscape} | {r.effort_condition} | "
          f"{r.student} vs {r.reference} | "
          f"{r.mean_delta:+.4f} | {r.wins}/{r.n} |")
    lines+=["","## Important interpretation safeguards","",
      "- The detection effort is **known by oracle in simulation**. Real",
      "  opportunistic datasets rarely provide this precise independent",
      "  surface; inaccurate proxy effort can make correction worse.",
      "- Applying an offset is a standard point-process idea and alone",
      "  is NOT a new algorithm. The potential contribution requires",
      "  native objective-aligned C++ forests, sampling-effort robustness,",
      "  matched raw/random-feature attribution and real-world external",
      "  ecological improvements beyond DeepMaxent and MaxEnt.",
      "- The research simulation has been generated during model",
      "  development: even if it favors NeuralForest, it cannot",
      "  establish universal superiority or independent confirmation.",
      "- Full generalization requires genuine independent survey",
      "  detection histories, spatially/temporally held-out surveys,",
      "  uncertainty of effort proxies and equal-budget spatial tuning.",
      "- Presence-background AUC can be artificially high because",
      "  recorder bias is predictable; Oracle habitat correlation is",
      "  the PRIMARY known-truth metric here.",
      "- Scores are relative habitat intensity, NEVER calibrated",
      "  species occupancy, pathogen infection or abundance estimates.",
      "",
      "See all_runs.csv, paired_effect.csv, paired_summary.csv,",
      "stage_selections.csv and group_means.csv for every metric,",
      "timing and unfavorable case."
    ]
    (RESULTS/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    manifest=dict(
       predeclared_cases=expected,models=len(METHODS),
       prediction_records=len(all_df),
       full_unbiased_and_biased_detection_design=True,
       true_effort_known_only_in_simulation=True,
       independent_imperfect_effort_proxy=True,
       true_habitat_oracle_excluded_from_all_training=True,
       geographically_disjoint_train_inner_outer=True,
       ecological_ground_truth_used_only_for_final_evaluation=True,
       nonindependent_correlated_spatial_folds=True,
       confirmatory_field_evidence=False,
       github_sha=os.getenv("GITHUB_SHA","local"))
    (RESULTS/"manifest.json").write_text(
            json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(f"Effort offset virtual species audit complete: "
          f"{len(all_df)} predictions across {len(n_cases)} spatial cases",
          flush=True)

if __name__=="__main__":main()
