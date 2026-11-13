"""Aggregate benchmark performance, selection overhead and time to predict."""
from __future__ import annotations
import json
from pathlib import Path
import platform
import sys
import numpy as np
import pandas as pd
from multimetric import build_reports

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
RAW=HERE/"out"
RESULTS=HERE/"results"
BASELINE=ROOT/"audit"/"baseline-v0.2-all_runs.csv"
SEEDS=(17,43,71,97,131,197,251,347,401,503)
NEW_SEEDS=(401,503)
SCENARIOS=19
MODELS=31
METRICS=("roc_auc","pr_auc","accuracy","balanced_accuracy","precision",
         "recall","specificity","f1","mcc","brier","logloss","boyce",
         "boyce_rank","ece","precision_at_10pct","recall_at_10pct")
MINIMIZE={"brier","logloss"}
def fmt(x,n=3):
    return "NA" if not np.isfinite(x) else f"{x:.{n}f}"
def scores_for(g,metric):
    column="roc_auc" if metric=="auc" else metric
    return g[column]
def summarize():
    paths=sorted(RAW.glob("*.csv"))
    if not paths: raise RuntimeError("No trial CSV files")
    data=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    # Each (scenario, seed, model) is unique. Missing fits remain explicit.
    expected=SCENARIOS*len(SEEDS)*MODELS
    if len(data)!=expected: raise RuntimeError(
        f"Expected {expected} score rows but found {len(data)}")
    if data[["scenario","seed","model"]].duplicated().any():
        raise RuntimeError("Duplicated model-scenario-seed result")
    failed=data[data.status!="ok"]
    if len(failed):
        detail=failed[["scenario","seed","model","error"]].head(12)
        raise RuntimeError(
            f"{len(failed)} of {expected} models failed; refusing to "
            f"publish an incomplete benchmark:\n{detail.to_string(index=False)}")
    pair=data[["scenario","seed"]].drop_duplicates()
    if len(pair)!=SCENARIOS*len(SEEDS) or set(data.seed)!=set(SEEDS):
        raise RuntimeError("Missing or duplicate scenario seeds")
    RESULTS.mkdir(parents=True,exist_ok=True)
    data.to_csv(RESULTS/"all_runs.csv",index=False)
    summary=[]
    for (scenario,model),g in data.groupby(["scenario","model"],sort=True):
        metric=g.selected_metric.iloc[0]
        v=scores_for(g,metric)
        valid=v[np.isfinite(v)]
        fresh=scores_for(g[g.seed.isin(NEW_SEEDS)],metric)
        fresh=fresh[np.isfinite(fresh)]
        row=dict(scenario=scenario,model=model,primary_metric=metric,
           primary_mean=valid.mean() if len(valid) else np.nan,
           primary_sd=valid.std(ddof=1) if len(valid)>1 else np.nan,
           new_seed_primary_mean=fresh.mean() if len(fresh) else np.nan,
           new_seed_primary_sd=fresh.std(ddof=1) if len(fresh)>1 else np.nan,
           n_seeds=len(g),n_success=int((g.status=="ok").sum()),
           mean_selection_seconds=g.selection_seconds.mean(),
           mean_final_fit_seconds=g.final_fit_seconds.mean(),
           mean_train_seconds=g.train_seconds.mean(),
           mean_predict_seconds=g.predict_seconds.mean(),
           mean_total_seconds=g.total_seconds.mean())
        for m in METRICS:
            row[m+"_mean"]=g[m].mean()
            row[m+"_sd"]=g[m].std()
        summary.append(row)
    sm=pd.DataFrame(summary)
    sm["rank"]=np.nan
    sm["new_seed_rank"]=np.nan
    for scenario,g in sm.groupby("scenario"):
        increasing=g.primary_metric.iloc[0] in MINIMIZE
        sm.loc[g.index,"rank"]=g.primary_mean.rank(
            method="min",ascending=increasing,na_option="bottom")
        sm.loc[g.index,"new_seed_rank"]=g.new_seed_primary_mean.rank(
            method="min",ascending=increasing,na_option="bottom")
    sm=sm.sort_values(["scenario","rank","model"])
    sm.to_csv(RESULTS/"summary.csv",index=False)
    multi_manifest=build_reports(data,sm,RESULTS)
    # Keep classical/undersampled forest baselines side by side.
    forest_models=[
        "RandomForest","RF_Bootstrap50","RF_Subsample50","RF_Subsample80",
        "RF_BalancedUndersampled","RF_BalancedUndersampledPrior",
        "ranger","NeuralForest_CART",
        "NeuralForest_SubSample50_Informed",
        "NeuralForest_UndersampledPlain",
        "NeuralForest_UndersampledInformed",
        "NeuralForest_Adaptive","NeuralForest_Calibrated"]
    comparisons=sm[sm.model.isin(forest_models)].copy()
    comparisons["is_undersampled"]=comparisons.model.str.contains(
        "Subsample|SubSample|Undersampled|Bootstrap50",case=False,regex=True)
    comparisons["relative_time_vs_classic_rf"]=np.nan
    comparisons["primary_gap_vs_classic_rf"]=np.nan
    comparisons["primary_advantage_vs_classic_rf"]=np.nan
    for scenario,g in comparisons.groupby("scenario"):
        rf=g[g.model=="RandomForest"]
        if len(rf)!=1: raise RuntimeError(f"Missing classic RF for {scenario}")
        root=rf.iloc[0]
        same=comparisons.scenario==scenario
        base_time=float(root.mean_total_seconds)
        comparisons.loc[same,"relative_time_vs_classic_rf"]=(
            comparisons.loc[same,"mean_total_seconds"]/base_time)
        gap=comparisons.loc[same,"primary_mean"]-float(root.primary_mean)
        comparisons.loc[same,"primary_gap_vs_classic_rf"]=gap
        comparisons.loc[same,"primary_advantage_vs_classic_rf"]=(
            -gap if root.primary_metric in MINIMIZE else gap)
    comparisons.to_csv(RESULTS/"sampling_comparison.csv",index=False)
    # Generated from the same score table, never a manually copied timing report.
    t=sm.groupby("model",as_index=False).agg(
        selection=("mean_selection_seconds","mean"),
        final_fit=("mean_final_fit_seconds","mean"),
        prediction=("mean_predict_seconds","mean"),
        total=("mean_total_seconds","mean"))
    t=t.sort_values(["total","model"])
    timing=[
        "# Computation time: NeuralForest and forest baselines",
        "",
        "Observed wall-clock seconds averaged across 19 scenarios and ten seeds.",
        "Numerical libraries were restricted to one CPU thread. Different runners",
        "may have distinct performance. Times do not include data generation",
        "or dependency installation. Internal search budgets differ by estimator.",
        "",
        "| Algorithm | Selection (s) | Final fit (s) | Prediction (s) | Total (s) |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for item in t.itertuples():
        timing.append(f"| {item.model} | {fmt(item.selection)} | "
          f"{fmt(item.final_fit)} | {fmt(item.prediction,4)} | {fmt(item.total)} |")
    timing.extend(["",
      "## Comparisons by scenario",
      "",
      "A full per-scenario table of the standard RF, 50% and 80% row",
      "subsampling, class-balanced under-sampling, ranger and informed",
      "NeuralForest variants is in sampling_comparison.csv. Both the",
      "primary objective gap and the total-time ratio versus RandomForest",
      "are stored. Positive primary_advantage_vs_classic_rf is favorable.",
      "",
      "All 5,890 per-seed score and timing records appear in all_runs.csv;",
      "the cross-seed means and standard deviations are in summary.csv.",
      "The training budget and the initial predictive objective are",
      "not perfectly matched across algorithm families."])
    (RESULTS/"COMPUTE_TIME.md").write_text("\n".join(timing)+"\n",encoding="utf8")
    # Historical comparisons are based only on seeds 17 and 43.
    compared=None
    if BASELINE.is_file():
        old=pd.read_csv(BASELINE)
        old=old[old.seed.isin((17,43))]
        new=data[data.seed.isin((17,43))]
        columns=["scenario","seed","model","primary_metric","baseline_value",
                 "updated_value","delta_improvement","baseline_train_seconds",
                 "updated_train_seconds","baseline_predict_seconds",
                 "updated_predict_seconds"]
        records=[]
        oldidx={(r.scenario,int(r.seed),r.model):r for r in old.itertuples()}
        for row in new.itertuples():
            prior=oldidx.get((row.scenario,int(row.seed),row.model))
            if prior is None:continue
            target=row.selected_metric
            oldcol="roc_auc" if target=="auc" else target
            oldscore=float(getattr(prior,oldcol))
            newscore=float(getattr(row,oldcol))
            sign=-1 if target in MINIMIZE else 1
            records.append(dict(zip(columns,[
              row.scenario,int(row.seed),row.model,target,
              oldscore,newscore,sign*(newscore-oldscore),
              float(prior.train_seconds),float(row.train_seconds),
              float(prior.predict_seconds),float(row.predict_seconds)])))
        compared=pd.DataFrame(records,columns=columns)
        compared.to_csv(RESULTS/"paired_v02_comparison.csv",index=False)
    manifest=dict(
        generator="QuentinLamboley/td-machinelearning",
        generator_revision="48c2b28ca7874cbef1f7dde15de962c6e56c5027",
        seeds=list(SEEDS),new_confirmation_seeds=list(NEW_SEEDS),
        scenarios=sorted(data.scenario.unique().tolist()),
        models=sorted(data.model.unique().tolist()),
        rows=len(data),successes=int((data.status=="ok").sum()),
        failures=data.loc[data.status!="ok",[
            "scenario","seed","model","error"]].fillna("").to_dict("records"),
        paired_old_rows=0 if compared is None else len(compared),
        multi_metric_report=multi_manifest,
        python_version=sys.version,platform=platform.platform())
    (RESULTS/"manifest.json").write_text(json.dumps(
        manifest,indent=2,ensure_ascii=False),encoding="utf8")
    lines=[
        "# Vampire benchmark: audited NeuralForest",
        "",
        "All times are measured wall-clock values on GitHub-hosted CPU",
        "runners with mathematical libraries constrained to one thread.",
        "Selection time includes internal hyperparameter and expert search.",
        "Final fit time measures the separately trained final estimator.",
        "Prediction time includes evaluation inference on the held-out set.",
        "The three stages are also available independently in summary.csv.",
        "",
        "The experiment covers 19 scenarios, ten seeds and 31 methods.",
        f"Successful runs: {manifest['successes']} / {len(data)}.",
        "Seeds 401 and 503 are new confirmation seeds absent from earlier v0.5 runs.",
        "",
        "## Overall ranking for each scenario",
        "",
        "| Scenario | Metric | Top three (ten-seed mean) | NeuralForest adaptatif | Rank | Selection (s) | Final fit (s) | Prediction (s) |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for name,g in sm.groupby("scenario"):
        top=g.loc[np.isfinite(g.primary_mean)].sort_values("rank").head(3)
        winners=", ".join(f"{r.model} {fmt(r.primary_mean)}"
                          for r in top.itertuples())
        nf=g[g.model=="NeuralForest_Adaptive"]
        if len(nf):
            row=nf.iloc[0]
            values=(fmt(row.primary_mean),str(int(row["rank"])),
                    fmt(row.mean_selection_seconds,2),
                    fmt(row.mean_final_fit_seconds,2),
                    fmt(row.mean_predict_seconds,3))
        else:
            values=("NA","NA","NA","NA","NA")
        lines.append(f"| {name} | {g.primary_metric.iloc[0]} | {winners} | "
                     f"{values[0]} | {values[1]} | {values[2]} | {values[3]} | {values[4]} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The validation set selects algorithms, model complexity and mixtures",
        "without consulting independent test outcomes. Non-differentiable",
        "metrics use a differentiable proxy in soft trees. Average precision",
        "uses a smooth listwise ranking surrogate, while AUC and Boyce use",
        "a pairwise logistic surrogate. Test performance always uses the",
        "original metric, not its surrogate.",
        "",
        "NeuralForest_Adaptive and NeuralForest_Calibrated include ridge-regularized linear and constant",
        "experts. Its performance must be interpreted as a hybrid ensemble,",
        "not attributed exclusively to learned tree divisions.",
        "",
        "Historical seed-matched v0.2 comparisons are in",
        "paired_v02_comparison.csv. Means over seeds 401 and 503 appear in",
        "new_seed_primary_mean and new_seed_rank in summary.csv.",
        "",
        "Presence-background training negatives are not confirmed human",
        "labels. The independent test population has true synthetic labels,",
        "which changes the interpretation of reported metrics.",
        "",
        "The identical-distribution negative control should give chance",
        "discrimination; a constant predictor often minimizes log loss.",
        "These synthetic datasets do not establish universal superiority.",
        "For 15 metric-specific comparisons and voting/stacking ensembles,",
        "consult METRICS_REPORT.md. The extra metrics are evaluated only",
        "on held-out test sets and were not used to select candidates.",
        "",
        "## Result files",
        "",
        "- all_runs.csv: all metrics, configurations and timed stages",
        "- balanced under-sampling uses equal minority/majority per tree",
        "  without replacement. Each tree uses a training-only class count.",
        "- summary.csv: mean, standard deviation, ranking and phase timings",
        "- paired_v02_comparison.csv: paired original-seed differences",
        "- paired_v03_comparison.csv: 4-seed paired v0.3 versus v0.4 comparison",
        "- paired_v041_comparison.csv: six matched seeds against NeuralForest v0.4.1",
        "- paired_v05_comparison.csv: eight matched seeds against v0.5",
        "- METRICS_REPORT.md: rankings across fifteen independent test metrics",
        "- all_metric_ranks.csv, metric_winners.csv: all metric rankings",
        "- ensemble_comparison.csv: voted and stacked ensemble comparison",
        "- paired_multi_metrics.csv: paired differences and 95% intervals",
        "- average_metric_ranks.csv: rank summaries across scenario families",
        "- sampling_comparison.csv: forest-by-forest subsampling comparison",
        "- COMPUTE_TIME.md: selection, final fit and prediction time by algorithm",
        "- manifest.json: provenance, seeds, failures and environment",
        "- benchmark/README.md: full experimental protocol and limitations"
    ])
    (RESULTS/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines),flush=True)
if __name__=="__main__":
    summarize()
