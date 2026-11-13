"""Independent-test multi-metric comparison and paired uncertainty reports.

Each model is fitted using the scenario's *predefined* primary objective.
Other metrics are test-set diagnostics, NOT alternative training/selection
targets. Never select a model by browsing these test-set rankings.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

METRICS=("roc_auc","pr_auc","balanced_accuracy","f1","mcc",
         "precision","recall","specificity","brier","logloss",
         "boyce","boyce_rank","ece","precision_at_10pct",
         "recall_at_10pct")
MINIMIZE={"brier","logloss","ece"}
FEATURED=("roc_auc","pr_auc","balanced_accuracy","f1",
          "brier","logloss","boyce","boyce_rank","ece",
          "recall_at_10pct")
REFERENCE_MODELS=(
    "RandomForest","RF_Bootstrap50","RF_Subsample50",
    "RF_BalancedUndersampled","RF_BalancedUndersampledPrior",
    "NeuralForest_CART","NeuralForest_UndersampledPlain",
    "NeuralForest_UndersampledInformed","CatBoost","XGBoost",
    "LightGBM","Voting_RF_ExtraTrees","Voting_RF_CatBoost",
    "Voting_Boosting","Voting_Heterogeneous",
    "Stacking_RF_XGB","Stacking_Boosting")
ENSEMBLE_PREFIXES=("Voting_","Stacking_")
def fmt(v,n=3):
    return "NA" if not np.isfinite(v) else f"{v:.{n}f}"
def is_ensemble(name):
    return str(name).startswith(ENSEMBLE_PREFIXES)

def paired_ci(diffs):
    values=np.asarray(diffs,dtype=float)
    values=values[np.isfinite(values)]
    n=len(values)
    if n<2: return (np.nan,np.nan,np.nan,n)
    mean=values.mean()
    se=values.std(ddof=1)/math.sqrt(n)
    q=student_t.ppf(.975,n-1)
    return (mean,mean-q*se,mean+q*se,n)

def build_reports(raw,summary,root):
    expected=set(METRICS)
    if not expected.issubset(raw.columns) or not expected.issubset(
        {k[:-5] for k in summary.columns if k.endswith("_mean")}):
        raise ValueError("Missing requested metric in raw or summary file")
    rankings=[]
    winners=[]
    all_models=sorted(raw.model.unique())
    for scenario,group in summary.groupby("scenario"):
        for metric in METRICS:
            vals=group[[ "model",metric+"_mean",metric+"_sd"]].copy()
            vals.columns=["model","mean","sd"]
            vals["scenario"]=scenario
            vals["metric"]=metric
            vals["direction"]="lower" if metric in MINIMIZE else "higher"
            vals["rank"]=vals["mean"].rank(
                method="min",ascending=(metric in MINIMIZE),
                na_option="bottom")
            if not np.isfinite(vals["mean"]).any():
                continue
            vals["model_family"]=np.where(
                vals.model.map(is_ensemble),"voting_or_stacking",
                np.where(vals.model.str.startswith("NeuralForest"),
                         "neural_forest","other"))
            rankings.extend(vals.to_dict("records"))
            ranked=vals[np.isfinite(vals["mean"])].sort_values(
                ["rank","model"])
            first=ranked.iloc[0]
            nf=ranked[ranked.model=="NeuralForest_Adaptive"]
            base=ranked[ranked.model=="RandomForest"]
            ensemble=ranked[ranked.model.map(is_ensemble)]
            win=dict(scenario=scenario,metric=metric,
                direction="minimize" if metric in MINIMIZE else "maximize",
                winner=first.model,winner_mean=float(first["mean"]),
                top_three="; ".join(
                    f"{r.model}: {fmt(r['mean'],4)}"
                    for _,r in ranked.head(3).iterrows()),
                nf_adaptive_mean=float(nf.iloc[0]["mean"]),
                nf_adaptive_rank=int(nf.iloc[0]["rank"]),
                classical_rf_mean=float(base.iloc[0]["mean"]),
                classical_rf_rank=int(base.iloc[0]["rank"]))
            if len(ensemble):
                win["best_ensemble"]=ensemble.iloc[0].model
                win["best_ensemble_mean"]=float(ensemble.iloc[0]["mean"])
                win["best_ensemble_rank"]=int(ensemble.iloc[0]["rank"])
            winners.append(win)
    ranks=pd.DataFrame(rankings)
    winners=pd.DataFrame(winners)
    ranks.sort_values(["scenario","metric","rank","model"]).to_csv(
        root/"all_metric_ranks.csv",index=False)
    winners.sort_values(["scenario","metric"]).to_csv(
        root/"metric_winners.csv",index=False)
    subset=summary[
        summary.model.map(is_ensemble) |
        summary.model.isin(("NeuralForest_Adaptive","NeuralForest_CART",
            "RandomForest","RF_BalancedUndersampled",
            "CatBoost","XGBoost","LightGBM"))]
    subset.to_csv(root/"ensemble_comparison.csv",index=False)

    # Paired differences in every test metric, matched within scenario
    # and generator seed. Positive always means NF Adaptive did better.
    comparison=[]
    for (scenario,metric),sub in raw.groupby(["scenario","selected_metric"]):
        candidate=sub[sub.model=="NeuralForest_Adaptive"].set_index("seed")
        if len(candidate)!=len(sub.seed.unique()):
            raise ValueError(f"Missing NeuralForest adaptive for {scenario}")
        for baseline in REFERENCE_MODELS:
            comp=sub[sub.model==baseline].set_index("seed")
            if comp.empty: continue
            shared=candidate.index.intersection(comp.index)
            for measure in METRICS:
                delta=(candidate.loc[shared,measure].to_numpy(dtype=float)-
                       comp.loc[shared,measure].to_numpy(dtype=float))
                if measure in MINIMIZE: delta=-delta
                mean,lo,hi,n=paired_ci(delta)
                comparison.append(dict(scenario=scenario,
                    scenario_selection_metric=metric,
                    compared_metric=measure,baseline=baseline,
                    nf_model="NeuralForest_Adaptive",n=n,
                    improvement_mean=mean,ci95_low=lo,ci95_high=hi,
                    nf_mean=candidate.loc[shared,measure].mean(),
                    baseline_mean=comp.loc[shared,measure].mean(),
                    ci95_excludes_zero=bool(n>=2 and
                         np.isfinite(lo) and (lo>0 or hi<0))))
    pairs=pd.DataFrame(comparison)
    pairs.to_csv(root/"paired_multi_metrics.csv",index=False)
    expected_pairs=19*len(REFERENCE_MODELS)*len(METRICS)
    if len(pairs)!=expected_pairs:
        raise RuntimeError(f"Expected {expected_pairs} paired rows, got {len(pairs)}")

    # Aggregate ranks only, not scores: scores of different metrics and
    # different scenario data-generating processes cannot be pooled.
    ranks_by_model=(ranks.groupby(["model","metric"]).agg(
        mean_rank=("rank","mean"),
        first_places=("rank",lambda x:int((x==1).sum())),
        scenarios=("scenario","nunique")).reset_index())
    ranks_by_model.to_csv(root/"average_metric_ranks.csv",index=False)
    m=summary.groupby("model").agg(
        mean_primary_rank=("rank","mean"),
        first_primary=("rank",lambda x:int((x==1).sum())),
        scenarios=("scenario","nunique"),
        mean_total_seconds=("mean_total_seconds","mean")).sort_values(
        ["mean_primary_rank","mean_total_seconds"])
    m.to_csv(root/"primary_rank_summary.csv")
    presence_only=m[m.scenarios<19]
    # Mean ranks across different scenario sets must never be mistaken
    # for directly comparable overall rankings.
    if not presence_only.empty:
        presence_only.to_csv(root/"pu_specific_ranks.csv")

    lines=[
      "# Performance by metric: complete ensemble comparison","",
      "Model selection and hyperparameter search used each scenario's",
      "predefined primary objective, **not** these test-set diagnostics.",
      "The independent outcomes for 19 scenarios, 10 seeds and all models",
      "are in all_runs.csv. Rankings are provided separately for all",
      f"{len(METRICS)} metrics, including proper scores, discrimination,",
      "fixed-threshold classification, top-decile retrieval, and Boyce.",
      "",
      "## Metric-level comparisons",
      "",
      "The following values average **ranks**, not incompatible numeric",
      "scores, across all 19 scenarios. PU-specific methods are excluded",
      "from overall mean-rank comparisons because they run on only three",
      "presence/background scenarios. Lower average rank is favorable;",
      "first-place counts are descriptive. These are not significance",
      "tests and do not establish universal algorithm dominance.",
      "",
      "| Metric | Best mean-rank method | Average rank |",
      "| --- | --- | ---: |"]
    for measure in METRICS:
        # PU-specific methods have only three applicable scenarios; do not
        # compare their three-case rank to a 19-case full-domain mean rank.
        view=ranks_by_model[(ranks_by_model.metric==measure)&
          (ranks_by_model.scenarios==19)].sort_values(
            ["mean_rank","model"])
        first=view.iloc[0]
        lines.append(f"| {measure} | {first.model} | {fmt(first.mean_rank,2)} |")
    lines += ["","## NeuralForest versus voted and stacked models","",
      "The table below compares representative families on every",
      "scenario's predefined metric, including actual mean score",
      "and the three measured compute stages. It does not imply the",
      "same hyperparameter budget across models.",
      "",
      "| Scenario | Primary metric | NF adaptive | Best ensemble | Ensemble score | Classical RF |",
      "| --- | --- | ---: | --- | ---: | ---: |"]
    for scenario,group in summary.groupby("scenario"):
        metric=group.primary_metric.iloc[0]
        nf=group[group.model=="NeuralForest_Adaptive"].iloc[0]
        rf=group[group.model=="RandomForest"].iloc[0]
        ensemble=group[group.model.map(is_ensemble)].sort_values(
            "primary_mean",ascending=metric in MINIMIZE)
        if ensemble.empty:continue
        top=ensemble.iloc[0]
        lines.append(
            f"| {scenario} | {metric} | {fmt(nf.primary_mean,4)} | "
            f"{top.model} | {fmt(top.primary_mean,4)} | "
            f"{fmt(rf.primary_mean,4)} |")
    lines += ["", "## Uncertainty and interpretation","",
      "paired_multi_metrics.csv contains all matched-seed differences",
      "between NeuralForest Adaptive and reference algorithms, with",
      "exploratory t-based 95% intervals. Positive improvement_mean",
      "favors NeuralForest. With thousands of model-metric-scenario",
      "comparisons, unadjusted intervals do not control multiplicity.",
      "Extreme class rarity can make AP and top-decile retrieval volatile.",
      "",
      "Equal-width-bin Boyce is sensitive to monotone score rescaling.",
      "boyce_rank first transforms all predictions into their ranks and",
      "is invariant to strictly monotone calibration without ties.",
      "It is a robustness diagnostic, not a standard replacement for",
      "the published continuous Boyce index. The numeric scale and",
      "ecological meaning of a habitat suitability score still matter.",
      "",
      "ECE is a descriptive calibration diagnostic and is not a proper",
      "scoring rule. Brier and log-loss are separately reported.",
      "Threshold metrics use the same fixed 0.5 cutoff for every",
      "algorithm and may unfairly penalize deliberately class-balanced",
      "or presence/background models. The top-decile retrieval metrics",
      "are threshold-independent with respect to probability scale.",
      "",
      "The simulated background labels are not verified negatives.",
      "Optimizing a Boyce value on pseudo-labels can disagree with",
      "the independent true-label target. No model is guaranteed to",
      "be optimal for every possible data-generating distribution.",
      "",
      "Presence-only nnPU sensitivity results are separately summarized",
      "in EXTERNAL_VALIDATION.md and pu_specific_ranks.csv.",
      "",
      "Files: all_metric_ranks.csv, metric_winners.csv,",
      "ensemble_comparison.csv, paired_multi_metrics.csv,",
      "average_metric_ranks.csv and primary_rank_summary.csv."]
    (root/"METRICS_REPORT.md").write_text(
        "\n".join(lines)+"\n",encoding="utf8")
    return dict(rows=len(ranks),pair_rows=len(pairs),
                metrics=len(METRICS),ensembles=len([x for x in all_models
                              if is_ensemble(x)]))
