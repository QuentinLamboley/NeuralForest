"""Aggregate benchmark performance, selection overhead and time to predict."""
from __future__ import annotations
import json
from pathlib import Path
import platform
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
RAW=HERE/"out"
RESULTS=HERE/"results"
BASELINE=ROOT/"audit"/"baseline-v0.2-all_runs.csv"
SEEDS=(17,43,71,97,131,197)
NEW_SEEDS=(131,197)
SCENARIOS=19
MODELS=17
METRICS=("roc_auc","pr_auc","accuracy","balanced_accuracy","precision",
         "recall","specificity","f1","mcc","brier","logloss","boyce")
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
        "The experiment covers 19 scenarios, six seeds and 17 methods.",
        f"Successful runs: {manifest['successes']} / {len(data)}.",
        "Seeds 131 and 197 were not used during the v0.3 experiment.",
        "",
        "## Overall ranking for each scenario",
        "",
        "| Scenario | Metric | Top three (six-seed mean) | NeuralForest calibrated | Rank | Selection (s) | Final fit (s) | Prediction (s) |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for name,g in sm.groupby("scenario"):
        top=g.loc[np.isfinite(g.primary_mean)].sort_values("rank").head(3)
        winners=", ".join(f"{r.model} {fmt(r.primary_mean)}"
                          for r in top.itertuples())
        nf=g[g.model=="NeuralForest_Calibrated"]
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
        "paired_v02_comparison.csv. Means over seeds 131 and 197 appear in",
        "new_seed_primary_mean and new_seed_rank in summary.csv.",
        "",
        "Presence-background training negatives are not confirmed human",
        "labels. The independent test population has true synthetic labels,",
        "which changes the interpretation of reported metrics.",
        "",
        "The identical-distribution negative control should give chance",
        "discrimination; a constant predictor often minimizes log loss.",
        "These synthetic datasets do not establish universal superiority.",
        "",
        "## Result files",
        "",
        "- all_runs.csv: all metrics, configurations and timed stages",
        "- summary.csv: mean, standard deviation, ranking and phase timings",
        "- paired_v02_comparison.csv: paired original-seed differences",
        "- paired_v03_comparison.csv: 4-seed paired v0.3 versus v0.4 comparison",
        "- manifest.json: provenance, seeds, failures and environment",
        "- benchmark/README.md: full experimental protocol and limitations"
    ])
    (RESULTS/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines),flush=True)
if __name__=="__main__":
    summarize()
