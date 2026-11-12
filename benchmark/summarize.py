"""Aggregate per-seed scores, time and scenario rankings."""
from __future__ import annotations
import json
from pathlib import Path
import platform
import sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
RAW=HERE/"out"
RESULTS=HERE/"results"
METRICS=("roc_auc","pr_auc","accuracy","balanced_accuracy","precision",
         "recall","specificity","f1","mcc","brier","logloss","boyce")
MINIMIZE={"brier","logloss"}

def fmt(x, n=3):
    return "NA" if not np.isfinite(x) else f"{x:.{n}f}"

def build():
    paths=sorted(RAW.glob("*.csv"))
    if not paths: raise RuntimeError("No results were generated")
    data=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    expected=32
    distinct=data[["scenario","seed"]].drop_duplicates()
    if len(distinct)!=expected:
        raise RuntimeError(f"Expected {expected} scenario-seed pairs, found {len(distinct)}")
    RESULTS.mkdir(parents=True,exist_ok=True)
    data.to_csv(RESULTS/"all_runs.csv",index=False)
    summary=[]
    for (scenario,model),g in data.groupby(["scenario","model"],sort=True):
        target=g.selected_metric.iloc[0]
        metric_column="roc_auc" if target=="auc" else target
        v=g[metric_column]
        ok=v[np.isfinite(v)]
        row=dict(scenario=scenario,model=model,primary_metric=target,
           primary_mean=ok.mean() if len(ok) else np.nan,
           primary_sd=ok.std() if len(ok)>1 else np.nan,
           n_seeds=len(g),n_success=int((g.status=="ok").sum()),
           mean_train_seconds=g.train_seconds.mean(),
           mean_predict_seconds=g.predict_seconds.mean(),
           mean_total_seconds=g.total_seconds.mean())
        for m in METRICS:
            row[m+"_mean"]=g[m].mean()
            row[m+"_sd"]=g[m].std()
        summary.append(row)
    sm=pd.DataFrame(summary)
    sm["rank"]=np.nan
    for scenario,g in sm.groupby("scenario"):
        direction=g.primary_metric.iloc[0] in MINIMIZE
        sm.loc[g.index,"rank"]=g.primary_mean.rank(
            method="min",ascending=direction,na_option="bottom")
    sm=sm.sort_values(["scenario","rank","model"])
    sm.to_csv(RESULTS/"summary.csv",index=False)
    manifest=dict(
        generator="QuentinLamboley/td-machinelearning",
        generator_revision="48c2b28ca7874cbef1f7dde15de962c6e56c5027",
        seeds=[17,43],
        scenarios=sorted(data.scenario.unique().tolist()),
        models=sorted(data.model.unique().tolist()),
        rows=len(data),successes=int((data.status=="ok").sum()),
        failures=data[data.status!="ok"][
            ["scenario","seed","model","error"]].fillna("").to_dict("records"),
        python_version=sys.version,platform=platform.platform())
    (RESULTS/"manifest.json").write_text(
        json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    lines=[
        "# Vampire benchmark results","","All results come from real model fits to",
        "independently generated training and test data. Two seeds were",
        "evaluated for each scenario. Runtime is measured on GitHub-hosted",
        "CPUs and includes internal selection and model fitting.",
        "",
        "## Model rankings by scenario","",
        "| Scenario | Selection metric | Three leading models | NeuralForest differential (score, rank) | Training + prediction (s) |",
        "| --- | --- | --- | --- | --- |"]
    for case,g in sm.groupby("scenario"):
        top=g.loc[np.isfinite(g.primary_mean)].sort_values("rank").head(3)
        winners=", ".join(f"{r.model}: {fmt(r.primary_mean)}"
                          for r in top.itertuples())
        nf=g[g.model=="NeuralForest_Differentiable"]
        nfs="NA";secs="NA"
        if len(nf):
            row=nf.iloc[0]
            nfs=f"{fmt(row.primary_mean)}, #{int(row['rank'])}"
            secs=fmt(row.mean_total_seconds,2)
        lines.append(f"| {case} | {g.primary_metric.iloc[0]} | {winners} | {nfs} | {secs} |")
    lines.extend([
       "","## Reading the results","",
       "The full score matrix (12 metrics, training time and prediction time)",
       "is in all_runs.csv. summary.csv has mean, standard deviation and rank",
       "for every combination of scenario and algorithm.",
       "",
       "Non-differentiable model-selection metrics (AUC, average precision,",
       "Boyce and F1) have proxy gradient losses in NeuralForest. The",
       "requested metric is nevertheless used for independent internal",
       "selection and external testing. A proxy loss is not its exact gradient.",
       "",
       "For presence-background designs, training negatives are unlabeled",
       "background observations, potentially containing vampires.",
       "True class labels are only used in the independent test population.",
       "The labels in the presence-background training task differ from",
       "true human-vampire labels and require separate interpretation.",
       "",
       "The 100% similarity condition has indistinguishable class-specific",
       "covariate distributions and should yield chance discrimination.",
       "A lack of advantage over simpler methods should be reported as such.",
       "",
       "Failures remain in the CSV and manifest rather than being excluded.",
       "All time comparisons are specific to this hardware and workload.",
       "See benchmark/README.md for caveats and exact configuration."]
    (RESULTS/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines),flush=True)
    print(f"Rows {len(data)}, successes {manifest['successes']}",flush=True)

if __name__=="__main__":build()
