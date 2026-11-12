"""Two versus three guided neural rounds on untouched simulator seeds."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"benchmark"))
from design import SCENARIOS,create_data
from run import metric_table,objective

SEEDS=(227,311)
OUT=ROOT/"audit"/"rounds_results.csv"
REPORT=ROOT/"audit"/"rounds_report.md"
LOWER={"brier","logloss"}

def run_one(scenario,seed):
    train,test,metadata=create_data(scenario,seed)
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)
        train.to_csv(path/"train.csv",index=False)
        test.to_csv(path/"test.csv",index=False)
        subprocess.run(["Rscript",str(ROOT/"audit"/"rounds_fit.R"),
                        str(path/"train.csv"),str(path/"test.csv"),
                        str(path/"output.json"),scenario.target_metric,str(seed)],
                       cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
        loaded=json.loads((path/"output.json").read_text(encoding="utf8"))
    rows=[]
    y=test["Vampirisme"].to_numpy(dtype=int)
    for m in loaded["models"]:
        p=np.asarray(m["probability"],dtype=float)
        if len(p)!=len(y) or not np.isfinite(p).all():
            raise RuntimeError("Invalid prediction length or probability")
        score=metric_table(y,p)
        rows.append(dict(
            scenario=scenario.key,seed=seed,rounds=m["rounds"],
            selected_round=m["selected_round"],
            primary_metric=scenario.target_metric,
            primary_score=objective(score,scenario.target_metric),
            train_n=len(train),test_n=len(test),
            selection_seconds=m["selection_seconds"],
            final_fit_seconds=m["final_fit_seconds"],
            predict_seconds=m["predict_seconds"],
            total_seconds=m["train_seconds"]+m["predict_seconds"],
            expert=m["expert"],**score))
    return rows

def main():
    allrows=[]
    for specs in SCENARIOS.values():
      for spec in specs:
        for seed in SEEDS:
          observations=run_one(spec,seed)
          if len(observations)!=2:raise RuntimeError("Missing round ablation")
          allrows.extend(observations)
          print(f"{spec.key}/{seed}: 2 rounds {observations[0]['primary_score']:.4f}"
                f" vs 3 rounds {observations[1]['primary_score']:.4f}",flush=True)
    if len(allrows)!=19*2*2:raise RuntimeError("Wrong ablation count")
    frame=pd.DataFrame(allrows)
    frame.to_csv(OUT,index=False)
    lines=[
        "# Neural round count ablation",
        "",
        "The 19 original scenarios were independently generated on seeds 227",
        "and 311, which were not evaluated by the previous benchmark.",
        "Two and three guided forest rounds use the same model structure",
        "and the same training and test observations. The internal holdout",
        "still selects the best round and expert without test labels.",
        "An extra round may be rejected; more compute is not guaranteed",
        "to improve generalization.",
        "",
        "| Scenario | Metric | Two rounds | Three rounds | Three minus two, favorable direction | Extra total time (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    improvements=[]
    for scenario,block in frame.groupby("scenario"):
        a=block[block.rounds==2]
        b=block[block.rounds==3]
        metric=str(a.primary_metric.iloc[0])
        gain=b.primary_score.mean()-a.primary_score.mean()
        if metric in LOWER:gain=-gain
        duration=b.total_seconds.mean()-a.total_seconds.mean()
        improvements.append(gain)
        lines.append(
            f"| {scenario} | {metric} | {a.primary_score.mean():.4f} |"
            f" {b.primary_score.mean():.4f} | {gain:+.4f} | {duration:+.3f} |")
    positive=sum(v>1e-8 for v in improvements)
    negative=sum(v< -1e-8 for v in improvements)
    ties=len(improvements)-positive-negative
    lines.extend([
       "",
       f"Three rounds improved {positive} scenarios, worsened {negative},",
       f"and tied {ties} at this precision on two new seeds.",
       "",
       "## Interpretation",
       "",
       "The core six-seed benchmark with 17 algorithms remains the primary",
       "cross-method comparison. This targeted ablation assesses whether",
       "the transported warm-start mechanism is exercised by an additional",
       "neural round, and records its separate computational overhead.",
       "The difference cannot be attributed exclusively to weight transport",
       "because the number of training rounds also changes.",
       "Do not infer general superiority from this synthetic two-seed ablation.",
       "",
       "The raw score vector and selection, final-fit and prediction",
       "durations for each case are available in rounds_results.csv."])
    REPORT.write_text("\n".join(lines)+"\n",encoding="utf8")
    print(f"Recorded {len(frame)} held-out evaluations.",flush=True)

if __name__=="__main__":
    main()
